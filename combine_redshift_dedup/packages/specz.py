# combine_redshift_dedup/packages/specz.py

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import dask.dataframe as dd

from hats_import.pipeline import ImportArguments, pipeline_with_client
from hats_import.catalog.file_readers import ParquetReader
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
import hats

from combine_redshift_dedup.packages.product_handle import ProductHandle

def prepare_catalog(entry, translation_config, temp_dir, compared_to_dict, combine_type="concatenate_and_mark_duplicates"):
    """
    Load, standardize, and prepare a redshift catalog for combination.

    This function:
    - Renames columns according to configuration.
    - Generates unique CRD_IDs for all rows.
    - Applies translation rules for z_flag and type homogenization.
    - Identifies and marks duplicates within the catalog (RA/DEC duplicates).
    - Applies tie-breaking logic, including delta_z threshold comparison.
    - Flags objects inside ComCam ECDFS field.
    - Saves the processed catalog as Parquet.

    Args:
        entry (dict): Catalog configuration entry.
        translation_config (dict): Config with translation rules and thresholds.
        temp_dir (str): Temporary output directory.
        compared_to_dict (defaultdict): Tracks pairs of compared objects.
        combine_type (str): Combine mode (concatenate / mark / remove duplicates).

    Returns:
        tuple: (output_path, ra_col_name, dec_col_name, internal_catalog_name)
    """
    # === Load catalog and rename columns according to configuration ===
    ph = ProductHandle(entry["path"])
    df = ph.to_ddf()
    product_name = entry["internal_name"]

    col_map = {v: k for k, v in entry["columns"].items() if v and v in df.columns}
    df = df.rename(columns=col_map)
    df["source"] = product_name

    # Ensure required columns exist, fill with NaN if missing
    for col in ["id", "ra", "dec", "z", "z_flag", "z_err", "type", "survey"]:
        if col not in df.columns:
            df[col] = np.nan

    # Standardize column types
    df["id"] = df["id"].astype(str)
    df["ra"] = df["ra"].astype(float)
    df["dec"] = df["dec"].astype(float)
    df["z"] = df["z"].astype(float)
    df["z_flag"] = df["z_flag"].astype(str)
    df["z_err"] = df["z_err"].astype(float)
    df["type"] = df["type"].astype(str)
    df["survey"] = df["survey"].astype(str).str.strip().str.upper()

    # === Generate unique CRD_IDs ===
    counter_path = os.path.join(temp_dir, "crd_id_counter.json")
    if os.path.exists(counter_path):
        with open(counter_path, "r") as f:
            last_counter = json.load(f).get("last_id", 0)
    else:
        last_counter = 0

    # Determine partition sizes and offsets for ID generation
    sizes = df.map_partitions(len).compute().tolist()
    offsets = [last_counter]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)

    # Assign CRD_ID to each row
    def generate_crd_id(partition, start):
        n = len(partition)
        partition = partition.copy()
        partition["CRD_ID"] = [f"CRD{start + i + 1}" for i in range(n)]
        return partition

    dfs = [
        df.get_partition(i).map_partitions(
            generate_crd_id, offset,
            meta=df._meta.assign(CRD_ID=str)
        )
        for i, offset in enumerate(offsets)
    ]
    df = dd.concat(dfs)

    with open(counter_path, "w") as f:
        json.dump({"last_id": last_counter + sum(sizes)}, f)

    # Initialize tie_result to 1 (default: kept)
    df["tie_result"] = 1

    # === Apply translation rules for z_flag and type ===
    translation_rules_uc = {
        survey_name.upper(): rules
        for survey_name, rules in translation_config.get("translation_rules", {}).items()
    }

    def apply_translation(row, key):
        survey = row.get("survey")
        rules = translation_rules_uc.get(survey, {})
        rule = rules.get(f"{key}_translation", {})
        val = row.get(key)

        if val in rule:
            return rule[val]
        if "conditions" in rule:
            for cond in rule["conditions"]:
                try:
                    if eval(cond["expr"], {}, row.to_dict()):
                        return cond["value"]
                except Exception:
                    continue
        if "default" in rule:
            return rule["default"]
        return np.nan

    # Compute homogenized z_flag and type
    df["z_flag_homogenized"] = df.map_partitions(
        lambda p: p.apply(lambda row: apply_translation(row, "z_flag"), axis=1),
        meta=("z_flag_homogenized", "f8")
    )
    df["type_homogenized"] = df.map_partitions(
        lambda p: p.apply(lambda row: apply_translation(row, "type"), axis=1),
        meta=("type_homogenized", "object")
    )

    # === Apply tie-breaking and duplicate removal if required ===
    if combine_type in ["concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"]:
        delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)
        tiebreaking_priority = translation_config.get("tiebreaking_priority", [])
        type_priority = translation_config.get("type_priority", {})

        if not tiebreaking_priority:
            warnings.warn(f"tiebreaking_priority is empty for catalog '{product_name}'. Proceeding with delta_z_threshold tie-breaking only.")
        else:
            for col in tiebreaking_priority:
                if col not in df.columns:
                    raise ValueError(
                        f"Tiebreaking column '{col}' is missing in catalog '{product_name}'."
                    )
                if df[col].isna().all().compute():
                    raise ValueError(
                        f"Tiebreaking column '{col}' is invalid in catalog '{product_name}' (all values are NaN)."
                    )
                if col != "type_homogenized":
                    col_dtype = df[col].dtype
                    if not np.issubdtype(col_dtype, np.number):
                        raise ValueError(
                            f"Tiebreaking column '{col}' must be numeric (except for 'type_homogenized'). Found dtype: {col_dtype}"
                        )

        # After this validation, check if there's *any* deduplication criterion
        if not tiebreaking_priority and (delta_z_threshold is None or delta_z_threshold == 0.0):
            raise ValueError(
                f"Cannot deduplicate catalog '{product_name}': tiebreaking_priority is empty and delta_z_threshold is not set or is zero. "
                f"Please provide at least one deduplication criterion."
            )

        # Identify duplicates by RA/DEC rounded to 6 decimals
        valid_coord_mask = df["ra"].notnull() & df["dec"].notnull()
        df_valid_coord = df[valid_coord_mask]
        df_valid_coord["ra_dec_key"] = (
            df_valid_coord["ra"].round(6).astype(str) + "_" +
            df_valid_coord["dec"].round(6).astype(str)
        )

        dup_counts = df_valid_coord.groupby("ra_dec_key").size().compute()
        keys_dup = dup_counts[dup_counts > 1].index.tolist()
        tie_updates = []

        if keys_dup:
            # Compute duplicates locally
            df_dup_local = df_valid_coord[df_valid_coord["ra_dec_key"].isin(keys_dup)].compute()

            for key in keys_dup:
                group = df_dup_local[df_dup_local["ra_dec_key"] == key].copy()
                group["tie_result"] = 0  # All start as eliminated
                surviving = group.copy()

                # Apply tiebreaking priority columns
                for priority_col in tiebreaking_priority:
                    if priority_col == "type_homogenized":
                        surviving["_priority_value"] = surviving["type_homogenized"].map(type_priority).fillna(0)
                    else:
                        surviving["_priority_value"] = surviving[priority_col]

                    if priority_col == "z_flag_homogenized":
                        ids_to_eliminate = surviving.loc[surviving["z_flag_homogenized"] == 6, "CRD_ID"].tolist()
                        group.loc[group["CRD_ID"].isin(ids_to_eliminate), "tie_result"] = 0
                        surviving = surviving[surviving["z_flag_homogenized"] != 6]

                    if surviving.empty:
                        break

                    max_val = surviving["_priority_value"].max()
                    surviving = surviving[surviving["_priority_value"] == max_val]
                    surviving = surviving.drop(columns=["_priority_value"], errors="ignore")
                    if len(surviving) == 1:
                        break

                # Mark surviving objects as tie_result = 2 (potential ties)
                group.loc[group["CRD_ID"].isin(surviving["CRD_ID"]), "tie_result"] = 2

                # Apply delta_z_threshold if needed
                if len(surviving) > 1 and delta_z_threshold > 0:
                    z_vals = surviving["z"].values
                    ids = surviving["CRD_ID"].values
                    delta_z_matrix = np.abs(z_vals[:, None] - z_vals[None, :])
                    remaining_ids = set(ids)

                    for i in range(len(ids)):
                        if ids[i] not in remaining_ids:
                            continue
                        for j in range(i + 1, len(ids)):
                            if ids[j] not in remaining_ids:
                                continue
                            if delta_z_matrix[i, j] <= delta_z_threshold:
                                group.loc[group["CRD_ID"] == ids[i], "tie_result"] = 2
                                group.loc[group["CRD_ID"] == ids[j], "tie_result"] = 0
                                remaining_ids.discard(ids[j])

                # Final clean-up of tie_result
                survivors = group[group["tie_result"] == 2]
                if len(survivors) == 1:
                    group.loc[group["tie_result"] == 2, "tie_result"] = 1
                elif len(survivors) == 0:
                    non_eliminated = group[group["tie_result"] != 0]
                    if len(non_eliminated) == 1:
                        group.loc[group["CRD_ID"] == non_eliminated.iloc[0]["CRD_ID"], "tie_result"] = 1

                tie_updates.append(group[["CRD_ID", "tie_result"]].copy())

                # Update compared_to_dict for reporting
                ids_all = group["CRD_ID"].tolist()
                for i, id1 in enumerate(ids_all):
                    for id2 in ids_all[i + 1:]:
                        compared_to_dict.setdefault(id1, []).append(id2)
                        compared_to_dict.setdefault(id2, []).append(id1)

            # Merge tie updates back to main dataframe
            if tie_updates:
                combined_update = pd.concat(tie_updates)
                tie_update_dd = dd.from_pandas(combined_update, npartitions=1)
                df = df.drop("tie_result", axis=1).merge(
                    tie_update_dd, on="CRD_ID", how="left"
                ).fillna({"tie_result": 1})

            # Save compared_to_dict for debugging / later use
            with open(os.path.join(temp_dir, "compared_to.json"), "w") as f:
                json.dump(compared_to_dict, f)

    # === Flag objects inside ComCam ECDFS field ===
    ra_cen = 53.13
    dec_cen = -28.10
    radius = 0.72

    def compute_in_cone(partition):
        partition = partition.copy()
        ra_rad = np.deg2rad(partition["ra"])
        dec_rad = np.deg2rad(partition["dec"])
        ra_cen_rad = np.deg2rad(ra_cen)
        dec_cen_rad = np.deg2rad(dec_cen)
        cos_angle = (
            np.sin(dec_cen_rad) * np.sin(dec_rad) +
            np.cos(dec_cen_rad) * np.cos(dec_rad) * np.cos(ra_rad - ra_cen_rad)
        )
        angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))
        partition["is_in_ComCam_ECDFS_field"] = (angle_deg <= radius).astype(int)
        return partition

    df = df.map_partitions(
        compute_in_cone,
        meta=df._meta.assign(is_in_ComCam_ECDFS_field=np.int64())
    )

    # Select and save final output columns
    df = df[
        ["CRD_ID", "id", "ra", "dec", "z", "z_flag", "z_err", "type", "survey",
         "source", "tie_result", "z_flag_homogenized", "type_homogenized",
         "is_in_ComCam_ECDFS_field"]
    ]
    out_path = os.path.join(temp_dir, f"prepared_{product_name}")
    df.to_parquet(out_path, write_index=False)

    return out_path, "ra", "dec", product_name

def import_catalog(path, ra_col, dec_col, artifact_name, output_path, client):
    """
    Import a Parquet catalog into HATS format.

    Args:
        path (str): Path to input Parquet files.
        ra_col (str): RA column name.
        dec_col (str): DEC column name.
        artifact_name (str): Name for the output HATS artifact.
        output_path (str): Directory to save output.
        client (Client): Dask client to run the pipeline.
    """
    if os.path.exists(os.path.join(path, "base")):
        parquet_files = glob.glob(os.path.join(path, "base", "*.parquet"))
    else:
        parquet_files = glob.glob(os.path.join(path, "*.parquet"))

    if len(parquet_files) == 0:
        raise ValueError(f"No Parquet files found at {path}")

    file_reader = ParquetReader()
    args = ImportArguments(
        ra_column=ra_col,
        dec_column=dec_col,
        input_file_list=parquet_files,
        file_reader=file_reader,
        output_artifact_name=artifact_name,
        output_path=output_path,
    )
    pipeline_with_client(args, client)

def generate_margin_cache_safe(hats_path, output_path, artifact_name, client):
    """
    Generate margin cache if partitions > 1; otherwise, skip gracefully.

    Args:
        hats_path (str): Path to HATS catalog.
        output_path (str): Path to save margin cache.
        artifact_name (str): Name of margin cache artifact.
        client (Client): Dask client.

    Returns:
        str or None: Path to margin cache or None if skipped.
    """
    try:
        catalog = hats.read_hats(hats_path)
        info = catalog.partition_info.as_dataframe().astype(int)
        if len(info) > 1:
            args = MarginCacheArguments(
                input_catalog_path=hats_path,
                output_path=output_path,
                margin_threshold=1.0,
                output_artifact_name=artifact_name
            )
            pipeline_with_client(args, client)
            return os.path.join(output_path, artifact_name)
        else:
            print(f"⚠️ Margin cache skipped: single partition for {artifact_name}")
            return None
    except ValueError as e:
        if "Margin cache contains no rows" in str(e):
            print(f"⚠️ {e} Proceeding without margin cache.")
            return None
        else:
            raise