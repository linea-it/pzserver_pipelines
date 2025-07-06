# combine_redshift_dedup/packages/specz.py

import os
import glob
import json
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
    Load, rename, and translate columns of a catalog. Apply tie-breaking for duplicate RA/DEC pairs.
    Generate unique CRD_IDs. Save the processed Parquet catalog for later steps.

    Args:
        entry (dict): Catalog configuration entry.
        translation_config (dict): Config with translation rules and delta_z_threshold.
        temp_dir (str): Path to temp directory for output files.
        compared_to_dict (defaultdict): Shared dict to store compared pairs (RA/DEC duplicates).
        combine_type (str): Type of combine (default is "concatenate_and_mark_duplicates").

    Returns:
        tuple: (output_path, ra_col_name, dec_col_name, internal_catalog_name)
    """
    # Load input catalog
    ph = ProductHandle(entry["path"])
    df = ph.to_ddf()
    product_name = entry["internal_name"]

    # Rename columns according to the config mapping
    col_map = {v: k for k, v in entry["columns"].items() if v and v in df.columns}
    df = df.rename(columns=col_map)

    # Add a source column to identify the catalog of origin
    df["source"] = product_name

    # Ensure all standard columns exist (create NaN columns if missing)
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

    sizes = df.map_partitions(len).compute().tolist()
    offsets = [last_counter]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)

    def generate_crd_id(partition, start):
        n = len(partition)
        partition = partition.copy()
        partition["CRD_ID"] = [f"CRD{start + i + 1}" for i in range(n)]
        return partition

    dfs = []
    for i, offset in enumerate(offsets):
        dfs.append(df.get_partition(i).map_partitions(generate_crd_id, offset, meta=df._meta.assign(CRD_ID=str)))

    df = dd.concat(dfs)

    new_last_counter = last_counter + sum(sizes)
    with open(counter_path, "w") as f:
        json.dump({"last_id": new_last_counter}, f)

    df["tie_result"] = 1
    df["z_flag_homogenized"] = np.nan
    df["type_homogenized"] = np.nan

    # === Deduplication logic (only if combine_type requests duplicate marking) ===
    if combine_type in ["concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"]:
        delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)

        valid_coord_mask = df["ra"].notnull() & df["dec"].notnull()
        df_valid_coord = df[valid_coord_mask]
        df_valid_coord["ra_dec_key"] = df_valid_coord["ra"].round(6).astype(str) + "_" + df_valid_coord["dec"].round(6).astype(str)

        dup_counts = df_valid_coord.groupby("ra_dec_key").size().compute()
        keys_dup = dup_counts[dup_counts > 1].index.tolist()

        if keys_dup:
            df_dup_local = df_valid_coord[df_valid_coord["ra_dec_key"].isin(keys_dup)].compute()

            for key in keys_dup:
                group = df_dup_local[df_dup_local["ra_dec_key"] == key]
                ids = group["CRD_ID"].tolist()
                for i, id1 in enumerate(ids):
                    for id2 in ids[i + 1:]:
                        compared_to_dict.setdefault(id1, []).append(id2)
                        compared_to_dict.setdefault(id2, []).append(id1)

            if delta_z_threshold > 0:
                delta_status = (
                    df_dup_local.groupby("ra_dec_key")["z"]
                    .apply(lambda z: "diff" if abs(z.max() - z.min()) > delta_z_threshold else "same")
                )

                keys_same = delta_status[delta_status == "same"].index.tolist()
                keys_diff = delta_status[delta_status == "diff"].index.tolist()

                def mark_tie_result(partition):
                    partition = partition.copy()
                    partition["ra_dec_key"] = partition["ra"].round(6).astype(str) + "_" + partition["dec"].round(6).astype(str)

                    dup_same = partition["ra_dec_key"].isin(keys_same)
                    dup_diff = partition["ra_dec_key"].isin(keys_diff)

                    partition["dup_rank"] = partition.groupby("ra_dec_key").cumcount()

                    same_mask = dup_same & (partition["dup_rank"] > 0)

                    partition.loc[dup_diff, "tie_result"] = 2
                    partition.loc[same_mask, "tie_result"] = 0

                    partition = partition.drop(columns=["dup_rank", "ra_dec_key"])

                    return partition
            else:
                def mark_tie_result(partition):
                    partition = partition.copy()
                    partition["ra_dec_key"] = partition["ra"].round(6).astype(str) + "_" + partition["dec"].round(6).astype(str)
                    dup_mask = partition["ra_dec_key"].isin(keys_dup)
                    partition.loc[dup_mask, "tie_result"] = 2
                    partition = partition.drop(columns=["ra_dec_key"])
                    return partition

            df = df.map_partitions(mark_tie_result)

            compared_to_path = os.path.join(temp_dir, "compared_to.json")
            with open(compared_to_path, "w") as f:
                json.dump(compared_to_dict, f)

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

    df["z_flag_homogenized"] = df.map_partitions(
        lambda p: p.apply(lambda row: apply_translation(row, "z_flag"), axis=1),
        meta=("z_flag_homogenized", "f8")
    )
    df["type_homogenized"] = df.map_partitions(
        lambda p: p.apply(lambda row: apply_translation(row, "type"), axis=1),
        meta=("type_homogenized", "object")
    )

    # === Final check for homogenized columns ===
    if combine_type in ["concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"]:
        zflag_nan = df["z_flag_homogenized"].isna().all().compute()
        type_nan = df["type_homogenized"].isna().all().compute()
        if zflag_nan and type_nan:
            raise ValueError(
                f"❌ Cannot mark/remove duplicates: catalog '{product_name}' has both z_flag_homogenized and type_homogenized fully NaN."
            )

    # === Add is_in_ComCam_ECDFS_field column ===
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

    df = df.map_partitions(compute_in_cone, meta=df._meta.assign(is_in_ComCam_ECDFS_field=np.int64()))

    final_cols = [
        "CRD_ID", "id", "ra", "dec", "z", "z_flag", "z_err", "type", "survey",
        "source", "tie_result", "z_flag_homogenized", "type_homogenized",
        "is_in_ComCam_ECDFS_field"
    ]
    df = df[final_cols]

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