# =====================
# Standard library
# =====================
import os
import re
import ast
import glob
import json
import warnings
import logging
import pathlib
from datetime import datetime
from collections import defaultdict

# =====================
# Third-party libraries
# =====================
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import get_client

# =====================
# Project-specific libraries
# =====================
import lsdb
import hats
from hats_import.pipeline import ImportArguments, pipeline_with_client
from hats_import.catalog.file_readers import ParquetReader
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from combine_redshift_dedup.packages.product_handle import ProductHandle


def prepare_catalog(entry, translation_config, logs_dir, temp_dir, combine_mode="concatenate_and_mark_duplicates"):
    """
    Load, standardize, and prepare a redshift catalog for combination.

    This function:
    - Renames columns according to configuration.
    - Generates unique CRD_IDs for all rows.
    - Applies translation rules for z_flag and instrument_type homogenization.
    - Identifies and marks duplicates within the catalog (RA/DEC duplicates).
    - Applies tie-breaking logic, including delta_z threshold comparison.
    - Flags objects inside ComCam ECDFS field.
    - Saves the processed catalog as Parquet.

    Args:
        entry (dict): Catalog configuration entry.
        translation_config (dict): Config with translation rules and thresholds.
        temp_dir (str): Temporary output directory.
        combine_mode (str): Combine mode (concatenate / mark / remove duplicates).

    Returns:
        tuple: (output_path, ra_col_name, dec_col_name, internal_catalog_name)
    """
    client = get_client()
    
    # === Logger setup ===
    product_name = entry["internal_name"]
    log_path = pathlib.Path(logs_dir) / "prepare_all.log"
    
    logger_prep = logging.getLogger("prepare_catalog_logger")
    logger_prep.setLevel(logging.INFO)
    
    if logger_prep.hasHandlers():
        logger_prep.handlers.clear()
    
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger_prep.addHandler(fh)
    
    logger_prep.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: prepare_catalog id={product_name}")

    # ===================

    # === Load catalog and rename columns according to configuration ===
    ph = ProductHandle(entry["path"])
    df = ph.to_ddf()

    # Apply renaming from YAML
    col_map = {v: k for k, v in entry["columns"].items() if v and v in df.columns}
    df = df.rename(columns=col_map)
    df["source"] = product_name

    # Ensure required columns exist, fill with NaN if missing
    for col in ["id", "ra", "dec", "z", "z_flag", "z_err", "instrument_type", "survey"]:
        if col not in df.columns:
            df[col] = np.nan

    # === Standardize column types ===
    
    # String columns (id, instrument_type, survey)
    for col in ["id", "instrument_type", "survey"]:
        if col in df.columns:
            try:
                df[col] = df[col].fillna("").astype(str)
                if col == "survey":
                    df[col] = df[col].str.strip().str.upper()
            except Exception as e:
                logger_prep.warning(f"⚠️ {product_name} Failed to convert column '{col}' to str: {e}")

    # --- FAST PATH pre-normalization: auxiliary columns ---
    # NOTE: Pre-normalize optional columns used by fast paths (type only here).
    if "type" in df.columns:
        try:
            # Keep lowercase to make membership checks trivial (s/g/p)
            df["type"] = df["type"].fillna("").astype(str).str.strip().str.lower()
        except Exception as e:
            logger_prep.warning(f"⚠️ {product_name} Failed to convert 'type' to str: {e}")

    # === Float columns (ra, dec, z, z_err, z_flag) ===
    for col in ["ra", "dec", "z", "z_err", "z_flag"]:
        if col in df.columns:
            try:
                if col == "z_flag":
                    # --- Special case mappings ---
                    jades_letter_to_score = {
                        "A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "E": 0.0
                    }
                    vimos_flag_to_score = {
                        "LRB_X": 0.0, "MR_X": 0.0,
                        "LRB_B": 1.0, "LRB_C": 1.0, "MR_C": 1.0,
                        "MR_B": 3.0,
                        "LRB_A": 4.0, "MR_A": 4.0
                    }
    
                    def map_special_partition(partition):
                        partition = partition.copy()
                        if "survey" not in partition or "z_flag" not in partition:
                            return partition
    
                        # --- Early exits if z_flag is already numeric ---
                        zf_num = pd.to_numeric(partition["z_flag"], errors="coerce")
                        nonnull = zf_num.notna()
                        if nonnull.any():
                            uniq_vals = pd.unique(zf_num[nonnull])
                            allowed = {0.0, 1.0, 2.0, 3.0, 4.0}
                            # 1) Already in VVDS-inspired discrete {0..4}
                            if set(float(v) for v in uniq_vals if pd.notna(v)).issubset(allowed):
                                partition["z_flag"] = zf_num
                                return partition
                            # 2) Confidence-like [0,1] with fractional values
                            zmin = zf_num[nonnull].min()
                            zmax = zf_num[nonnull].max()
                            has_fractional = ((zf_num > 0.0) & (zf_num < 1.0)).any()
                            if pd.notna(zmin) and pd.notna(zmax) and (zmin >= 0.0) and (zmax <= 1.0) and has_fractional:
                                partition["z_flag"] = zf_num
                                return partition
    
                        # --- Otherwise, treat as string and apply special mappings ---
                        z_flag_str = partition["z_flag"].astype(str)
                        mask_jades = partition["survey"].str.upper() == "JADES"
                        mask_vimos = partition["survey"].str.upper() == "VIMOS"
    
                        def map_jades(val):
                            if pd.isna(val) or val == "":
                                return np.nan
                            return jades_letter_to_score.get(str(val).strip().upper(), np.nan)
    
                        def map_vimos(val):
                            if pd.isna(val) or val == "":
                                return np.nan
                            return vimos_flag_to_score.get(str(val).strip().upper(), np.nan)
    
                        z_flag_str = z_flag_str.where(~mask_jades, z_flag_str.map(map_jades))
                        z_flag_str = z_flag_str.where(~mask_vimos, z_flag_str.map(map_vimos))
    
                        partition["z_flag"] = z_flag_str
                        return partition
    
                    df = df.map_partitions(map_special_partition)
    
                # --- Finally, cast to float ---
                df[col] = df[col].astype(float)
    
            except Exception as e:
                logger_prep.warning(
                    f"⚠️ {product_name} Failed to convert column '{col}' to float: {e}"
                )

    # === Generate unique CRD_IDs using product_name prefix ===
    match = re.match(r"(\d+)_", product_name)
    if not match:
        raise ValueError(f"❌ Could not extract numeric prefix from internal_name '{product_name}'")
    catalog_prefix = match.group(1)  # e.g., "001"

    # Define path and initialize dict for all modes
    compared_to_filename = f"compared_to_dict_{catalog_prefix}.json"
    compared_to_path = os.path.join(temp_dir, compared_to_filename)
    compared_to_dict_solo = defaultdict(set)
    
    # Get partition sizes and offsets
    sizes = df.map_partitions(len).compute().tolist()
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    
    # Assign CRD_IDs: CRD{catalog_prefix}_{i}
    def generate_crd_id(partition, start):
        n = len(partition)
        partition = partition.copy()
        partition["CRD_ID"] = [f"CRD{catalog_prefix}_{start + i + 1}" for i in range(n)]
        return partition
    
    dfs = [
        df.get_partition(i).map_partitions(
            generate_crd_id, offset,
            meta=df._meta.assign(CRD_ID=str)
        )
        for i, offset in enumerate(offsets)
    ]
    df = dd.concat(dfs)

    # Initialize tie_result to 1 (default: kept)
    df["tie_result"] = 1

    # === Apply translation rules for z_flag and instrument_type ===
    translation_rules_uc = {
        survey_name.upper(): rules
        for survey_name, rules in translation_config.get("translation_rules", {}).items()
    }

    def apply_translation(row, key):
        survey = row.get("survey")
        rules = translation_rules_uc.get(survey, {})
        rule = rules.get(f"{key}_translation", {})
        val = row.get(key)
    
        # === Normalize value ===
        if pd.isna(val):
            val_norm = "" if isinstance(val, str) else val
        elif isinstance(val, (int, float, np.number)):
            val_norm = val
        else:
            val_norm = str(val).strip().lower()
    
        # === Normalize rule keys ===
        normalized_rule = {}
        for k, v in rule.items():
            if k in {"conditions", "default"}:
                continue
            if isinstance(val_norm, (int, float, np.number)):
                normalized_rule[k] = v
            else:
                k_norm = str(k).strip().lower()
                normalized_rule[k_norm] = v
    
        # === Try direct match ===
        if val_norm in normalized_rule:
            return normalized_rule[val_norm]
    
        # === Evaluate conditions ===
        if "conditions" in rule:
            context = row.to_dict()
            try:
                for cond in rule["conditions"]:
                    expr = cond.get("expr")
                    if expr and eval(expr, {}, context):
                        return cond["value"]
            except Exception as e:
                raise ValueError(
                    f"Error evaluating condition '{expr}' for survey '{survey}': {e}"
                )
    
        # === Fallback ===
        return rule.get("default", "" if isinstance(val_norm, str) else np.nan)


    # Check and compute homogenized z_flag and instrument_type only if not present
    tiebreaking_priority = translation_config.get("tiebreaking_priority", [])
    instrument_type_priority = translation_config.get("instrument_type_priority", {})
    
    # Only generate homogenized columns if they are required for tie-breaking
    # --- FAST PATHS: try deterministic derivations first; if they fail, fallback to YAML translation ---
    if "z_flag_homogenized" in tiebreaking_priority:
        if "z_flag_homogenized" not in df.columns:

            # Decide if z_flag can be interpreted as quality in [0,1] with at least one fractional value
            def can_use_zflag_as_quality() -> bool:
                if "z_flag" not in df.columns:
                    return False
                try:
                    cnt = df["z_flag"].count()
                    minv = df["z_flag"].min()
                    maxv = df["z_flag"].max()
                    frac_cnt = df[(df["z_flag"] > 0.0) & (df["z_flag"] < 1.0)]["z_flag"].count()
                    cnt, minv, maxv, frac_cnt = dask.compute(cnt, minv, maxv, frac_cnt)  # single round-trip
                    if cnt == 0 or not np.isfinite(minv) or not np.isfinite(maxv):
                        return False
                    if not (minv >= 0.0 and maxv <= 1.0):
                        return False
                    return frac_cnt > 0
                except Exception as e:
                    logger_prep.warning(f"⚠️ {product_name} Could not validate 'z_flag' as quality-like: {e}")
                    return False

            # Map [0,1] quality-like z_flag to VVDS-inspired bins
            def quality_like_to_flag(x):
                if pd.isna(x):
                    return np.nan
                x = float(x)
                if x == 0.0:
                    return 0.0
                if 0.0 < x < 0.7:
                    return 1.0
                if 0.7 <= x < 0.9:
                    return 2.0
                if 0.9 <= x < 0.99:
                    return 3.0
                if 0.99 <= x <= 1.0:
                    return 4.0
                return np.nan  # outside expected range

            used_zflag_quality_fastpath = False
            if can_use_zflag_as_quality():
                logger_prep.info(f"{product_name} Using 'z_flag' (quality-like) fast path for z_flag_homogenized.")
                df["z_flag_homogenized"] = df["z_flag"].map_partitions(
                    lambda s: s.apply(quality_like_to_flag),
                    meta=("z_flag_homogenized", "f8"),
                )
                used_zflag_quality_fastpath = True
            else:
                # Fallback to YAML translation as before
                logger_prep.info(f"{product_name} z_flag not quality-like; falling back to YAML translation for z_flag_homogenized.")
                df["z_flag_homogenized"] = df.map_partitions(
                    lambda p: p.apply(lambda row: apply_translation(row, "z_flag"), axis=1),
                    meta=("z_flag_homogenized", "f8")
                )
        else:
            logger_prep.warning(
                f"{product_name} Column 'z_flag_homogenized' already exists in catalog '{product_name}'. Skipping translation."
            )
    
    if "instrument_type_homogenized" in tiebreaking_priority:
        if "instrument_type_homogenized" not in df.columns:
            # Fast path from type ∈ {s,g,p}
            def can_use_type_for_instrument() -> bool:
                if "type" not in df.columns:
                    return False
                try:
                    allowed = {"s", "g", "p"}
                    uniques = df["type"].dropna().unique().compute().tolist()
                    # Filter empty strings that may appear after fillna
                    uniques = [u for u in uniques if isinstance(u, str) and u != ""]
                    if len(uniques) == 0:
                        return False
                    return all(u in allowed for u in uniques)
                except Exception as e:
                    logger_prep.warning(f"⚠️ {product_name} Could not validate 'type' values: {e}")
                    return False

            used_type_fastpath = False
            if can_use_type_for_instrument():
                logger_prep.info(f"{product_name} Using 'type' fast path for instrument_type_homogenized.")
                # Copy as-is (already normalized to lower-case)
                df["instrument_type_homogenized"] = df["type"].astype(object)
                used_type_fastpath = True
            else:
                # Fallback to YAML translation as before
                df["instrument_type_homogenized"] = df.map_partitions(
                    lambda p: p.apply(lambda row: apply_translation(row, "instrument_type"), axis=1),
                    meta=("instrument_type_homogenized", "object")
                )
        else:
            logger_prep.warning(f"{product_name} Column 'instrument_type_homogenized' already exists in catalog '{product_name}'. Skipping translation.")

    # === Apply tie-breaking and duplicate removal if required ===
    if combine_mode in ["concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"]:
        delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)
        tiebreaking_priority = translation_config.get("tiebreaking_priority", [])
        instrument_type_priority = translation_config.get("instrument_type_priority", {})

        if not tiebreaking_priority:
            logger_prep.warning(f"⚠️ {product_name} tiebreaking_priority is empty for catalog '{product_name}'. Proceeding with delta_z_threshold tie-breaking only.")
        else:
            for col in tiebreaking_priority:
                if col not in df.columns:
                    raise ValueError(
                        f"Tiebreaking column '{col}' is missing in catalog '{product_name}'."
                    )
                if (df[col].isna() | (df[col] == "")).all().compute():
                    raise ValueError(
                        f"Tiebreaking column '{col}' is invalid in catalog '{product_name}' (all values are NaN)."
                    )
                if col != "instrument_type_homogenized":
                    col_dtype = df[col].dtype
                    if not np.issubdtype(col_dtype, np.number):
                        try:
                            df[col] = df[col].astype(float)
                            logger_prep.info(f"ℹ️ {product_name} Column '{col}' cast to float for tie-breaking.")
                        except Exception as e:
                            raise ValueError(
                                f"Tiebreaking column '{col}' must be numeric (except for 'instrument_type_homogenized'). "
                                f"Attempted cast to float but failed. Error: {e}"
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
                    if priority_col == "instrument_type_homogenized":
                        surviving["_priority_value"] = surviving["instrument_type_homogenized"].map(instrument_type_priority).fillna(-np.inf)
                    else:
                        surviving["_priority_value"] = surviving[priority_col].fillna(-np.inf)

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

                # Update compared_to_dict_solo for reporting
                ids_all = group["CRD_ID"].tolist()
                for i, id1 in enumerate(ids_all):
                    for id2 in ids_all[i + 1:]:
                        compared_to_dict_solo[id1].add(id2)
                        compared_to_dict_solo[id2].add(id1)

            # Merge tie updates back to main dataframe
            if tie_updates:
                combined_update = pd.concat(tie_updates)
                tie_update_dd = dd.from_pandas(combined_update, npartitions=1)
                
                # Ensure consistent string type before merging
                df["CRD_ID"] = df["CRD_ID"].astype(str)
                tie_update_dd["CRD_ID"] = tie_update_dd["CRD_ID"].astype(str)
                
                df = df.drop("tie_result", axis=1).merge(
                    tie_update_dd, on="CRD_ID", how="left"
                ).fillna({"tie_result": 1})

    # Save compared_to_dict even if it's empty (for concatenate mode)
    with open(compared_to_path, "w") as f:
        json.dump({k: sorted(v) for k, v in compared_to_dict_solo.items()}, f)

    # === Flag objects inside any DP1 region ===
    # Hardcoded list of DP1 regions: (RA_center, DEC_center, radius_deg)
    # Source of centers: https://rtn-011.lsst.io/
    dp1_regions = [
        (6.02,   -72.08, 2.5),   # 47 Tuc
        (37.86,    6.98, 2.5),   # Rubin SV 38 7
        (40.00,  -34.45, 2.5),   # Fornax dSph
        (53.13,  -28.10, 2.5),   # ECDFS
        (59.10,  -48.73, 2.5),   # EDFS
        (95.00,  -25.00, 2.5),   # Rubin SV 95 -25
        (106.23, -10.51, 2.5),   # Seagull
    ]

    # Precompute centers in radians
    ra_centers = np.deg2rad([r[0] for r in dp1_regions])
    dec_centers = np.deg2rad([r[1] for r in dp1_regions])
    radii = [r[2] for r in dp1_regions]
    
    def compute_in_dp1_fields(partition):
        partition = partition.copy()
        ra_rad = np.deg2rad(partition["ra"].values)
        dec_rad = np.deg2rad(partition["dec"].values)
    
        # Initialize boolean mask for whether each object is inside any region
        in_any_field = np.zeros(len(partition), dtype=bool)
    
        for ra_cen, dec_cen, radius_deg in zip(ra_centers, dec_centers, radii):
            # Compute angular distance between object and region center
            cos_angle = (
                np.sin(dec_cen) * np.sin(dec_rad) +
                np.cos(dec_cen) * np.cos(dec_rad) * np.cos(ra_rad - ra_cen)
            )
            angle_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))
            in_field = angle_deg <= radius_deg
    
            # Mark objects inside this region
            in_any_field |= in_field
    
        partition["is_in_DP1_fields"] = in_any_field.astype(int)
        return partition
    
    # Apply the region check to each Dask partition
    df = df.map_partitions(
        compute_in_dp1_fields,
        meta=df._meta.assign(is_in_DP1_fields=np.int64())
    )

    def extract_variables_from_expr(expr):
        """Extracts variable names used in a Python expression string."""
        try:
            tree = ast.parse(expr, mode="eval")
        except Exception:
            return set()
        
        class VariableVisitor(ast.NodeVisitor):
            def __init__(self):
                self.variables = set()
    
            def visit_Name(self, node):
                self.variables.add(node.id)
                self.generic_visit(node)
    
        visitor = VariableVisitor()
        visitor.visit(tree)
        return visitor.variables
    
    # Default required output columns
    final_columns = [
        "CRD_ID", "id", "ra", "dec", "z", "z_flag", "z_err", "instrument_type", "survey",
        "source", "tie_result", "is_in_DP1_fields"
    ]
    
    # Add homogenized columns only if they were generated or needed
    if "z_flag_homogenized" in df.columns:
        final_columns.append("z_flag_homogenized")
    if "instrument_type_homogenized" in df.columns:
        final_columns.append("instrument_type_homogenized")
    
    # Ensure all tiebreaking columns are included if present in the DataFrame
    extra_columns = [
        col for col in tiebreaking_priority
        if col not in final_columns and col in df.columns
    ]
    final_columns += extra_columns
    
    # Detect additional columns used in conditional expressions (e.g., "ZCAT_PRIMARY")
    extra_expr_columns = set()
    for ruleset in translation_rules_uc.values():
        for key in ["z_flag_translation", "instrument_type_translation"]:
            rule = ruleset.get(key, {})
            for cond in rule.get("conditions", []):
                expr = cond.get("expr", "")
                variables = extract_variables_from_expr(expr)
                extra_expr_columns.update(variables)
    
    # Exclude standard and already-included columns
    standard_cols = {"id", "ra", "dec", "z", "z_flag", "z_err", "instrument_type", "survey"}
    already_included = set(final_columns)
    needed_expr_cols = [
        col for col in extra_expr_columns
        if col not in standard_cols and col in df.columns and col not in already_included
    ]
    final_columns += needed_expr_cols
    
    # Remove duplicates while preserving order
    final_columns = list(dict.fromkeys(final_columns))
    
    # Select only columns that actually exist in the DataFrame
    df = df[[col for col in final_columns if col in df.columns]]
    
    # Save output Parquet
    out_path = os.path.join(temp_dir, f"prepared_{product_name}")
    df.to_parquet(out_path, write_index=False)
    
    # =============================
    # HATS IMPORT + MARGIN CACHE
    # =============================
    hats_path = ""
    margin_cache_path = ""
    
    if combine_mode != "concatenate":
        if client is None:
            raise ValueError(
                f"❌ Dask client is required for combine_mode='{combine_mode}' "
                f"(needed to run import_catalog and generate_margin_cache)."
            )
    
        match = re.match(r"(\d+)_", product_name)
        if not match:
            raise ValueError(f"❌ Could not extract numeric prefix from internal_name '{product_name}'")
        prefix_str = match.group(1)  # ✅ mantém o zero à esquerda, ex: "010"
    
        artifact_hats = f"cat{prefix_str}_hats"
        artifact_margin = f"cat{prefix_str}_margin"
    
        # === Import to HATS format
        import_catalog(
            path=out_path,
            ra_col="ra",
            dec_col="dec",
            artifact_name=artifact_hats,
            output_path=temp_dir,
            logs_dir=logs_dir,
            logger=logger_prep,
            client=client,
        )
    
        # === Generate margin cache
        generate_margin_cache_safe(
            hats_path=os.path.join(temp_dir, artifact_hats),
            output_path=temp_dir,
            artifact_name=artifact_margin,
            logs_dir=logs_dir,
            logger=logger_prep,
            client=client,
        )
    
        hats_path = os.path.join(temp_dir, artifact_hats)
        margin_cache_path = os.path.join(temp_dir, artifact_margin)

    logger_prep.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: prepare_catalog id={product_name}")
    
    return hats_path, "ra", "dec", product_name, margin_cache_path, compared_to_path


def import_catalog(path, ra_col, dec_col, artifact_name, output_path, logs_dir, logger, client, size_threshold_mb=500):
    """
    Import a Parquet catalog into HATS format.
    Uses lightweight method for small catalogs (< size_threshold_mb).
    
    Args:
        path (str): Path to input Parquet files or directory.
        ra_col (str): RA column name.
        dec_col (str): DEC column name.
        artifact_name (str): Output HATS catalog name.
        output_path (str): Directory to save HATS output.
        client (Client): Dask client for large catalog import.
        size_threshold_mb (int): Threshold to decide method (in MB).
    """
    
    # Detect parquet file paths
    base_path = os.path.join(path, "base")
    if os.path.exists(base_path):
        parquet_files = glob.glob(os.path.join(base_path, "*.parquet"))
    else:
        parquet_files = glob.glob(os.path.join(path, "*.parquet"))

    if not parquet_files:
        raise ValueError(f"No Parquet files found at {path}")

    # Compute total size
    total_size_mb = sum(os.path.getsize(f) for f in parquet_files) / 1024**2

    hats_path = os.path.join(output_path, artifact_name)

    if total_size_mb <= size_threshold_mb:
        logger.info(f"⚡ Small catalog detected ({total_size_mb:.1f} MB). Using direct `to_hats()` method.")
        df = pd.read_parquet(parquet_files)
        catalog = lsdb.from_dataframe(
            df,
            catalog_name=artifact_name,
            ra_column=ra_col,
            dec_column=dec_col
        )
        catalog.to_hats(hats_path, overwrite=True)
        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: import_catalog id={artifact_name}")
    else:
        logger.info(f"🧱 Large catalog detected ({total_size_mb:.1f} MB). Using distributed HATS import.")
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
        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: import_catalog id={artifact_name}")

def generate_margin_cache_safe(hats_path, output_path, artifact_name, logs_dir, logger, client):
    """
    Generate margin cache if partitions > 1; otherwise, skip gracefully.
    If resume fails due to missing critical files, clean and regenerate.
    
    Args:
        hats_path (str): Path to HATS catalog.
        output_path (str): Path to save margin cache.
        artifact_name (str): Name of margin cache artifact.
        client (Client): Dask client.

    Returns:
        str or None: Path to margin cache or None if skipped.
    """

    margin_dir = os.path.join(output_path, artifact_name)
    intermediate_dir = os.path.join(margin_dir, "intermediate")
    critical_file = os.path.join(intermediate_dir, "margin_pair.csv")

    # Check for broken resumption state
    if os.path.exists(intermediate_dir) and not os.path.exists(critical_file):
        logger.info(f"⚠️ Detected incomplete margin cache at {margin_dir}. Deleting to force regeneration...")
        try:
            shutil.rmtree(margin_dir)
        except Exception as e:
            logger.info(f"❌ Failed to delete corrupted margin cache at {margin_dir}: {e}")
            raise

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
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: generate_margin_cache id={artifact_name}")
            return margin_dir
        else:
            logger.info(f"⚠️ Margin cache skipped: single partition for {artifact_name}")
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: generate_margin_cache id={artifact_name}")
            return None
    except ValueError as e:
        if "Margin cache contains no rows" in str(e):
            logger.info(f"⚠️ {e} Proceeding without margin cache.")
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: generate_margin_cache id={artifact_name}")
            return None
        else:
            raise