# =====================
# Built-in modules
# =====================
import json
import logging
import os
import pathlib
import warnings
from collections import defaultdict
from datetime import datetime

# =====================
# Third-party libraries
# =====================
import dask.dataframe as dd
import lsdb
import numpy as np
import pandas as pd

# =====================
# Project-specific libraries
# =====================
from combine_redshift_dedup.packages.specz import (
    import_catalog,
    _normalize_string_series_to_na,
)

def crossmatch_tiebreak(
    left_cat,
    right_cat,
    tiebreaking_priority,
    logs_dir,
    temp_dir,
    step,
    client,
    compared_to_left,
    compared_to_right,
    instrument_type_priority,
    translation_config,
    do_import=True,
):

    # === Logger setup ===
    log_path = pathlib.Path(logs_dir) / "crossmatch_and_merge_all.log"
    logger_x = logging.getLogger("crossmatch_and_merge_logger")
    logger_x.setLevel(logging.INFO)
    if logger_x.hasHandlers():
        logger_x.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger_x.addHandler(fh)

    output_xmatch_dir = os.path.join(temp_dir, f"xmatch_step{step}")
    crossmatch_radius = translation_config.get("crossmatch_radius_arcsec", 1.0)

    # Run spatial crossmatch (within radius) and persist as HATS
    xmatched = left_cat.crossmatch(
        right_cat,
        radius_arcsec=crossmatch_radius,
        n_neighbors=1,
        suffixes=("left", "right"),
    )
    xmatched.to_hats(output_xmatch_dir, overwrite=True)

    # Bring crossmatched HATS to Dask DataFrame
    df = lsdb.read_hats(output_xmatch_dir)._ddf

    # ------------------------------------------------------------------
    # 1) HARD TYPE NORMALIZATION to match specz.py (nullable dtypes)
    # ------------------------------------------------------------------
    # Float-like columns → Float64 (nullable)
    float_bases = ["ra", "dec", "z", "z_flag", "z_err", "z_flag_homogenized"]
    for base in float_bases:
        for side in ("left", "right"):
            col = f"{base}{side}"
            if col in df.columns:
                coerced = dd.to_numeric(df[col], errors="coerce")
                df[col] = coerced.map_partitions(
                    lambda s: s.astype("Float64"),
                    meta=pd.Series(pd.array([], dtype="Float64")),
                )

    # String-like columns → StringDtype("string") with NA (never "")
    str_bases = [
        "CRD_ID",
        "id",
        "instrument_type",
        "instrument_type_homogenized",
        "survey",
        "source",
    ]
    for base in str_bases:
        for side in ("left", "right"):
            col = f"{base}{side}"
            if col in df.columns:
                df[col] = df[col].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype="string")),
                )

    # Tie result columns can be nullable ints; we keep whatever is there but
    # it's useful to stabilize as Int64 for robust comparisons/join later.
    for col in ("tie_resultleft", "tie_resultright"):
        if col in df.columns:
            df[col] = dd.to_numeric(df[col], errors="coerce").map_partitions(
                lambda s: s.astype("Int64"),
                meta=pd.Series(pd.array([], dtype="Int64")),
            )

    # ------------------------------------------------------------------
    # 2) VALIDATE tiebreaking_priority configuration (post-normalization)
    # ------------------------------------------------------------------
    if not tiebreaking_priority:
        logger_x.warning("⚠️ tiebreaking_priority is empty. Proceeding directly to delta_z_threshold tie-breaking.")
    else:
        for col in tiebreaking_priority:
            col_left = f"{col}left"
            col_right = f"{col}right"

            if col_left not in df.columns or col_right not in df.columns:
                raise ValueError(
                    f"Tiebreaking column '{col}' is not present in both catalogs. "
                    f"Missing: {'left' if col_left not in df.columns else 'right'}."
                )

            # Fail if both sides are entirely NA/empty
            left_all_na = df[col_left].isna().all().compute()
            right_all_na = df[col_right].isna().all().compute()
            if left_all_na and right_all_na:
                raise ValueError(
                    f"Tiebreaking column '{col}' is present in both catalogs, but all values are NaN in the crossmatch result."
                )

            # All tiebreak columns (except instrument_type_homogenized) must be numeric.
            if col != "instrument_type_homogenized":
                # If dtype isn't numeric (rare after coercion), try to coerce now
                if not (pd.api.types.is_numeric_dtype(df[col_left].dtype) and
                        pd.api.types.is_numeric_dtype(df[col_right].dtype)):
                    try:
                        for c in (col_left, col_right):
                            coerced = dd.to_numeric(df[c], errors="coerce")
                            df[c] = coerced.map_partitions(
                                lambda s: s.astype("Float64"),
                                meta=pd.Series(pd.array([], dtype="Float64")),
                            )
                        logger_x.info(f"ℹ️ Cast '{col}' (left/right) to Float64 for tie-breaking.")
                    except Exception as e:
                        raise ValueError(
                            f"Tiebreaking column '{col}' must be numeric in both catalogs (except 'instrument_type_homogenized'). "
                            f"Found dtypes: left={df[col_left].dtype}, right={df[col_right].dtype}. Error coercing: {e}"
                        )

    # Remove pairs where either source was already rejected
    # (works fine with Int64, comparisons broadcast)
    if "tie_resultleft" in df.columns and "tie_resultright" in df.columns:
        df = df[(df["tie_resultleft"] != 0) & (df["tie_resultright"] != 0)]

    # Threshold for delta-z secondary disambiguation
    delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)

    if not tiebreaking_priority and (delta_z_threshold is None or float(delta_z_threshold) == 0.0):
        raise ValueError(
            "Cannot deduplicate: tiebreaking_priority is empty and delta_z_threshold is not set or is zero. "
            "Please define at least one deduplication criterion."
        )

    # Load hard-tie accumulator
    hard_tie_path = os.path.join(temp_dir, "hard_tie_ids.json")
    if os.path.exists(hard_tie_path):
        with open(hard_tie_path, "r") as f:
            hard_tie_cumulative = set(json.load(f))
    else:
        hard_tie_cumulative = set()

    # ------------------------------------------------------------------
    # 3) ROW-WISE TIE DECISION
    # ------------------------------------------------------------------
    def decide_tie(row):
        left_was_tie = row.get("tie_resultleft", 1) == 2
        right_was_tie = row.get("tie_resultright", 1) == 2

        zflag_left = row.get("z_flag_homogenizedleft")
        zflag_right = row.get("z_flag_homogenizedright")

        # Star rule: if z_flag_homogenized == 6, prefer the non-star or drop both if both stars
        if pd.notna(zflag_left) and zflag_left == 6 and pd.notna(zflag_right) and zflag_right == 6:
            return (0, 0)
        elif pd.notna(zflag_left) and zflag_left == 6:
            return (0, 2 if right_was_tie else 1)
        elif pd.notna(zflag_right) and zflag_right == 6:
            return (2 if left_was_tie else 1, 0)

        if not tiebreaking_priority:
            # Defer to delta-z (hard tie for now)
            return (2, 2)

        first_non_nan_side = None

        for col in tiebreaking_priority:
            v1 = row.get(f"{col}left")
            v2 = row.get(f"{col}right")

            if col == "instrument_type_homogenized":
                # Map categorical to numeric priority; NA -> 0
                v1 = instrument_type_priority.get(v1, 0)
                v2 = instrument_type_priority.get(v2, 0)

            # Both present → decide immediately
            if pd.notna(v1) and pd.notna(v2):
                if v1 > v2:
                    return (2 if left_was_tie else 1, 0)
                elif v2 > v1:
                    return (0, 2 if right_was_tie else 1)
                else:
                    continue  # equal → next criterion

            # Track the first criterion where only one side has value
            if pd.notna(v1) and pd.isna(v2) and first_non_nan_side is None:
                first_non_nan_side = "left"
            elif pd.isna(v1) and pd.notna(v2) and first_non_nan_side is None:
                first_non_nan_side = "right"

        if first_non_nan_side == "left":
            return (2 if left_was_tie else 1, 0)
        elif first_non_nan_side == "right":
            return (0, 2 if right_was_tie else 1)
        else:
            # No information to break tie
            return (2, 2)

    tie_results = df.map_partitions(
        lambda p: p.apply(decide_tie, axis=1, result_type="expand"),
        meta={0: "i8", 1: "i8"},
    )
    df = df.assign(tie_left=tie_results[0], tie_right=tie_results[1])

    # ------------------------------------------------------------------
    # 4) COLLECT PAIRS FOR DELTA-Z PROCESSING (work in pandas)
    # ------------------------------------------------------------------
    pairs = df[["CRD_IDleft", "CRD_IDright", "tie_left", "tie_right", "zleft", "zright"]].compute()

    # Build new compared_to links from this step
    compared_to_new = defaultdict(set)
    for _, row in pairs.iterrows():
        left_id = str(row["CRD_IDleft"])
        right_id = str(row["CRD_IDright"])
        compared_to_new[left_id].add(right_id)
        compared_to_new[right_id].add(left_id)

    logger_x.info(f"🧮 New pairs this step: {sum(len(v) for v in compared_to_new.values())} total links in {len(compared_to_new)} objects")

    # Merge with previous compared_to dicts
    compared_to_dict = defaultdict(set)
    for d in [compared_to_left, compared_to_right]:
        for k, v in d.items():
            compared_to_dict[str(k)].update(map(str, v))
    for k, new_vals in compared_to_new.items():
        compared_to_dict[str(k)].update(map(str, new_vals))

    total_links = sum(len(v) for v in compared_to_dict.values())
    logger_x.info(f"🔗 Compared_to (merged): {total_links} total links across {len(compared_to_dict)} objects")
    logger_x.info(f"🔢 Step {step}: Compared_to merge completed")

    # Persist compared_to
    compared_to_dict = {k: sorted(v) for k, v in compared_to_dict.items()}
    compared_to_path = os.path.join(temp_dir, f"compared_to_xmatch_step{step}.json")
    with open(compared_to_path, "w") as f:
        json.dump(compared_to_dict, f)

    # ------------------------------------------------------------------
    # 5) DELTA-Z TIE RESOLUTION ON HARD TIES
    # ------------------------------------------------------------------
    if float(delta_z_threshold) > 0.0:
        def apply_delta_z_fix(pairs_df: pd.DataFrame) -> pd.DataFrame:
            pairs_df = pairs_df.copy()
            mask = (pairs_df["tie_left"] == 2) & (pairs_df["tie_right"] == 2)
            for idx, row in pairs_df[mask].iterrows():
                z1 = row["zleft"]
                z2 = row["zright"]
                # Skip if any side is NA
                if pd.isna(z1) or pd.isna(z2):
                    continue
                if abs(float(z1) - float(z2)) <= float(delta_z_threshold):
                    left_crd_id = str(row["CRD_IDleft"])
                    right_crd_id = str(row["CRD_IDright"])
                    left_in_hard = left_crd_id in hard_tie_cumulative
                    right_in_hard = right_crd_id in hard_tie_cumulative

                    if left_in_hard and not right_in_hard:
                        pairs_df.at[idx, "tie_right"] = 0
                    elif right_in_hard and not left_in_hard:
                        pairs_df.at[idx, "tie_left"] = 0
                    else:
                        n_left = len(compared_to_dict.get(left_crd_id, []))
                        n_right = len(compared_to_dict.get(right_crd_id, []))
                        if n_left > n_right:
                            pairs_df.at[idx, "tie_right"] = 0
                            pairs_df.at[idx, "tie_left"] = 1
                        elif n_right > n_left:
                            pairs_df.at[idx, "tie_left"] = 0
                            pairs_df.at[idx, "tie_right"] = 1
                        else:
                            pairs_df.at[idx, "tie_right"] = 0
                            pairs_df.at[idx, "tie_left"] = 1
            return pairs_df

        pairs = apply_delta_z_fix(pairs)

    # ------------------------------------------------------------------
    # 6) COLLAPSE LEFT/RIGHT TIE RESULTS → SINGLE tie_result PER CRD_ID
    # ------------------------------------------------------------------
    expanded = pd.concat([
        pairs[["CRD_IDleft", "tie_left"]].rename(columns={"CRD_IDleft": "CRD_ID", "tie_left": "tie_result"}),
        pairs[["CRD_IDright", "tie_right"]].rename(columns={"CRD_IDright": "CRD_ID", "tie_right": "tie_result"}),
    ])

    def consolidate_results(series: pd.Series) -> int:
        if (series == 0).any():
            return 0
        if (series == 2).any():
            return 2
        return 1

    expanded["CRD_ID"] = expanded["CRD_ID"].astype(str)
    final_results = expanded.groupby("CRD_ID")["tie_result"].apply(consolidate_results)

    # Update hard ties set and persist
    hard_tie_ids = final_results[final_results == 2].index.tolist()
    hard_tie_cumulative.update(hard_tie_ids)
    with open(hard_tie_path, "w") as f:
        json.dump(list(hard_tie_cumulative), f)

    # ------------------------------------------------------------------
    # 7) APPLY FINAL RESULTS BACK TO LEFT/RIGHT AND MERGE
    # ------------------------------------------------------------------
    def apply_final(df_part: pd.DataFrame) -> pd.DataFrame:
        # Preserve dtype; map may insert NaN → fill with existing value
        out = df_part.copy()
        mapped = out["CRD_ID"].map(final_results)
        out["tie_result"] = mapped.fillna(out["tie_result"])
        return out

    left_df = left_cat._ddf.map_partitions(apply_final, meta=left_cat._ddf._meta)
    right_df = right_cat._ddf.map_partitions(apply_final, meta=right_cat._ddf._meta)

    merged = dd.concat([left_df, right_df])

    # ------------------------------------------------------------------
    # 8) STABILIZE DTYPES (nullable) BEFORE SAVING
    # ------------------------------------------------------------------
    expected_types = {
        "CRD_ID": "string",
        "id": "string",
        "ra": "Float64",
        "dec": "Float64",
        "z": "Float64",
        "z_flag": "Float64",
        "z_err": "Float64",
        "instrument_type": "string",
        "survey": "string",
        "source": "string",
        "tie_result": "Int64",
        "z_flag_homogenized": "Float64",
        "instrument_type_homogenized": "string",
    }
    # Include any custom tiebreaking columns in the output type map
    for col in tiebreaking_priority:
        expected_types[col] = "string" if col == "instrument_type_homogenized" else "Float64"

    # Cast using the same semantics as specz.py (no fillna("") for strings!)
    for col, dtype in expected_types.items():
        if col not in merged.columns:
            continue
        try:
            if dtype == "string":
                merged[col] = merged[col].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype="string")),
                )
            elif dtype == "Float64":
                merged[col] = dd.to_numeric(merged[col], errors="coerce").map_partitions(
                    lambda s: s.astype("Float64"),
                    meta=pd.Series(pd.array([], dtype="Float64")),
                )
            elif dtype == "Int64":
                merged[col] = dd.to_numeric(merged[col], errors="coerce").map_partitions(
                    lambda s: s.astype("Int64"),
                    meta=pd.Series(pd.array([], dtype="Int64")),
                )
        except Exception as e:
            logger_x.warning(f"⚠️ Failed to cast column '{col}' to {dtype}: {e}")

    # Save merged parquet
    merged_path = os.path.join(temp_dir, f"merged_step{step}")
    merged.to_parquet(merged_path, write_index=False)

    # Optionally import merged to HATS
    if do_import:
        import_catalog(merged_path, "ra", "dec", f"merged_step{step}_hats", temp_dir, logs_dir, logger_x, client)
        logger_x.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: crossmatch_and_merge id=merged_step{step}")
        return lsdb.read_hats(os.path.join(temp_dir, f"merged_step{step}_hats")), compared_to_path
    else:
        logger_x.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: crossmatch_and_merge id=merged_step{step}")
        return merged_path, compared_to_path


def crossmatch_tiebreak_safe(
    left_cat,
    right_cat,
    tiebreaking_priority,
    logs_dir,
    temp_dir,
    step,
    client,
    compared_to_left,
    compared_to_right,
    instrument_type_priority,
    translation_config,
    do_import=True,
):
    # === Logger setup ===
    log_path = pathlib.Path(logs_dir) / "crossmatch_and_merge_all.log"
    logger_x = logging.getLogger("crossmatch_and_merge_logger")
    logger_x.setLevel(logging.INFO)

    # Clear previous handlers to avoid duplicate log lines
    if logger_x.hasHandlers():
        logger_x.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger_x.addHandler(fh)

    logger_x.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: crossmatch_and_merge id=merged_step{step}")

    # === Try main crossmatch logic ===
    try:
        return crossmatch_tiebreak(
            left_cat=left_cat,
            right_cat=right_cat,
            tiebreaking_priority=tiebreaking_priority,
            logs_dir=logs_dir,
            temp_dir=temp_dir,
            step=step,
            client=client,
            compared_to_left=compared_to_left,
            compared_to_right=compared_to_right,
            instrument_type_priority=instrument_type_priority,
            translation_config=translation_config,
            do_import=do_import,
        )

    # === Fallback logic for non-overlapping catalogs or empty result ===
    except RuntimeError as e:
        if (
            "The output catalog is empty" in str(e)
            or "Catalogs do not overlap" in str(e)
        ):
            logger_x.info(f"⚠️ {e} Proceeding by merging left and right without crossmatching.")
    
            # === Merge input compared_to dicts cumulatively ===
            compared_to_dict = defaultdict(set)
            for d in [compared_to_left, compared_to_right]:
                for k, v in d.items():
                    k_str = str(k)
                    v_str = map(str, v)
                    compared_to_dict[k_str].update(v_str)
            
            # === Convert to sorted lists and save ===
            compared_to_dict = {k: sorted(v) for k, v in compared_to_dict.items()}
            
            compared_to_path = os.path.join(temp_dir, f"compared_to_xmatch_step{step}.json")
            with open(compared_to_path, "w") as f:
                json.dump(compared_to_dict, f)

            # === Concatenate left and right Dask DataFrames ===
            left_df = left_cat._ddf
            right_df = right_cat._ddf
            merged = dd.concat([left_df, right_df])

            # === Save merged result to disk ===
            merged_path = os.path.join(temp_dir, f"merged_step{step}")
            merged.to_parquet(merged_path, write_index=False)

            # === Import merged catalog to HATS format if requested ===
            if do_import:
                import_catalog(
                    merged_path,
                    "ra",
                    "dec",
                    f"merged_step{step}_hats",
                    temp_dir,
                    logs_dir,
                    logger_x,
                    client,
                )
                logger_x.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: crossmatch_and_merge id=merged_step{step}")
                return lsdb.read_hats(os.path.join(temp_dir, f"merged_step{step}_hats")), compared_to_path
            else:
                logger_x.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: crossmatch_and_merge id=merged_step{step}")
                return merged_path, compared_to_path
        else:
            # Re-raise unexpected exceptions
            raise
