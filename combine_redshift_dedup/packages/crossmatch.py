import os
import json
import warnings
import numpy as np
import pandas as pd
import dask.dataframe as dd
import lsdb

from combine_redshift_dedup.packages.utils import get_global_logger, log_and_print

def crossmatch_tiebreak(left_cat, right_cat, tiebreaking_priority, temp_dir, step, client, compared_to_dict, instrument_type_priority, translation_config, do_import=True):

    logger = get_global_logger()
    
    output_xmatch_dir = os.path.join(temp_dir, f"xmatch_step{step}")

    crossmatch_radius = translation_config.get("crossmatch_radius_arcsec", 1.0)
    
    # Perform a spatial crossmatch between the two catalogs within specified radius
    xmatched = left_cat.crossmatch(
        right_cat,
        radius_arcsec=crossmatch_radius,
        n_neighbors=1,
        suffixes=("left", "right")
    )

    # Save the crossmatched catalog in HATS format
    xmatched.to_hats(output_xmatch_dir, overwrite=True)

    # Read the crossmatched catalog as Dask DataFrame
    df = lsdb.read_hats(output_xmatch_dir)._ddf

    # Validate tiebreaking_priority configuration
    if not tiebreaking_priority:
        logger.warning("⚠️ tiebreaking_priority is empty. Proceeding directly to delta_z_threshold tie-breaking.")
    else:
        for col in tiebreaking_priority:
            col_left = f"{col}left"
            col_right = f"{col}right"
        
            if col_left not in df.columns or col_right not in df.columns:
                raise ValueError(
                    f"Tiebreaking column '{col}' is not present in both catalogs. "
                    f"Missing: {'left' if col_left not in df.columns else 'right'}."
                )
        
            left_all_nan = df[col_left].isna().all().compute()
            right_all_nan = df[col_right].isna().all().compute()
        
            if left_all_nan and right_all_nan:
                raise ValueError(
                    f"Tiebreaking column '{col}' is present in both catalogs, but all values are NaN in the crossmatch result."
                )
        
            # Check if custom tiebreaking column is numeric (unless it's instrument_type_homogenized)
            if col != "instrument_type_homogenized":
                if not (pd.api.types.is_numeric_dtype(df[col_left].dtype) and pd.api.types.is_numeric_dtype(df[col_right].dtype)):
                    raise ValueError(
                        f"Tiebreaking column '{col}' must be numeric in both catalogs (except for 'instrument_type_homogenized'). "
                        f"Found dtypes: left={df[col_left].dtype}, right={df[col_right].dtype}."
                    )

    # Remove pairs where either source was already rejected
    df = df[(df["tie_resultleft"] != 0) & (df["tie_resultright"] != 0)]

    # Load delta_z threshold from config
    delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)

    # If no tiebreaking_priority and no delta_z_threshold → fail
    if not tiebreaking_priority and (delta_z_threshold is None or delta_z_threshold == 0.0):
        raise ValueError(
            "Cannot deduplicate: tiebreaking_priority is empty and delta_z_threshold is not set or is zero. "
            "Please define at least one deduplication criterion."
        )

    # Load existing hard tie list or initialize empty set
    hard_tie_path = os.path.join(temp_dir, "hard_tie_ids.json")
    if os.path.exists(hard_tie_path):
        with open(hard_tie_path, "r") as f:
            hard_tie_cumulative = set(json.load(f))
    else:
        hard_tie_cumulative = set()

    # Tie-breaking function applied row by row
    def decide_tie(row):
        left_was_tie = row.get("tie_resultleft", 1) == 2
        right_was_tie = row.get("tie_resultright", 1) == 2
    
        zflag_left = row.get("z_flag_homogenizedleft")
        zflag_right = row.get("z_flag_homogenizedright")
    
        # Special case: if both are stars (z_flag == 6), eliminate both
        if pd.notna(zflag_left) and zflag_left == 6 and pd.notna(zflag_right) and zflag_right == 6:
            return (0, 0)
        # If only the left is a star, eliminate left
        elif pd.notna(zflag_left) and zflag_left == 6:
            return (0, 2 if right_was_tie else 1)
        # If only the right is a star, eliminate right
        elif pd.notna(zflag_right) and zflag_right == 6:
            return (2 if left_was_tie else 1, 0)
    
        # If no tiebreaking columns provided, mark as hard tie (to be resolved by delta_z)
        if not tiebreaking_priority:
            return (2, 2)
    
        # Track the first column where one side was not NaN and the other was
        first_non_nan_col = None
        first_non_nan_side = None
    
        # Loop through all tiebreaking columns in priority order
        for col in tiebreaking_priority:
            v1 = row.get(f"{col}left")
            v2 = row.get(f"{col}right")
    
            # Apply type priority mapping if column is instrument_type_homogenized
            if col == "instrument_type_homogenized":
                v1 = instrument_type_priority.get(v1, 0)
                v2 = instrument_type_priority.get(v2, 0)
    
            if pd.notnull(v1) and pd.notnull(v2):
                # If both values are valid, decide immediately
                if v1 > v2:
                    return (2 if left_was_tie else 1, 0)
                elif v2 > v1:
                    return (0, 2 if right_was_tie else 1)
                # If values are equal, continue to next column
                else:
                    continue
    
            elif pd.notnull(v1) and pd.isnull(v2):
                # Found a column where left has value and right does not
                if first_non_nan_col is None:
                    first_non_nan_col = col
                    first_non_nan_side = "left"
    
            elif pd.isnull(v1) and pd.notnull(v2):
                # Found a column where right has value and left does not
                if first_non_nan_col is None:
                    first_non_nan_col = col
                    first_non_nan_side = "right"
    
            # If both are NaN, move to next column without setting first_non_nan
    
        # After checking all columns, if no pair of values resolved the tie:
        if first_non_nan_side == "left":
            return (2 if left_was_tie else 1, 0)
        elif first_non_nan_side == "right":
            return (0, 2 if right_was_tie else 1)
        else:
            # No information to break tie → hard tie
            return (2, 2)

    # Apply tie-breaking function
    tie_results = df.map_partitions(
        lambda p: p.apply(decide_tie, axis=1, result_type="expand"),
        meta={0: "i8", 1: "i8"}
    )
    df = df.assign(tie_left=tie_results[0], tie_right=tie_results[1])

    # Collect pairs for delta_z processing
    pairs = df[["CRD_IDleft", "CRD_IDright", "tie_left", "tie_right", "zleft", "zright"]].compute()

    # Update compared_to_dict for all pairs
    for _, row in pairs.iterrows():
        compared_to_dict.setdefault(str(row["CRD_IDleft"]), []).append(str(row["CRD_IDright"]))
        compared_to_dict.setdefault(str(row["CRD_IDright"]), []).append(str(row["CRD_IDleft"]))

    # Save compared_to_dict to file
    compared_to_path = os.path.join(temp_dir, "compared_to.json")
    with open(compared_to_path, "w") as f:
        json.dump(compared_to_dict, f)

    # Apply delta_z tie-breaking on hard ties
    if delta_z_threshold > 0:
        def apply_delta_z_fix(pairs_df):
            pairs_df = pairs_df.copy()
            mask = (pairs_df["tie_left"] == 2) & (pairs_df["tie_right"] == 2)
            for idx, row in pairs_df[mask].iterrows():
                z1 = row["zleft"]
                z2 = row["zright"]
                if pd.notnull(z1) and pd.notnull(z2):
                    if abs(z1 - z2) <= delta_z_threshold:
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

    # Collapse tie results across left/right
    expanded = pd.concat([
        pairs[["CRD_IDleft", "tie_left"]].rename(columns={"CRD_IDleft": "CRD_ID", "tie_left": "tie_result"}),
        pairs[["CRD_IDright", "tie_right"]].rename(columns={"CRD_IDright": "CRD_ID", "tie_right": "tie_result"})
    ])

    # Consolidate into single tie_result per CRD_ID
    def consolidate_results(series):
        if (series == 0).any():
            return 0
        if (series == 2).any():
            return 2
        return 1

    expanded["CRD_ID"] = expanded["CRD_ID"].astype(str)
    final_results = expanded.groupby("CRD_ID")["tie_result"].apply(consolidate_results)

    # Update hard ties list
    hard_tie_ids = final_results[final_results == 2].index.tolist()
    hard_tie_cumulative.update(hard_tie_ids)
    with open(hard_tie_path, "w") as f:
        json.dump(list(hard_tie_cumulative), f)

    # Apply final tie_result and compared_to to original catalogs
    def apply_final(df_part):
        df_part["tie_result"] = df_part["CRD_ID"].map(final_results).fillna(df_part["tie_result"])
        df_part["compared_to"] = df_part["CRD_ID"].map(lambda x: ",".join(compared_to_dict.get(str(x), [])))
        return df_part

    left_df = left_cat._ddf.map_partitions(apply_final)
    right_df = right_cat._ddf.map_partitions(apply_final)

    # Combine catalogs and save result
    merged = dd.concat([left_df, right_df])
    
    # Ensure stable dtypes before saving to avoid ArrowNotImplementedError
    expected_types = {
        "CRD_ID": "string",
        "id": "string",
        "ra": "float64",
        "dec": "float64",
        "z": "float64",
        "z_flag": "string",
        "z_err": "float64",
        "instrument_type": "string",
        "survey": "string",
        "source": "string",
        "tie_result": "int64",
        "compared_to": "string",
        "z_flag_homogenized": "float64",
        "instrument_type_homogenized": "string",
    }
    
    # Add tiebreaking_priority columns dynamically
    for col in tiebreaking_priority:
        if col != "instrument_type_homogenized":
            expected_types[col] = "float64"
        else:
            expected_types[col] = "string"
    
    for col, dtype in expected_types.items():
        if col in merged.columns:
            try:
                if dtype == "string":
                    merged[col] = merged[col].fillna("").astype("string")
                else:
                    merged[col] = merged[col].astype(dtype)
            except Exception as e:
                logger.warning(f"⚠️ Failed to cast column '{col}' to {dtype}: {e}")
    
    merged_path = os.path.join(temp_dir, f"merged_step{step}")
    merged.to_parquet(merged_path, write_index=False)

    # Import merged catalog to HATS
    if do_import:
        from combine_redshift_dedup.packages.specz import import_catalog
        import_catalog(merged_path, "ra", "dec", f"merged_step{step}_hats", temp_dir, client)
        return lsdb.read_hats(os.path.join(temp_dir, f"merged_step{step}_hats"))
    else:
        return merged_path

def crossmatch_tiebreak_safe(*args, **kwargs):

    logger = get_global_logger()
    
    try:
        return crossmatch_tiebreak(*args, **kwargs)
    except RuntimeError as e:
        if (
            "The output catalog is empty" in str(e)
            or "Catalogs do not overlap" in str(e)
        ):
            log_and_print(f"⚠️ {e} Proceeding by merging left and right without crossmatching.", logger)
            left_cat, right_cat = args[0], args[1]
            temp_dir = args[3]
            step = args[4]
            client = args[5]
            compared_to_dict = args[6]

            left_df = left_cat._ddf
            right_df = right_cat._ddf
            merged = dd.concat([left_df, right_df])

            merged = merged.assign(
                compared_to=merged["id"].map(lambda x: ",".join(compared_to_dict.get(str(x), [])))
            )

            merged_path = os.path.join(temp_dir, f"merged_step{step}")
            merged.to_parquet(merged_path, write_index=False)

            # Import merged catalog to HATS
            if kwargs.get("do_import", True):  # novo controle aqui também
                from combine_redshift_dedup.packages.specz import import_catalog
                import_catalog(merged_path, "ra", "dec", f"merged_step{step}_hats", temp_dir, client)
                return lsdb.read_hats(os.path.join(temp_dir, f"merged_step{step}_hats"))
            else:
                return merged_path
        else:
            raise