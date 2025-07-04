import os
import json
import numpy as np
import pandas as pd
import dask.dataframe as dd
import lsdb

def crossmatch_tiebreak(left_cat, right_cat, tiebreaking_priority, temp_dir, step, client, compared_to_dict, type_priority, translation_config):
    output_xmatch_dir = os.path.join(temp_dir, f"xmatch_step{step}")

    # Perform a spatial crossmatch between the two catalogs within 1 arcsec radius
    xmatched = left_cat.crossmatch(
        right_cat,
        radius_arcsec=1.0,
        n_neighbors=1,
        suffixes=("left", "right")
    )
    # Save the crossmatched catalog in HATS format
    xmatched.to_hats(output_xmatch_dir, overwrite=True)

    # Read the crossmatched catalog as Dask DataFrame
    df = lsdb.read_hats(output_xmatch_dir)._ddf

    # Filter out pairs where either side was already rejected (tie_result == 0)
    df = df[(df["tie_resultleft"] != 0) & (df["tie_resultright"] != 0)]

    # Get delta_z threshold from configuration
    delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)

    # Load or initialize cumulative list of hard ties
    hard_tie_path = os.path.join(temp_dir, "hard_tie_ids.json")
    if os.path.exists(hard_tie_path):
        with open(hard_tie_path, "r") as f:
            hard_tie_cumulative = set(json.load(f))
    else:
        hard_tie_cumulative = set()

    # Function to apply primary tie-breaking rules
    def decide_tie(row):
        left_was_tie = row.get("tie_resultleft", 1) == 2
        right_was_tie = row.get("tie_resultright", 1) == 2

        zflag_left = row.get("z_flag_homogenizedleft")
        zflag_right = row.get("z_flag_homogenizedright")

        if pd.notna(zflag_left) and zflag_left == 6 and pd.notna(zflag_right) and zflag_right == 6:
            return (0, 0)
        elif pd.notna(zflag_left) and zflag_left == 6:
            return (0, 2 if right_was_tie else 1)
        elif pd.notna(zflag_right) and zflag_right == 6:
            return (2 if left_was_tie else 1, 0)

        for col in tiebreaking_priority:
            v1 = row.get(f"{col}left")
            v2 = row.get(f"{col}right")

            if col == "type_homogenized":
                v1 = type_priority.get(v1, 0)
                v2 = type_priority.get(v2, 0)

            if pd.notnull(v1) and pd.notnull(v2):
                if v1 > v2:
                    return (2 if left_was_tie else 1, 0)
                elif v2 > v1:
                    return (0, 2 if right_was_tie else 1)
                else:
                    continue
            elif pd.notnull(v1):
                return (2 if left_was_tie else 1, 0)
            elif pd.notnull(v2):
                return (0, 2 if right_was_tie else 1)

        return (2, 2)

    tie_results = df.map_partitions(
        lambda p: p.apply(decide_tie, axis=1, result_type="expand"),
        meta={0: "i8", 1: "i8"}
    )
    df = df.assign(tie_left=tie_results[0], tie_right=tie_results[1])

    # Compute pairs and update compared_to_dict before delta_z_fix
    pairs = df[["CRD_IDleft", "CRD_IDright", "tie_left", "tie_right", "zleft", "zright"]].compute()

    for _, row in pairs.iterrows():
        compared_to_dict.setdefault(str(row["CRD_IDleft"]), []).append(str(row["CRD_IDright"]))
        compared_to_dict.setdefault(str(row["CRD_IDright"]), []).append(str(row["CRD_IDleft"]))

    compared_to_path = os.path.join(temp_dir, "compared_to.json")
    with open(compared_to_path, "w") as f:
        json.dump(compared_to_dict, f)

    # Apply delta_z fix using computed pairs
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

    # Consolidate tie results
    expanded = pd.concat([
        pairs[["CRD_IDleft", "tie_left"]].rename(columns={"CRD_IDleft": "CRD_ID", "tie_left": "tie_result"}),
        pairs[["CRD_IDright", "tie_right"]].rename(columns={"CRD_IDright": "CRD_ID", "tie_right": "tie_result"})
    ])

    def consolidate_results(series):
        if (series == 0).any():
            return 0
        if (series == 2).any():
            return 2
        return 1

    expanded["CRD_ID"] = expanded["CRD_ID"].astype(str)
    final_results = expanded.groupby("CRD_ID")["tie_result"].apply(consolidate_results)

    hard_tie_ids = final_results[final_results == 2].index.tolist()
    hard_tie_cumulative.update(hard_tie_ids)
    with open(hard_tie_path, "w") as f:
        json.dump(list(hard_tie_cumulative), f)

    def apply_final(df_part):
        df_part["tie_result"] = df_part["CRD_ID"].map(final_results).fillna(df_part["tie_result"])
        df_part["compared_to"] = df_part["CRD_ID"].map(lambda x: ",".join(compared_to_dict.get(str(x), [])))
        return df_part

    left_df = left_cat._ddf.map_partitions(apply_final)
    right_df = right_cat._ddf.map_partitions(apply_final)

    merged = dd.concat([left_df, right_df])
    merged_path = os.path.join(temp_dir, f"merged_step{step}")
    merged.to_parquet(merged_path, write_index=False)

    from combine_redshift_dedup.packages.specz import import_catalog
    import_catalog(merged_path, "ra", "dec", f"merged_step{step}_hats", temp_dir, client)

    return lsdb.read_hats(os.path.join(temp_dir, f"merged_step{step}_hats"))

def crossmatch_tiebreak_safe(*args, **kwargs):
    try:
        return crossmatch_tiebreak(*args, **kwargs)
    except RuntimeError as e:
        if "The output catalog is empty" in str(e):
            print(f"⚠️ {e} Proceeding by merging left and right without crossmatching.")
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

            from combine_redshift_dedup.packages.specz import import_catalog
            import_catalog(merged_path, "ra", "dec", f"merged_step{step}_hats", temp_dir, client)

            return lsdb.read_hats(os.path.join(temp_dir, f"merged_step{step}_hats"))
        else:
            raise