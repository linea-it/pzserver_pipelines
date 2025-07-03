import os
import json
import numpy as np
import pandas as pd
import dask.dataframe as dd
import lsdb

def crossmatch_tiebreak(left_cat, right_cat, tiebreaking_priority, temp_dir, step, client, compared_to_dict, type_priority):
    output_xmatch_dir = os.path.join(temp_dir, f"xmatch_step{step}")

    # Perform crossmatch
    xmatched = left_cat.crossmatch(
        right_cat,
        radius_arcsec=1.0,
        n_neighbors=1,
        suffixes=("left", "right")
    )
    xmatched.to_hats(output_xmatch_dir, overwrite=True)
    df = lsdb.read_hats(output_xmatch_dir)._ddf

    # Filter out no-match rows
    df = df[(df["tie_resultleft"] != 0) & (df["tie_resultright"] != 0)]

    delta_z_threshold = 0.0  # Pode passar como argumento ou pegar do config
    if hasattr(right_cat, 'delta_z_threshold'):
        delta_z_threshold = right_cat.delta_z_threshold

    def decide_tie(row):
        left_was_tie = row.get("tie_resultleft", 1) == 2
        right_was_tie = row.get("tie_resultright", 1) == 2
    
        zflag_left = row.get("z_flag_homogenizedleft")
        zflag_right = row.get("z_flag_homogenizedright")
    
        # Se ambos são 6, zera os dois
        if pd.notna(zflag_left) and zflag_left == 6 and pd.notna(zflag_right) and zflag_right == 6:
            return (0, 0)
    
        # Se left é 6
        elif pd.notna(zflag_left) and zflag_left == 6:
            return (0, 2 if right_was_tie else 1)
    
        # Se right é 6
        elif pd.notna(zflag_right) and zflag_right == 6:
            return (2 if left_was_tie else 1, 0)
    
        # Standard tiebreaking
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
    
        # Se não decidiu, marca ambos como 2 para avaliar delta_z depois
        return (2, 2)

    tie_results = df.map_partitions(
        lambda p: p.apply(decide_tie, axis=1, result_type="expand"),
        meta={0: "i8", 1: "i8"}
    )
    df = df.assign(tie_left=tie_results[0], tie_right=tie_results[1])

    # Avalia delta_z onde tie = (2,2)
    def apply_delta_z_fix(p):
        p = p.copy()
        mask = (p["tie_left"] == 2) & (p["tie_right"] == 2)
        for idx, row in p[mask].iterrows():
            z1 = row["zleft"]
            z2 = row["zright"]
            if pd.notnull(z1) and pd.notnull(z2):
                if abs(z1 - z2) <= delta_z_threshold:
                    # Mantém left, elimina right
                    p.at[idx, "tie_left"] = 1
                    p.at[idx, "tie_right"] = 0
        return p

    df = df.map_partitions(apply_delta_z_fix)

    pairs = df[["idleft", "idright", "tie_left", "tie_right"]].compute()

    # Update compared_to_dict
    for _, row in pairs.iterrows():
        compared_to_dict.setdefault(str(row["idleft"]), []).append(str(row["idright"]))
        compared_to_dict.setdefault(str(row["idright"]), []).append(str(row["idleft"]))

    compared_to_path = os.path.join(temp_dir, "compared_to.json")
    with open(compared_to_path, "w") as f:
        json.dump(compared_to_dict, f)

    # Prepare final tie_result map
    expanded = pd.concat([
        pairs[["idleft", "tie_left"]].rename(columns={"idleft": "id", "tie_left": "tie_result"}),
        pairs[["idright", "tie_right"]].rename(columns={"idright": "id", "tie_right": "tie_result"})
    ])

    def consolidate_results(series):
        if (series == 0).any():
            return 0
        if (series == 2).any():
            return 2
        return 1

    final_results = expanded.groupby("id")["tie_result"].apply(consolidate_results)

    def apply_final(df_part):
        df_part["tie_result"] = df_part["id"].map(final_results).fillna(df_part["tie_result"])
        df_part["compared_to"] = df_part["id"].map(lambda x: ",".join(compared_to_dict.get(str(x), [])))
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