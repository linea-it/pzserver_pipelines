import os
import json
import numpy as np
import pandas as pd
import dask.dataframe as dd
import lsdb

def crossmatch_tiebreak(left_cat, right_cat, tiebreaking_priority, temp_dir, step, client, compared_to_dict, type_priority):
    output_xmatch_dir = os.path.join(temp_dir, f"xmatch_step{step}")

    xmatched = left_cat.crossmatch(
        right_cat,
        radius_arcsec=1.0,
        n_neighbors=1,
        suffixes=("left", "right")
    )
    xmatched.to_hats(output_xmatch_dir, overwrite=True)
    df = lsdb.read_hats(output_xmatch_dir)._ddf

    df = df[(df["tie_resultleft"] != 0) & (df["tie_resultright"] != 0)]

    def decide_tie(row):
        left_was_tie = row.get("tie_resultleft", 1) == 2
        right_was_tie = row.get("tie_resultright", 1) == 2

        for i, col in enumerate(tiebreaking_priority):
            v1 = row.get(f"{col}left")
            v2 = row.get(f"{col}right")

            if col.startswith("z_flag_homogenized"):
                if v1 == 6: v1 = np.nan
                if v2 == 6: v2 = np.nan

            if col == "type_homogenized":
                v1 = type_priority.get(v1, 0)
                v2 = type_priority.get(v2, 0)

            if pd.notnull(v1) and pd.notnull(v2):
                if v1 > v2:
                    return 2 if left_was_tie else 1
                elif v2 > v1:
                    return 2 if right_was_tie else 0
                else:
                    continue
            elif pd.notnull(v1):
                continue
            elif pd.notnull(v2):
                continue

        return 2

    df = df.assign(
        tie_result=df.map_partitions(
            lambda p: p.apply(decide_tie, axis=1),
            meta=("tie_result", "i8")
        )
    )

    pairs = df[["idleft", "idright", "tie_result"]].compute()

    for _, row in pairs.iterrows():
        compared_to_dict.setdefault(str(row["idleft"]), []).append(str(row["idright"]))
        compared_to_dict.setdefault(str(row["idright"]), []).append(str(row["idleft"]))

    # Save compared_to_dict
    compared_to_path = os.path.join(temp_dir, "compared_to.json")
    with open(compared_to_path, "w") as f:
        json.dump(compared_to_dict, f)

    # Build final tiebreak results
    expanded = []
    for _, row in pairs.iterrows():
        if row["tie_result"] == 1:
            expanded.append({"id": row["idleft"], "tie_result": 1})
            expanded.append({"id": row["idright"], "tie_result": 0})
        elif row["tie_result"] == 0:
            expanded.append({"id": row["idleft"], "tie_result": 0})
            expanded.append({"id": row["idright"], "tie_result": 1})
        else:
            expanded.append({"id": row["idleft"], "tie_result": 2})
            expanded.append({"id": row["idright"], "tie_result": 2})

    all_results = pd.DataFrame(expanded)

    def consolidate_results(series):
        if (series == 0).any():
            return 0
        if (series == 2).any():
            return 2
        return 1

    final_results = all_results.groupby("id")["tie_result"].apply(consolidate_results)

    def apply_final_result(df):
        df["tie_result"] = df["id"].map(final_results).fillna(df["tie_result"])
        df["compared_to"] = df["id"].map(lambda x: ",".join(compared_to_dict.get(str(x), [])))
        return df

    left_df = left_cat._ddf.map_partitions(apply_final_result)
    right_df = right_cat._ddf.map_partitions(apply_final_result)

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