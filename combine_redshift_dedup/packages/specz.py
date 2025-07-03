# combine_redshift_dedup/packages/specz.py

import os
import glob
import numpy as np
import pandas as pd
import dask.dataframe as dd

from hats_import.pipeline import ImportArguments, pipeline_with_client
from hats_import.catalog.file_readers import ParquetReader
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
import hats

from combine_redshift_dedup.packages.product_handle import ProductHandle

def prepare_catalog(entry, translation_config, temp_dir, combine_type="resolve_duplicates"):
    ph = ProductHandle(entry["path"])
    df = ph.to_ddf()
    product_name = entry["internal_name"]

    # Renomeia colunas
    col_map = {v: k for k, v in entry["columns"].items() if v and v in df.columns}
    df = df.rename(columns=col_map)

    # Adiciona source
    df["source"] = product_name

    # Garante colunas padrão
    for col in ["id", "ra", "dec", "z", "z_flag", "z_err", "type", "survey"]:
        if col not in df.columns:
            df[col] = np.nan

    df["id"] = df["id"].astype(str)
    df["ra"] = df["ra"].astype(float)
    df["dec"] = df["dec"].astype(float)
    df["z"] = df["z"].astype(float)
    df["z_err"] = df["z_err"].astype(float)
    df["z_flag"] = df["z_flag"].astype(str)

    # Inicializa tie_result e homogenizados
    df["tie_result"] = 1
    df["z_flag_homogenized"] = np.nan
    df["type_homogenized"] = np.nan

    # ===============================
    # LÓGICA DE ELIMINAÇÃO DE DUPLICADOS POR ID
    # ===============================
    delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)
    dup_counts = df.groupby("id").size().compute()
    ids_dup = dup_counts[dup_counts > 1].index.tolist()

    if ids_dup:
        print(f"⚠️ {product_name}: Found duplicate ids. Applying delta_z_threshold = {delta_z_threshold}")

        df_dup_local = df[df["id"].isin(ids_dup)].compute()

        delta_status = (
            df_dup_local.groupby("id")["z"]
            .apply(lambda z: "diff" if z.max() - z.min() > delta_z_threshold else "same")
        )

        ids_same = delta_status[delta_status == "same"].index.tolist()
        ids_diff = delta_status[delta_status == "diff"].index.tolist()

        def mark_tie_result(partition):
            partition = partition.copy()
            dup_ids_same = partition["id"].isin(ids_same)
            dup_ids_diff = partition["id"].isin(ids_diff)

            # Para grupos com z consistente: first = 1, resto = 0
            partition["dup_rank"] = partition.groupby("id").cumcount()
            same_mask = dup_ids_same & (partition["dup_rank"] > 0)
            partition.loc[same_mask, "tie_result"] = 0
            partition = partition.drop(columns="dup_rank")

            # Para grupos com z inconsistente: todos = 2
            partition.loc[dup_ids_diff, "tie_result"] = 2

            return partition

        df = df.map_partitions(mark_tie_result)

        if ids_diff:
            print(f"⚠️ {product_name}: Duplicate ids with differing z beyond threshold found: {ids_diff[:5]} ...")

    # ===============================
    # HOMOGENEIZAÇÃO
    # ===============================
    def apply_translation(row, key):
        survey = row.get("survey")
        rules = translation_config.get("translation_rules", {}).get(survey, {})
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

    # ===============================
    # FINALIZAÇÃO
    # ===============================
    final_cols = [
        "id", "ra", "dec", "z", "z_flag", "z_err", "type", "survey",
        "source", "tie_result", "z_flag_homogenized", "type_homogenized"
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