# combine_redshift_dedup/packages/specz.py

import os
import glob
import numpy as np
import dask.dataframe as dd
from hats_import.pipeline import ImportArguments, pipeline_with_client
from hats_import.catalog.file_readers import ParquetReader
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
import hats

from combine_redshift_dedup.packages.product_handle import ProductHandle

def prepare_catalog(entry, translation_config, temp_dir):
    """
    Load, rename and translate columns of a catalog. Save the intermediate Parquet result.

    Args:
        entry (dict): Catalog configuration entry from YAML.
        translation_config (dict): Translation rules from flags_translation.yaml.
        temp_dir (str): Directory to save the prepared Parquet.

    Returns:
        tuple: (Parquet path, RA column name, DEC column name, internal catalog name)
    """
    # Use ProductHandle to read input file
    ph = ProductHandle(entry["path"])
    df = ph.to_ddf()

    product_name = entry["internal_name"]

    # Add source column
    df["source"] = product_name

    # Rename columns according to mapping
    col_map = {v: k for k, v in entry["columns"].items() if v and v in df.columns}
    df = df.rename(columns=col_map)

    # Ensure standard columns exist
    for col in ["id", "ra", "dec", "z", "z_flag", "z_err", "type", "survey"]:
        if col not in df.columns:
            df[col] = np.nan

    # Enforce types
    df["id"] = df["id"].astype(str)
    df["ra"] = df["ra"].astype(float)
    df["dec"] = df["dec"].astype(float)
    df["z"] = df["z"].astype(float)
    df["z_err"] = df["z_err"].astype(float)
    df["z_flag"] = df["z_flag"].astype(str)

    # Initialize control columns
    df["tie_result"] = 1
    df["z_flag_homogenized_ozdes"] = np.nan
    df["z_flag_homogenized_vvds"] = np.nan
    df["type_homogenized"] = np.nan

    # Apply translation row by row
    def translate_row(row):
        survey = row.get("survey", None)
        rules = translation_config.get("translation_rules", {}).get(survey, {})
        zflag = row.get("z_flag", None)

        z_ozdes = np.nan
        z_vvds = np.nan
        type_homog = np.nan

        def process_flag(rule, zflag_val, row):
            if isinstance(rule, dict):
                direct = {k: v for k, v in rule.items() if k != "conditions"}
                if zflag_val in direct:
                    return direct[zflag_val]
                if "conditions" in rule:
                    for cond in rule["conditions"]:
                        try:
                            if eval(cond["expr"], {}, {"row": row}):
                                return cond["value"]
                        except Exception:
                            continue
            return np.nan

        if "z_flag_translation" in rules:
            if "ozdes" in rules["z_flag_translation"]:
                z_ozdes = process_flag(rules["z_flag_translation"]["ozdes"], zflag, row)
            if "vvds" in rules["z_flag_translation"]:
                z_vvds = process_flag(rules["z_flag_translation"]["vvds"], zflag, row)

        if "type_translation" in rules:
            tval = row.get("type", None)
            tmap = rules["type_translation"]
            type_homog = tmap.get(tval, tmap.get("default", np.nan)) if isinstance(tmap, dict) else tmap

        return {
            "z_flag_homogenized_ozdes": z_ozdes,
            "z_flag_homogenized_vvds": z_vvds,
            "type_homogenized": type_homog
        }

    translated = df.map_partitions(lambda p: p.apply(translate_row, axis=1, result_type="expand"),
                                   meta={
                                       "z_flag_homogenized_ozdes": "f8",
                                       "z_flag_homogenized_vvds": "f8",
                                       "type_homogenized": "object"
                                   })

    df = df.assign(
        z_flag_homogenized_ozdes=translated["z_flag_homogenized_ozdes"],
        z_flag_homogenized_vvds=translated["z_flag_homogenized_vvds"],
        type_homogenized=translated["type_homogenized"]
    )

    final_cols = [
        "id", "ra", "dec", "z", "z_flag", "z_err", "type", "survey",
        "source", "tie_result", "z_flag_homogenized_ozdes",
        "z_flag_homogenized_vvds", "type_homogenized"
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