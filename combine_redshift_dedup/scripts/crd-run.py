import argparse
import os
import time
import json
import shutil
import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path
from collections import defaultdict

from dask.distributed import Client

from combine_redshift_dedup.packages.utils import (
    load_yml, dump_yml, setup_logger, log_step, read_completed_steps
)
from combine_redshift_dedup.packages.executor import get_executor
from combine_redshift_dedup.packages.specz import (
    prepare_catalog, import_catalog, generate_margin_cache_safe
)
from combine_redshift_dedup.packages.crossmatch import (
    crossmatch_tiebreak_safe
)

from combine_redshift_dedup.packages.product_handle import save_dataframe

import lsdb

def main(config_path, cwd=".", base_dir_override=None):
    """
    Main pipeline function to process and combine redshift catalogs.

    Args:
        config_path (str): Path to the YAML configuration file.
        cwd (str): Working directory, defaults to current directory.

    Workflow:
        - Load configuration files.
        - Prepare individual catalogs (standardization, translation, etc.).
        - Combine catalogs using one of the supported modes:
          concatenate, concatenate_and_mark_duplicates, concatenate_and_remove_duplicates.
        - Clean and save the final combined catalog.
    """

    delete_temp_files = True  # Flag to clean temporary files along the pipeline

    # Load main config first to get directories
    config = load_yml(config_path)
    base_dir = base_dir_override if base_dir_override else config["base_dir"]
    output_dir = os.path.join(base_dir, config["output_dir"])
    logs_dir = os.path.join(base_dir, config["logs_dir"])
    temp_dir = os.path.join(base_dir, config["temp_dir"])

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Now setup logger pointing to logs_dir
    logger = setup_logger("combine_redshift_dedup", logdir=logs_dir)
    logger.info(f"📄 Loading config from {config_path}")

    # Load translation config
    path_to_translation_file = config.get("flags_translation_file")
    if path_to_translation_file is None:
        logger.error("❌ Missing 'flags_translation_file' in config!")
        return
    translation_config = load_yml(path_to_translation_file)

    log_file = os.path.join(temp_dir, "process_resume.log")
    compared_to_path = os.path.join(temp_dir, "compared_to.json")

    catalogs = config["inputs"]["specz"]
    tiebreaking_priority = translation_config.get("tiebreaking_priority", [])
    type_priority = translation_config.get("type_priority", {})
    combine_type = config.get("combine_type", "concatenate_and_mark_duplicates").lower()
    output_format = config.get('output_format', 'parquet').lower()

    completed = read_completed_steps(log_file)

    # Initialize Dask cluster and client
    cluster = get_executor(config["executor"], logs_dir=logs_dir)
    client = Client(cluster)

    # Load or initialize compared_to dict
    if os.path.exists(compared_to_path):
        with open(compared_to_path, "r") as f:
            compared_to_dict = json.load(f)
        compared_to_dict = defaultdict(list, compared_to_dict)
    else:
        compared_to_dict = defaultdict(list)

    prepared_paths = []

    logger.info(f"🧰 Preparing {len(catalogs)} input catalogs")

    # Prepare each individual catalog
    for entry in catalogs:
        tag = f"prepare_{entry['internal_name']}"
        filename = os.path.basename(entry["path"])
        logger.info(f"📦 Preparing catalog: {entry['internal_name']} ({filename})")

        if tag not in completed:
            path_info = prepare_catalog(entry, translation_config, temp_dir, compared_to_dict, combine_type)
            log_step(log_file, tag)
        else:
            logger.info(f"⏩ Catalog {entry['internal_name']} already prepared. Skipping.")
            path_info = (
                os.path.join(temp_dir, f"prepared_{entry['internal_name']}"),
                "ra",
                "dec",
                entry["internal_name"]
            )
        prepared_paths.append(path_info)

    final_base_path = os.path.join(output_dir, config['output_name'])

    if combine_type == "concatenate":
        logger.info("🔗 Combining catalogs by simple concatenation (concatenate mode)")
        dfs = [dd.read_parquet(p[0]) for p in prepared_paths]
        df_final = dd.concat(dfs)
        df_final = df_final.compute()

    elif combine_type in ["concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"]:
        logger.info(f"🔍 Combining catalogs with duplicate marking ({combine_type} mode)")

        # Validate deduplication criteria
        delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)
        if not tiebreaking_priority:
            if delta_z_threshold is None or delta_z_threshold == 0.0:
                logger.error("❌ Cannot deduplicate: tiebreaking_priority is empty and delta_z_threshold is not set or is zero.")
                client.close()
                cluster.close()
                return
            else:
                logger.warning("⚠️ tiebreaking_priority is empty. Proceeding with delta_z_threshold tie-breaking only.")

        # Import first catalog (cat0)
        import_tag = f"import_{prepared_paths[0][3]}"
        if import_tag not in completed:
            logger.info(f"📥 Importing first catalog: {prepared_paths[0][3]}")
            import_catalog(prepared_paths[0][0], "ra", "dec", "cat0_hats", temp_dir, client)
            log_step(log_file, import_tag)

        # Recover most recent cat_prev
        for j in reversed(range(1, len(prepared_paths))):
            xmatch_tag = f"crossmatch_step{j}"
            merged_path = os.path.join(temp_dir, f"merged_step{j}_hats")
            if xmatch_tag in completed and os.path.exists(merged_path):
                cat_prev = lsdb.read_hats(merged_path)
                logger.info(f"🔁 Resuming from previous merged result: {merged_path}")
                start_i = j + 1
                break
        else:
            cat_prev = lsdb.read_hats(os.path.join(temp_dir, "cat0_hats"))
            start_i = 1

        for i in range(start_i, len(prepared_paths)):
            xmatch_tag = f"crossmatch_step{i}"
            internal_name = prepared_paths[i][3]
            filename = os.path.basename(prepared_paths[i][0])

            if xmatch_tag not in completed:
                logger.info(f"📥 Importing catalog: {internal_name} ({filename})")
                import_tag = f"import_{internal_name}"
                if import_tag not in completed:
                    import_catalog(prepared_paths[i][0], "ra", "dec", f"cat{i}_hats", temp_dir, client)
                    log_step(log_file, import_tag)

                margin_tag = f"margin_cache_{internal_name}"
                if margin_tag not in completed:
                    logger.info(f"🧩 Generating margin cache for: {internal_name}")
                    margin_cache_path = generate_margin_cache_safe(
                        os.path.join(temp_dir, f"cat{i}_hats"),
                        temp_dir,
                        f"cat{i}_margin",
                        client
                    )
                    if margin_cache_path:
                        log_step(log_file, margin_tag)
                else:
                    margin_cache_path = os.path.join(temp_dir, f"cat{i}_margin")

                if margin_cache_path and os.path.exists(margin_cache_path):
                    cat_curr = lsdb.read_hats(os.path.join(temp_dir, f"cat{i}_hats"), margin_cache=margin_cache_path)
                else:
                    logger.warning(f"⚠️ No margin cache found for {internal_name}. Proceeding without it.")
                    cat_curr = lsdb.read_hats(os.path.join(temp_dir, f"cat{i}_hats"))

                logger.info(f"🔄 Crossmatching previous result with: {internal_name}")
                cat_prev = crossmatch_tiebreak_safe(
                    cat_prev, cat_curr, tiebreaking_priority, temp_dir, i, client,
                    compared_to_dict, type_priority, translation_config
                )

                if delete_temp_files and i > 1:
                    for path in [
                        f"merged_step{i-1}",
                        f"merged_step{i-1}_hats",
                        f"cat{i}_hats",
                        f"cat{i}_margin",
                        f"xmatch_step{i}"
                    ]:
                        full_path = os.path.join(temp_dir, path)
                        if os.path.exists(full_path):
                            shutil.rmtree(full_path)
                            logger.info(f"🗑️ Deleted temporary directory {full_path}")

                log_step(log_file, xmatch_tag)
            else:
                logger.info(f"⏩ Skipping already completed step: {xmatch_tag}")

        final_merged = os.path.join(temp_dir, f"merged_step{len(prepared_paths)-1}")
        if not os.path.exists(final_merged):
            logger.error(f"❌ Final merged Parquet folder not found: {final_merged}")
            client.close()
            cluster.close()
            return

        if os.path.exists(compared_to_path):
            with open(compared_to_path, "r") as f:
                compared_to_dict = json.load(f)
        else:
            compared_to_dict = {}

        df_final = dd.read_parquet(final_merged).compute()
        df_final["compared_to"] = df_final["CRD_ID"].map(lambda x: ",".join(compared_to_dict.get(str(x), [])))

        if combine_type == "concatenate_and_remove_duplicates":
            before = len(df_final)
            df_final = df_final[df_final["tie_result"] == 1]
            after = len(df_final)
            logger.info(f"🧹 Removed duplicates: kept {after} of {before} rows (tie_result == 1)")

    else:
        logger.error(f"❌ Unknown combine_type: {combine_type}")
        client.close()
        cluster.close()
        return

    if combine_type == "concatenate" and "tie_result" in df_final.columns:
        logger.info("ℹ️ Dropping column 'tie_result' (not needed for combine_type == concatenate)")
        df_final = df_final.drop(columns=["tie_result"])

    # Drop columns that are fully NaN or empty before saving
    for col in df_final.columns:
        dtype = df_final[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            all_missing = df_final[col].apply(lambda x: pd.isna(x) or str(x).strip() == "").all()
        else:
            all_missing = df_final[col].isna().all()
        if all_missing:
            logger.info(f"ℹ️ Dropping column '{col}' (all values missing or empty)")
            df_final = df_final.drop(columns=[col])

    # Save the cleaned final catalog
    save_dataframe(df_final, final_base_path, output_format)
    logger.info(f"✅ Final combined catalog saved at {final_base_path}.{output_format}")

    if delete_temp_files:
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"🧹 Deleted entire temp_dir {temp_dir} after successful pipeline completion")
        except Exception as e:
            logger.warning(f"⚠️ Could not delete temp_dir {temp_dir}: {e}")

    client.close()
    cluster.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine redshift catalogs with deduplication or concatenation.")
    parser.add_argument("config_path", help="Path to YAML config file")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory (default: current dir)")
    parser.add_argument("--base_dir", default=None, help="Override base_dir from config file")
    args = parser.parse_args()

    start_time = time.time()
    main(args.config_path, args.cwd, args.base_dir)
    duration = time.time() - start_time
    print(f"✅ Pipeline completed in {duration:.2f} seconds.")