import argparse
import os
import re
import time
import json
import shutil
import pandas as pd
import numpy as np
from dask import delayed, compute
import dask.dataframe as dd
from pathlib import Path
from collections import defaultdict
from dask.distributed import Client

# === Imports from internal packages ===
from combine_redshift_dedup.packages.utils import (
    load_yml, dump_yml, setup_logger, set_global_logger,
    log_step, read_completed_steps,
    update_process_info, log_and_print,
    configure_warning_handler, configure_exception_hook,
    get_global_logger
)
from combine_redshift_dedup.packages.executor import get_executor
from combine_redshift_dedup.packages.specz import (
    prepare_catalog, import_catalog, generate_margin_cache_safe
)
from combine_redshift_dedup.packages.crossmatch import crossmatch_tiebreak_safe
from combine_redshift_dedup.packages.product_handle import save_dataframe
import lsdb

def main(config_path, cwd=".", base_dir_override=None):
    """
    Main pipeline function to process and combine redshift catalogs.

    Args:
        config_path (str): Path to the YAML configuration file.
        cwd (str): Working directory.
        base_dir_override (str): Base directory for outputs and logs.
    """
    delete_temp_files = True

    # === Load main config ===
    config = load_yml(config_path)
    param_config = config.get("param", {})
    
    if base_dir_override is None:
        raise ValueError("❌ You must specify --base_dir via the command line.")
    base_dir = base_dir_override

    output_root_dir = config["output_root_dir"]
    output_dir = config["output_dir"]
    output_root_dir_and_output_dir = os.path.join(output_root_dir, output_dir)
    output_name = config["output_name"]
    output_format = config.get('output_format', 'parquet').lower()
    
    logs_dir = os.path.join(base_dir, "process_info")
    temp_dir = os.path.join(base_dir, "temp")

    os.makedirs(output_root_dir_and_output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # === Load or initialize process.yml ===
    process_info_path = os.path.join(base_dir, "process.yml")
    if not os.path.exists(process_info_path):
        dump_yml(process_info_path, {})
    
    process_info = load_yml(process_info_path) or {}
    
    # Define default failure status early (will be overwritten at the end if successful)
    update_process_info(process_info, process_info_path, "status", "Failed")
    update_process_info(process_info, process_info_path, "start_time", str(pd.Timestamp.now()))

    # === Setup logger ===
    logger = setup_logger("combine_redshift_dedup", logdir=logs_dir)
    set_global_logger(logger)
    log_and_print(f"📄 Loading config from {config_path}", logger)
    configure_warning_handler(logger)
    configure_exception_hook(logger, process_info, process_info_path)

    # === Load translation configuration ===
    path_to_translation_file = param_config.get("flags_translation_file")
    if path_to_translation_file is None:
        logger.error("❌ Missing 'flags_translation_file' in config!")
        return
    translation_config = load_yml(path_to_translation_file)

    # === Set up log and comparison tracking files ===
    log_file = os.path.join(temp_dir, "process_resume.log")
    compared_to_path = os.path.join(temp_dir, "compared_to.json")

    # === Sort input catalogs by file size ===
    catalogs_unsorted = config["inputs"]["specz"]
    def get_file_size_mb(path):
        try:
            return os.path.getsize(path) / 1024 / 1024
        except Exception:
            return float("inf")

    catalogs_sorted = sorted(
        catalogs_unsorted,
        key=lambda entry: get_file_size_mb(entry["path"])
    )
    logger.info("📏 Catalogs sorted by disk size:")
    for entry in catalogs_sorted:
        size_mb = get_file_size_mb(entry["path"])
        logger.info(f" - {entry['internal_name']}: {size_mb:.1f} MB")

    catalogs = catalogs_sorted
    tiebreaking_priority = translation_config.get("tiebreaking_priority", [])
    instrument_type_priority = translation_config.get("instrument_type_priority", {})
    combine_mode = param_config.get("combine_type", "concatenate_and_mark_duplicates").lower()
    completed = read_completed_steps(log_file)

    # === Initialize Dask cluster ===
    cluster = get_executor(config["executor"], logs_dir=logs_dir)
    client = Client(cluster)
    
    # === Prepare catalogs in parallel using Dask Delayed ===
    logger.info(f"🧰 Preparing {len(catalogs)} input catalogs")
    
    delayed_prepared_paths = []
    
    for entry in catalogs:
        tag = f"prepare_{entry['internal_name']}"
        filename = os.path.basename(entry["path"])
        logger.info(f"📦 Preparing catalog: {entry['internal_name']} ({filename})")
    
        if tag not in completed:
            result = delayed(prepare_catalog)(entry, translation_config, temp_dir, combine_mode)
            delayed_prepared_paths.append(result)
        else:
            logger.info(f"⏩ Catalog {entry['internal_name']} already prepared. Skipping.")
            prepared_path = os.path.join(temp_dir, f"prepared_{entry['internal_name']}")
            match = re.match(r"(\d+)_", entry["internal_name"])
            if match:
                catalog_prefix = match.group(1)
                compared_path = os.path.join(temp_dir, f"compared_to_dict_{catalog_prefix}.json")
            else:
                compared_path = ""
            delayed_prepared_paths.append(delayed((prepared_path, "ra", "dec", entry["internal_name"], compared_path)))
    
    # === Trigger parallel computation ===
    results = compute(*delayed_prepared_paths)
    prepared_paths = [r[:4] for r in results]
    compared_to_paths = [r[4] for r in results]
    
    # === Log completed steps
    for entry in catalogs:
        tag = f"prepare_{entry['internal_name']}"
        if tag not in completed:
            log_step(log_file, tag)
    
    # === Merge all individual compared_to_dict files ===
    final_compared_to_dict = defaultdict(list)
    for path in compared_to_paths:
        if not path or not os.path.exists(path):
            logger.warning(f"⚠️ Missing compared_to_dict file: {path}")
            continue
        with open(path, "r") as f:
            local_dict = json.load(f)
        for k, v in local_dict.items():
            final_compared_to_dict[k].extend(v)
    
    # Deduplicate entries
    for k in final_compared_to_dict:
        final_compared_to_dict[k] = sorted(set(final_compared_to_dict[k]))
    
    # Save final dictionary
    with open(compared_to_path, "w") as f:
        json.dump(final_compared_to_dict, f)
    
    logger.info(f"📝 Saved final merged compared_to_dict to {compared_to_path}")

    # === Begin combination logic ===
    final_base_path = os.path.join(output_root_dir_and_output_dir, output_name)

    if combine_mode == "concatenate":
        logger.info("🔗 Combining catalogs by simple concatenation (concatenate mode)")
        dfs = [dd.read_parquet(p[0]) for p in prepared_paths]
        df_final = dd.concat(dfs).compute()

    elif combine_mode in ["concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"]:
        logger.info(f"🔍 Combining catalogs with duplicate marking ({combine_mode} mode)")

        delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)
        if not tiebreaking_priority:
            if delta_z_threshold is None or delta_z_threshold == 0.0:
                logger.error("❌ Cannot deduplicate: tiebreaking_priority is empty and delta_z_threshold is not set or is zero.")
                client.close(); cluster.close(); return
            else:
                logger.warning("⚠️ tiebreaking_priority is empty. Proceeding with delta_z_threshold tie-breaking only.")

        # === Import first catalog ===
        import_tag = f"import_{prepared_paths[0][3]}"
        if import_tag not in completed:
            logger.info(f"📥 Importing first catalog: {prepared_paths[0][3]}")
            import_catalog(prepared_paths[0][0], "ra", "dec", "cat0_hats", temp_dir, client)
            log_step(log_file, import_tag)

        for j in reversed(range(1, len(prepared_paths))):
            merged_path = os.path.join(temp_dir, f"merged_step{j}_hats")
            if f"crossmatch_step{j}" in completed and os.path.exists(merged_path):
                cat_prev = lsdb.read_hats(merged_path)
                logger.info(f"🔁 Resuming from previous merged result: {merged_path}")
                start_i = j + 1
                break
        else:
            cat_prev = lsdb.read_hats(os.path.join(temp_dir, "cat0_hats"))
            start_i = 1

        # === Iterative crossmatching ===
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
                        os.path.join(temp_dir, f"cat{i}_hats"), temp_dir, f"cat{i}_margin", client
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
                    final_compared_to_dict, instrument_type_priority, translation_config
                )

                if delete_temp_files and i > 1:
                    for path in [
                        f"merged_step{i-1}", f"merged_step{i-1}_hats",
                        f"cat{i}_hats", f"cat{i}_margin", f"xmatch_step{i}"
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
            client.close(); cluster.close(); return

        compared_to_dict = final_compared_to_dict

        df_final = dd.read_parquet(final_merged).compute()
        df_final["compared_to"] = df_final["CRD_ID"].map(lambda x: ",".join(compared_to_dict.get(str(x), [])))

        if combine_mode == "concatenate_and_remove_duplicates":
            before = len(df_final)
            df_final = df_final[df_final["tie_result"] == 1]
            after = len(df_final)
            logger.info(f"🧹 Removed duplicates: kept {after} of {before} rows (tie_result == 1)")

    else:
        logger.error(f"❌ Unknown combine_mode: {combine_mode}")
        client.close(); cluster.close(); return

    # === Final cleanup and save ===
    if combine_mode == "concatenate" and "tie_result" in df_final.columns:
        logger.info("ℹ️ Dropping column 'tie_result' (not needed for combine_mode == concatenate)")
        df_final = df_final.drop(columns=["tie_result"])

    for col in df_final.columns:
        dtype = df_final[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            all_missing = df_final[col].apply(lambda x: pd.isna(x) or str(x).strip() == "").all()
        else:
            all_missing = df_final[col].isna().all()
        if all_missing:
            logger.info(f"ℹ️ Dropping column '{col}' (all values missing or empty)")
            df_final = df_final.drop(columns=[col])

    save_dataframe(df_final, final_base_path, output_format)
    logger.info(f"✅ Final combined catalog saved at {final_base_path}.{output_format}")

    relative_path = os.path.join(output_dir, f"{output_name}.{output_format}")

    expected_columns = ["id", "ra", "dec", "z", "z_flag", "z_err", "survey"]
    columns_assoc = {col: col for col in expected_columns if col in df_final.columns}

    update_process_info(process_info, process_info_path, "outputs", [{
        "path": relative_path,
        "root_dir": output_root_dir,
        "role": "main",
        "columns_assoc": columns_assoc
    }])
    update_process_info(process_info, process_info_path, "end_time", str(pd.Timestamp.now()))
    update_process_info(process_info, process_info_path, "status", "Successful")

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
    parser.add_argument("--base_dir", required=True, help="Base directory for outputs and logs")
    args = parser.parse_args()

    start_time = time.time()
    main(args.config_path, args.cwd, args.base_dir)
    duration = time.time() - start_time
    logger = get_global_logger()
    log_and_print(f"✅ Pipeline completed in {duration:.2f} seconds.", logger)