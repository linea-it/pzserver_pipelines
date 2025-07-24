import argparse
import os
import re
import time
import json
import shutil
import warnings
import pandas as pd
import numpy as np
from dask import delayed, compute
import dask.dataframe as dd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dask.distributed import Client, performance_report

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
    logger.info(f"📄 Loading config from {config_path}")
    configure_warning_handler(logger)

    # Suppress only the "Sending large graph" warnings from Dask
    warnings.filterwarnings(
        "ignore",
        message=".*Sending large graph of size.*",
        category=UserWarning,
        module="distributed"
    )
    
    configure_exception_hook(logger, process_info, process_info_path)

    logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: pipeline_init id=pipeline_init")

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
    
    # === Global Dask time profile report ===
    global_report_path = os.path.join(logs_dir, "main_dask_report.html")
    
    with performance_report(filename=global_report_path):
    
        logger.info(f"🧰 Preparing {len(catalogs)} input catalogs")
    
        futures = []
        
        for entry in catalogs:
            tag = f"prepare_{entry['internal_name']}"
            filename = os.path.basename(entry["path"])
            logger.info(f"📦 Preparing catalog: {entry['internal_name']} ({filename})")
        
            if tag not in completed:
                future = client.submit(
                    prepare_catalog,
                    entry,
                    translation_config,
                    logs_dir,
                    temp_dir,
                    combine_mode  # ✅ não passe client
                )
                futures.append(future)
            else:
                logger.info(f"⏩ Catalog {entry['internal_name']} already prepared. Skipping.")
                
                match = re.match(r"(\d+)_", entry["internal_name"])
                if match:
                    catalog_prefix = match.group(1)
                    artifact_hats = f"cat{catalog_prefix}_hats"
                    artifact_margin = f"cat{catalog_prefix}_margin"
                    hats_path = os.path.join(temp_dir, artifact_hats)
                    margin_path = os.path.join(temp_dir, artifact_margin)
                    compared_path = os.path.join(temp_dir, f"compared_to_dict_{catalog_prefix}.json")
                else:
                    raise ValueError(
                        f"❌ Could not extract numeric prefix from internal_name '{entry['internal_name']}' "
                        f"(expected format: '001_catalogname'). Cannot generate HATS/margin/cache paths."
                    )
        
                futures.append(
                    client.submit(
                        lambda x: x,
                        (hats_path, "ra", "dec", entry["internal_name"], margin_path, compared_path)
                    )
                )
    
        # === Trigger parallel execution and wait for results ===
        preparation_report_path = os.path.join(logs_dir, "preparation_dask_report.html")
        
        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: pipeline_init id=pipeline_init")
        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: prepare_catalogs id=prepare_catalogs")
        
        results = client.gather(futures)
        
        # === Unpack returned values
        prepared_hats_paths = [r[:4] for r in results]          # hats_path, ra, dec, internal_name
        prepared_cache_paths = [r[4] for r in results]          # margin_cache_path
        compared_to_paths = [r[5] for r in results]             # compared_to_path
        prepared_paths = [(os.path.join(temp_dir, f"prepared_{r[3]}"), r[1], r[2], r[3]) for r in results]  # traditional Parquet
        
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
        
        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: prepare_catalogs id=prepare_catalogs")
        
        if combine_mode == "concatenate":
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: consolidate id=consolidate")
            logger.info("🔗 Combining catalogs by simple concatenation (concatenate mode)")
            dfs = [dd.read_parquet(p[0]) for p in prepared_paths]
            df_final = dd.concat(dfs).compute()
        
        elif combine_mode in ["concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"]:
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: crossmatch_catalogs id=crossmatch_catalogs")
            logger.info(f"🔍 Combining catalogs with duplicate marking ({combine_mode} mode)")
        
            delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)
            if not tiebreaking_priority:
                if delta_z_threshold is None or delta_z_threshold == 0.0:
                    logger.error("❌ Cannot deduplicate: tiebreaking_priority is empty and delta_z_threshold is not set or is zero.")
                    client.close(); cluster.close(); return
                else:
                    logger.warning("⚠️ tiebreaking_priority is empty. Proceeding with delta_z_threshold tie-breaking only.")
        
            # === Load initial catalog
            initial_hats_path = prepared_hats_paths[0][0]
            initial_margin_path = prepared_cache_paths[0]
            
            if (
                initial_margin_path
                and os.path.exists(initial_margin_path)
                and os.path.exists(os.path.join(initial_margin_path, "properties"))
            ):
                cat_prev = lsdb.read_hats(initial_hats_path, margin_cache=initial_margin_path)
            else:
                if not os.path.exists(initial_margin_path):
                    logger.warning(f"⚠️ Margin cache folder not found: {initial_margin_path}")
                else:
                    logger.warning(f"⚠️ Margin cache exists but has no 'properties' file: {initial_margin_path}")
                cat_prev = lsdb.read_hats(initial_hats_path)

            for j in reversed(range(1, len(prepared_hats_paths))):
                merged_path = os.path.join(temp_dir, f"merged_step{j}_hats")
                if f"crossmatch_step{j}" in completed and os.path.exists(merged_path):
                    cat_prev = lsdb.read_hats(merged_path)
                    logger.info(f"🔁 Resuming from previous merged result: {merged_path}")
                    start_i = j + 1
                    break
            else:
                start_i = 1
        
            # === Iterative crossmatching ===
            for i in range(start_i, len(prepared_hats_paths)):
                xmatch_tag = f"crossmatch_step{i}"
                hats_path = prepared_hats_paths[i][0]
                margin_path = prepared_cache_paths[i]
                internal_name = prepared_hats_paths[i][3]
                filename = os.path.basename(hats_path)
            
                if xmatch_tag not in completed:
                    # === Verifica se margin cache é válido
                    if margin_path and os.path.exists(margin_path) and os.path.exists(os.path.join(margin_path, "properties")):
                        cat_curr = lsdb.read_hats(hats_path, margin_cache=margin_path)
                    else:
                        logger.warning(f"⚠️ No valid margin cache found for {internal_name}. Proceeding without it.")
                        cat_curr = lsdb.read_hats(hats_path)
            
                    logger.info(f"🔄 Crossmatching previous result with: {internal_name}")
                    is_last = (i == len(prepared_hats_paths) - 1)
            
                    cat_prev = crossmatch_tiebreak_safe(
                        cat_prev,
                        cat_curr,
                        tiebreaking_priority,
                        logs_dir,
                        temp_dir,
                        i,
                        client,
                        final_compared_to_dict,
                        instrument_type_priority,
                        translation_config,
                        do_import=not is_last  # necessário para salvar merged_stepN
                    )
            
                    # === Deletar temporários se necessário
                    if delete_temp_files and i > 1:
                        match = re.match(r"(\d+)_", internal_name)
                        if not match:
                            raise ValueError(f"❌ Could not extract numeric prefix from internal_name '{internal_name}'")
                        prefix = match.group(1)
            
                        for path in [
                            f"merged_step{i-1}", f"merged_step{i-1}_hats",
                            f"cat{prefix}_hats", f"cat{prefix}_margin", f"xmatch_step{i}"
                        ]:
                            full_path = os.path.join(temp_dir, path)
                            if os.path.exists(full_path):
                                shutil.rmtree(full_path)
                                logger.info(f"🗑️ Deleted temporary directory {full_path}")
            
                    log_step(log_file, xmatch_tag)
                else:
                    logger.info(f"⏩ Skipping already completed step: {xmatch_tag}")

            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: crossmatch_catalogs id=crossmatch_catalogs")
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: consolidate id=consolidate")
        
            final_merged = os.path.join(temp_dir, f"merged_step{len(prepared_paths)-1}")
            if not os.path.exists(final_merged):
                logger.error(f"❌ Final merged Parquet folder not found: {final_merged}")
                client.close(); cluster.close(); return
        
            df_final = dd.read_parquet(final_merged).compute()
        
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

    logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: consolidate id=consolidate")

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