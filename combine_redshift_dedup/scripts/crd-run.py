# =====================
# Built-in modules
# =====================
import argparse
import json
import os
import re
import shutil
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# =====================
# Third-party libraries
# =====================
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import compute, delayed
from dask.distributed import Client, performance_report
import lsdb

# =====================
# Project-specific libraries
# =====================
from combine_redshift_dedup.packages.crossmatch import crossmatch_tiebreak_safe
from combine_redshift_dedup.packages.executor import get_executor
from combine_redshift_dedup.packages.product_handle import save_dataframe
from combine_redshift_dedup.packages.specz import (
    prepare_catalog,
    USE_ARROW_TYPES, DTYPE_STR, DTYPE_FLOAT, DTYPE_INT, DTYPE_BOOL, DTYPE_INT8,
)

from combine_redshift_dedup.packages.utils import (
    configure_exception_hook,
    configure_warning_handler,
    dump_yml,
    get_global_logger,
    load_yml,
    log_and_print,
    log_step,
    read_completed_steps,
    set_global_logger,
    setup_logger,
    update_process_info
)


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
        
        # === Unpack into a unified structure per catalog
        prepared_catalogs_info = [
            {
                "hats_path": r[0],
                "ra": r[1],
                "dec": r[2],
                "internal_name": r[3],
                "margin_cache": r[4],
                "compared_to_path": r[5],
                "prepared_path": os.path.join(temp_dir, f"prepared_{r[3]}"),
            }
            for r in results
        ]
        
        # === Log completed steps
        for entry in catalogs:
            tag = f"prepare_{entry['internal_name']}"
            if tag not in completed:
                log_step(log_file, tag)
               
        # === Begin combination logic ===
        final_base_path = os.path.join(output_root_dir_and_output_dir, output_name)
        
        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: prepare_catalogs id=prepare_catalogs")

        def load_compared_to(path):
            with open(path) as f:
                data = json.load(f)
            return {k: set(v) for k, v in data.items()}

        def cleanup_previous_step(step, temp_dir, prepared_catalogs_info, logger):
            """
            Delete temporary files from the previous step if they exist.
        
            Args:
                step (int): Current step number in crossmatch (i).
                temp_dir (str): Path to the temporary working directory.
                prepared_catalogs_info (list of dict): Metadata about each prepared catalog.
                logger (Logger): Logger instance.
            """
            prev_step = step - 1
        
            # === Delete merged_step{prev_step} parquet
            prev_merged = os.path.join(temp_dir, f"merged_step{prev_step}")
            if os.path.exists(prev_merged):
                shutil.rmtree(prev_merged)
                logger.info(f"🗑️ Deleted merged parquet folder: {prev_merged}")
        
            # === Delete merged_step{prev_step}_hats
            prev_merged_hats = os.path.join(temp_dir, f"merged_step{prev_step}_hats")
            if os.path.exists(prev_merged_hats):
                shutil.rmtree(prev_merged_hats)
                logger.info(f"🗑️ Deleted merged HATS folder: {prev_merged_hats}")
        
            # === Delete xmatch_step{prev_step} folder (if it exists)
            prev_xmatch_step = os.path.join(temp_dir, f"xmatch_step{prev_step}")
            if os.path.exists(prev_xmatch_step):
                shutil.rmtree(prev_xmatch_step)
                logger.info(f"🗑️ Deleted xmatch step folder: {prev_xmatch_step}")
        
            # === Delete compared_to_xmatch_step{prev_step}.json
            prev_compared_to_json = os.path.join(temp_dir, f"compared_to_xmatch_step{prev_step}.json")
            if os.path.exists(prev_compared_to_json):
                os.remove(prev_compared_to_json)
                logger.info(f"🗑️ Deleted compared_to JSON: {prev_compared_to_json}")
        
            # === Delete artifacts of catalog at index `prev_step`
            if prev_step < len(prepared_catalogs_info):
                prev_info = prepared_catalogs_info[prev_step]
                for path, label in [
                    (prev_info.get("prepared_path"), "prepared_path"),
                    (prev_info.get("hats_path"), "hats_path"),
                    (prev_info.get("margin_cache"), "margin_cache"),
                    (prev_info.get("compared_to_path"), "compared_to_path")
                ]:
                    if path and os.path.exists(path):
                        try:
                            if os.path.isdir(path):
                                shutil.rmtree(path)
                            else:
                                os.remove(path)
                            logger.info(f"🧹 Deleted previous {label}: {path}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not delete {label} at {path}: {e}")

        if combine_mode == "concatenate":
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: consolidate id=consolidate")
            logger.info("🔗 Combining catalogs by simple concatenation (concatenate mode)")
        
            dfs = [dd.read_parquet(info["prepared_path"]) for info in prepared_catalogs_info]
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
        
            # === Load initial catalog (step 0)
            initial_info = prepared_catalogs_info[0]
            initial_hats_path = initial_info["hats_path"]
            initial_margin_path = initial_info["margin_cache"]
        
            if (
                initial_margin_path and
                os.path.exists(initial_margin_path) and
                os.path.exists(os.path.join(initial_margin_path, "properties"))
            ):
                cat_prev = lsdb.read_hats(initial_hats_path, margin_cache=initial_margin_path)
            else:
                if not os.path.exists(initial_margin_path):
                    logger.warning(f"⚠️ Margin cache folder not found: {initial_margin_path}")
                else:
                    logger.warning(f"⚠️ Margin cache exists but has no 'properties' file: {initial_margin_path}")
                cat_prev = lsdb.read_hats(initial_hats_path)
        
            # === Resume logic: check if any merged step was already completed
            for j in reversed(range(1, len(prepared_catalogs_info))):
                merged_path = os.path.join(temp_dir, f"merged_step{j}_hats")
                if f"crossmatch_step{j}" in completed and os.path.exists(merged_path):
                    cat_prev = lsdb.read_hats(merged_path)
                    logger.info(f"🔁 Resuming from previous merged result: {merged_path}")
                    start_i = j + 1
        
                    compared_to_path = os.path.join(temp_dir, f"compared_to_xmatch_step{j}.json")
                    if not os.path.exists(compared_to_path):
                        logger.error(f"❌ Expected compared_to_xmatch_step{j}.json not found.")
                        client.close(); cluster.close(); return
        
                    final_compared_to_dict = load_compared_to(compared_to_path)

                    break
            else:
                start_i = 1
                initial_compared_path = initial_info["compared_to_path"]
                final_compared_to_dict = load_compared_to(initial_compared_path)

            # === Iterative crossmatching
            for i in range(start_i, len(prepared_catalogs_info)):
                xmatch_tag = f"crossmatch_step{i}"
                curr_info = prepared_catalogs_info[i]
                hats_path = curr_info["hats_path"]
                margin_path = curr_info["margin_cache"]
                internal_name = curr_info["internal_name"]
                compared_path = curr_info["compared_to_path"]
                
                if xmatch_tag not in completed:
                    # Load catalog with margin cache if available
                    if margin_path and os.path.exists(margin_path) and os.path.exists(os.path.join(margin_path, "properties")):
                        cat_curr = lsdb.read_hats(hats_path, margin_cache=margin_path)
                    else:
                        logger.warning(f"⚠️ No valid margin cache found for {internal_name}. Proceeding without it.")
                        cat_curr = lsdb.read_hats(hats_path)
            
                    logger.info(f"🔄 Crossmatching previous result with: {internal_name}")
                    is_last = (i == len(prepared_catalogs_info) - 1)
            
                    # Perform crossmatch with tie-breaking logic
                    cat_prev, compared_to_path = crossmatch_tiebreak_safe(
                        left_cat=cat_prev,
                        right_cat=cat_curr,
                        tiebreaking_priority=tiebreaking_priority,
                        logs_dir=logs_dir,
                        temp_dir=temp_dir,
                        step=i,
                        client=client,
                        compared_to_left=final_compared_to_dict,
                        compared_to_right=load_compared_to(compared_path),
                        instrument_type_priority=instrument_type_priority,
                        translation_config=translation_config,
                        do_import=not is_last
                    )
            
                    # Update compared_to dict
                    final_compared_to_dict = load_compared_to(compared_to_path)
                    log_step(log_file, xmatch_tag)
            
                    # Clean temporary files from previous step
                    if delete_temp_files:
                        cleanup_previous_step(i, temp_dir, prepared_catalogs_info, logger)
                else:
                    logger.info(f"⏩ Skipping already completed step: {xmatch_tag}")
            
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: crossmatch_catalogs id=crossmatch_catalogs")
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: consolidate id=consolidate")
            
            final_merged = os.path.join(temp_dir, f"merged_step{len(prepared_catalogs_info)-1}")
            if not os.path.exists(final_merged):
                logger.error(f"❌ Final merged Parquet folder not found: {final_merged}")
                client.close(); cluster.close(); return
            
            # === Load final compared_to dict
            final_step = len(prepared_catalogs_info) - 1
            final_compared_to_path = os.path.join(temp_dir, f"compared_to_xmatch_step{final_step}.json")
            if not os.path.exists(final_compared_to_path):
                logger.error(f"❌ Final compared_to file not found: {final_compared_to_path}")
                client.close(); cluster.close(); return
            
            final_compared_to_dict = load_compared_to(final_compared_to_path)
            
            # Build RHS DataFrame (CRD_ID -> compared_to)
            compared_to_series = pd.Series({
                k: ", ".join(sorted(v)) if v else ""
                for k, v in final_compared_to_dict.items()
            }).sort_index()
            
            compared_to_df = (
                compared_to_series.rename("compared_to")
                .reset_index()
                .rename(columns={"index": "CRD_ID"})
            )
            
            # Arrow-backed strings
            compared_to_df["CRD_ID"] = compared_to_df["CRD_ID"].astype(DTYPE_STR)
            compared_to_df["compared_to"] = compared_to_df["compared_to"].astype(DTYPE_STR)
            
            # To Dask
            rhs = dd.from_pandas(compared_to_df, npartitions=1, sort=False)

            # Load merged parquet (LHS)
            df_final = dd.read_parquet(final_merged)
            
            # === Option A: cast both sides to object dtype for merge ===
            try:
                df_final = df_final.assign(CRD_ID=df_final["CRD_ID"].astype(DTYPE_STR))
                rhs = rhs.assign(CRD_ID=rhs["CRD_ID"].astype(DTYPE_STR))
                df_final = dd.merge(df_final, rhs, how="left", on="CRD_ID")
            except Exception:
                # Fallback for older Dask/Pandas edge cases.
                df_final = df_final.assign(CRD_ID=df_final["CRD_ID"].astype("object"))
                rhs = rhs.astype({"CRD_ID": "object"})
                df_final = dd.merge(df_final, rhs, how="left", on="CRD_ID")
            
            df_final = df_final.compute()

            if USE_ARROW_TYPES:
                df_final = df_final.convert_dtypes(dtype_backend="pyarrow")

                for c in df_final.columns:
                    if pd.api.types.is_string_dtype(df_final[c].dtype):
                        df_final[c] = df_final[c].astype(DTYPE_STR)

            df_final["CRD_ID"] = df_final["CRD_ID"].astype(DTYPE_STR)
            
            if "compared_to" in df_final.columns:
                df_final["compared_to"] = df_final["compared_to"].astype(DTYPE_STR)
            
            # Remove duplicates if configured
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