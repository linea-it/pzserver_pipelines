import argparse
import os
import time
import json
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

def main(config_path, cwd="."):
    logger = setup_logger("combine_redshift_dedup", logdir=cwd)
    logger.info(f"Loading config from {config_path}")

    config = load_yml(config_path)
    path_to_translation_file = config.get("flags_translation_file")
    if path_to_translation_file is None:
        logger.error("Missing 'flags_translation_file' in config!")
        return
    translation_config = load_yml(path_to_translation_file)

    base_dir = config["base_dir"]
    output_dir = os.path.join(base_dir, config["output_dir"])
    logs_dir = os.path.join(base_dir, config["logs_dir"])
    temp_dir = os.path.join(base_dir, config["temp_dir"])
    log_file = os.path.join(logs_dir, "process_resume.log")
    compared_to_path = os.path.join(temp_dir, "compared_to.json")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    catalogs = config["inputs"]["specz"]
    tiebreaking_priority = config["tiebreaking_priority"]
    type_priority = {"s": 3, "g": 2, "p": 1}

    completed = read_completed_steps(log_file)

    cluster = get_executor(config["executor"])
    client = Client(cluster)

    # Load or initialize compared_to_dict
    if os.path.exists(compared_to_path):
        with open(compared_to_path, "r") as f:
            compared_to_dict = json.load(f)
        compared_to_dict = defaultdict(list, compared_to_dict)
    else:
        compared_to_dict = defaultdict(list)

    prepared_paths = []

    logger.info(f"Preparing {len(catalogs)} catalogs")

    for entry in catalogs:
        tag = f"prepare_{entry['internal_name']}"
        if tag not in completed:
            path_info = prepare_catalog(entry, translation_config, temp_dir)
            log_step(log_file, tag)
        else:
            path_info = (
                os.path.join(temp_dir, f"prepared_{entry['internal_name']}"),
                "ra",
                "dec",
                entry["internal_name"]
            )
        prepared_paths.append(path_info)

    import_tag = f"import_{prepared_paths[0][3]}"
    if import_tag not in completed:
        import_catalog(prepared_paths[0][0], "ra", "dec", "cat0_hats", temp_dir, client)
        log_step(log_file, import_tag)

    cat_prev = lsdb.read_hats(os.path.join(temp_dir, "cat0_hats"))

    for i in range(1, len(prepared_paths)):
        import_tag = f"import_{prepared_paths[i][3]}"
        if import_tag not in completed:
            import_catalog(prepared_paths[i][0], "ra", "dec", f"cat{i}_hats", temp_dir, client)
            log_step(log_file, import_tag)

        margin_tag = f"margin_cache_{prepared_paths[i][3]}"
        if margin_tag not in completed:
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
            logger.warning(f"No margin cache found for cat{i}_hats. Proceeding without margin cache.")
            cat_curr = lsdb.read_hats(os.path.join(temp_dir, f"cat{i}_hats"))

        xmatch_tag = f"crossmatch_step{i}"
        if xmatch_tag not in completed:
            cat_prev = crossmatch_tiebreak_safe(
                cat_prev, cat_curr, tiebreaking_priority, temp_dir, i, client, compared_to_dict, type_priority
            )
            log_step(log_file, xmatch_tag)
        else:
            cat_prev = lsdb.read_hats(os.path.join(temp_dir, f"merged_step{i}_hats"))

    # Final output
    final_base_path = os.path.join(output_dir, config['output_name'])
    output_format = config.get('output_format', 'parquet').lower()

    # Load latest compared_to_dict
    if os.path.exists(compared_to_path):
        with open(compared_to_path, "r") as f:
            compared_to_dict = json.load(f)
    else:
        compared_to_dict = {}

    df_final = dd.read_parquet(os.path.join(temp_dir, f"merged_step{len(prepared_paths)-1}")).compute()
    df_final["compared_to"] = df_final["id"].map(lambda x: ",".join(compared_to_dict.get(str(x), [])))

    save_dataframe(df_final, final_base_path, output_format)
    logger.info(f"✅ Final catalog saved at {final_base_path}.{output_format}")

    client.close()
    cluster.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine redshift catalogs with deduplication.")
    parser.add_argument("config_path", help="Path to YAML config file")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory (default: current dir)")
    args = parser.parse_args()

    start_time = time.time()
    main(args.config_path, args.cwd)
    duration = time.time() - start_time
    print(f"✅ Pipeline completed in {duration:.2f} seconds.")