# -*- coding: utf-8 -*-
"""
crd-run.py — Orchestrates the CRC pipeline (prepare → auto-crossmatch → crossmatch → deduplicate → export)

Pipeline semantics required by the user:

1) PREPARE
   - Writes Parquet at: prepared_<internal_name>
   - Builds Collection at: prepared_<internal_name>_hats
   - Returns: prepared_<internal_name>_hats

2) AUTO SELF-CROSSMATCH
   - Input Collection: prepared_<internal_name>_hats
   - Writes new Collection: prepared_<internal_name>_hats_auto
   - Returns: prepared_<internal_name>_hats_auto

3) CROSSMATCH (iterative)
   - Must use ONLY prepared_<internal_name>_hats_auto

4) DEDUP + EXPORT
"""

# =====================
# Built-in modules
# =====================
import argparse
import os
import re
import time
import shutil
import warnings
from datetime import datetime

# =====================
# Third-party libraries
# =====================
import dask
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, as_completed, performance_report, wait, get_client
import lsdb

# =====================
# Project-specific libraries
# =====================
from combine_redshift_dedup.packages.crossmatch_cross import crossmatch_tiebreak_safe
from combine_redshift_dedup.packages.deduplication import deduplicate_pandas
from combine_redshift_dedup.packages.executor import get_executor
from combine_redshift_dedup.packages.product_handle import save_dataframe
from combine_redshift_dedup.packages.specz import (
    prepare_catalog,
    USE_ARROW_TYPES,
    DTYPE_STR,
)
from combine_redshift_dedup.packages.crossmatch_auto import crossmatch_auto
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
    update_process_info,
)

# ---------------------------------------------------------------------------
# Worker task for auto self-crossmatch
# ---------------------------------------------------------------------------
def _auto_cross_worker(info: dict, logs_dir: str, translation_config: dict):
    """
    Start from prepared_*_hats → self-xmatch → write prepared_*_hats_auto.
    We DO NOT start from *_hats_auto here to avoid chaining *_auto_auto.
    """
    import os
    import lsdb
    from combine_redshift_dedup.packages.crossmatch_auto import crossmatch_auto

    hats_path = info["prepared_path"] + "_hats"  # must exist from prepare step
    if not os.path.isdir(hats_path):
        raise FileNotFoundError(f"Expected prepared collection not found: {hats_path}")

    cat = lsdb.open_catalog(hats_path)
    out_auto = crossmatch_auto(
        catalog=cat,
        collection_path=hats_path,            # base; function writes "<base>_auto"
        logs_dir=logs_dir,
        translation_config=translation_config,
    )
    return out_auto  # prepared_*_hats_auto

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def main(config_path: str, cwd: str = ".", base_dir_override: str | None = None) -> None:
    delete_temp_files = True  # set True to aggressively clean intermediates

    # --- Load config ---
    config = load_yml(config_path)
    param_config = config.get("param", {})
    if base_dir_override is None:
        raise ValueError("❌ You must specify --base_dir via the command line.")
    base_dir = base_dir_override

    output_root_dir = config["output_root_dir"]
    output_dir = config["output_dir"]
    output_root_dir_and_output_dir = os.path.join(output_root_dir, output_dir)
    output_name = config["output_name"]
    output_format = config.get("output_format", "parquet").lower()

    logs_dir = os.path.join(base_dir, "process_info")
    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(output_root_dir_and_output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # --- process.yml bookkeeping ---
    process_info_path = os.path.join(base_dir, "process.yml")
    if not os.path.exists(process_info_path):
        dump_yml(process_info_path, {})
    process_info = load_yml(process_info_path) or {}
    update_process_info(process_info, process_info_path, "status", "Failed")
    update_process_info(process_info, process_info_path, "start_time", str(pd.Timestamp.now()))

    # --- Logger & warnings ---
    logger = setup_logger("combine_redshift_dedup", logdir=logs_dir)
    set_global_logger(logger)
    logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: pipeline_init id=pipeline_init")
    logger.info(f"📄 Loading config from {config_path}")
    configure_warning_handler(logger)
    warnings.filterwarnings("ignore", message=".*Sending large graph of size.*", category=UserWarning, module="distributed")
    configure_exception_hook(logger, process_info, process_info_path)

    # --- Translation file ---
    path_to_translation_file = param_config.get("flags_translation_file")
    if path_to_translation_file is None:
        logger.error("❌ Missing 'flags_translation_file' in config!")
        return
    translation_config = load_yml(path_to_translation_file)

    # --- Inputs sorted by size ---
    catalogs_unsorted = config["inputs"]["specz"]

    def _filesize_mb(p: str) -> float:
        try:
            return os.path.getsize(p) / 1024 / 1024
        except Exception:
            return float("inf")

    catalogs = sorted(catalogs_unsorted, key=lambda e: _filesize_mb(e["path"]))
    logger.info("📏 Catalogs sorted by disk size:")
    for entry in catalogs:
        logger.info(" - %s: %.1f MB", entry["internal_name"], _filesize_mb(entry["path"]))

    combine_mode = param_config.get("combine_type", "concatenate_and_mark_duplicates").lower()
    completed = read_completed_steps(os.path.join(temp_dir, "process_resume.log"))

    # --- Dask cluster/client ---
    cluster = get_executor(config["executor"], logs_dir=logs_dir)
    client = Client(cluster)

    global_report_path = os.path.join(logs_dir, "main_dask_report.html")
    with performance_report(filename=global_report_path):
        logger.info("🧰 Preparing %d input catalogs", len(catalogs))

        # ---------------------------------------------------------------
        # 1) PREPARE (bounded concurrency)
        # ---------------------------------------------------------------
        max_inflight = 5

        def _prebuilt_result_tuple(entry: dict) -> tuple[str, str, str, str, str]:
            """Locate the prepared *collection* for an already prepared catalog.
            We expect 'prepared_<name>_hats'. Auto-cross will build '<...>_hats_auto' later."""
            base = os.path.join(temp_dir, f"prepared_{entry['internal_name']}")
            hats = f"{base}_hats"
            if os.path.isdir(hats):
                return (hats, "ra", "dec", entry["internal_name"], "")
            # If only hats_auto exists (resumed after auto), we still accept it here.
            hats_auto = f"{hats}_auto"
            if os.path.isdir(hats_auto):
                return (hats_auto, "ra", "dec", entry["internal_name"], "")
            logger.warning("⚠️ Could not find prepared collection for %s. Expected %s or %s",
                           entry["internal_name"], hats, hats_auto)
            # Return expected hats path anyway so that failures are explicit later
            return (hats, "ra", "dec", entry["internal_name"], "")

        def _submit_prepare(entry: dict):
            return client.submit(
                prepare_catalog,
                entry,
                translation_config,
                logs_dir,
                temp_dir,
                combine_mode,
                pure=False,
            )

        queue: list[dict] = []
        results: list[tuple[str, str, str, str, str]] = []
        for entry in catalogs:
            tag = f"prepare_{entry['internal_name']}"
            if tag in completed:
                logger.info("⏩ Catalog %s already prepared. Skipping heavy work.", entry["internal_name"])
                results.append(_prebuilt_result_tuple(entry))
            else:
                queue.append(entry)

        logger.info("🧵 Concurrency for prepare: max_inflight=%d, to_prepare=%d, already_prepared=%d",
                    max_inflight, len(queue), len(results))

        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: pipeline_init id=pipeline_init")
        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: prepare_catalogs id=prepare_catalogs")

        ac = as_completed()
        inflight = []
        for _ in range(min(max_inflight, len(queue))):
            fut = _submit_prepare(queue.pop(0))
            ac.add(fut); inflight.append(fut)

        prepared_count = 0
        while inflight:
            fut = next(ac); inflight.remove(fut)
            results.append(fut.result()); prepared_count += 1
            if queue:
                fut2 = _submit_prepare(queue.pop(0))
                ac.add(fut2); inflight.append(fut2)

        logger.info("✅ Prepared %d new catalogs; total results now %d / %d", prepared_count, len(results), len(catalogs))
        if len(results) != len(catalogs):
            raise RuntimeError("Internal error: prepared results count mismatch.")

        # Normalize results → metadata per item
        prepared_catalogs_info = [
            {
                "collection_path": r[0],                                      # prepared_<name>_hats (or hats_auto if resumed)
                "prepared_path": os.path.join(temp_dir, f"prepared_{r[3]}"),  # Parquet folder
                "ra": r[1],
                "dec": r[2],
                "internal_name": r[3],
            } for r in results
        ]

        # Mark prepares as done
        resume_log = os.path.join(temp_dir, "process_resume.log")
        for entry in catalogs:
            tag = f"prepare_{entry['internal_name']}"
            if tag not in completed:
                log_step(resume_log, tag)

        # ---------------------------------------------------------------
        # 2) AUTO SELF-CROSSMATCH (parallel)
        # ---------------------------------------------------------------
        if combine_mode in ("concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"):
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: auto_crossmatch id=auto_crossmatch")

            queue_auto = []
            for info in prepared_catalogs_info:
                tag = f"autocross_{info['internal_name']}"
                if tag in completed:
                    logger.info("⏩ Auto crossmatch already done for %s. Skipping.", info["internal_name"])
                else:
                    queue_auto.append(info)

            max_inflight_auto = int(param_config.get("auto_cross_max_inflight", 5))
            logger.info("🧵 Concurrency for auto-cross: max_inflight=%d, to_run=%d, already_done=%d",
                        max_inflight_auto, len(queue_auto), len(prepared_catalogs_info) - len(queue_auto))

            ac2 = as_completed(); inflight2 = []; fut2info = {}

            # Seed
            for _ in range(min(max_inflight_auto, len(queue_auto))):
                info = queue_auto.pop(0)
                fut = client.submit(
                    _auto_cross_worker,
                    info,
                    logs_dir,
                    translation_config,
                    pure=False,
                )
                ac2.add(fut); inflight2.append(fut); fut2info[fut] = info

            auto_done = 0
            while inflight2:
                fut = next(ac2); inflight2.remove(fut)
                try:
                    ret = fut.result()
                except Exception as e:
                    info_err = fut2info.pop(fut)
                    logger.error("❌ Auto crossmatch failed for %s: %s", info_err["internal_name"], e)
                    raise

                info_ok = fut2info.pop(fut)
                info_ok["collection_path"] = ret  # prepared_*_hats_auto

                log_step(resume_log, f"autocross_{info_ok['internal_name']}")
                auto_done += 1

                if queue_auto:
                    info = queue_auto.pop(0)
                    fut2 = client.submit(
                        _auto_cross_worker,
                        info,
                        logs_dir,
                        translation_config,
                        pure=False,
                    )
                    ac2.add(fut2); inflight2.append(fut2); fut2info[fut2] = info

            logger.info("✅ Auto crossmatch completed for %d catalogs", auto_done)
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: auto_crossmatch id=auto_crossmatch")

        # After auto-cross (including skipped via resume), enforce that everyone points to *_hats_auto
        for info in prepared_catalogs_info:
            hats_auto = info["prepared_path"] + "_hats_auto"
            if os.path.isdir(hats_auto):
                info["collection_path"] = hats_auto
            else:
                raise FileNotFoundError(
                    f"Auto-cross output not found for {info['internal_name']}: {hats_auto}. "
                    "Auto-cross must succeed before crossmatch."
                )

        # ---------------------------------------------------------------
        # 3) CROSSMATCH (no tie-breaking)
        # ---------------------------------------------------------------
        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: prepare_catalogs id=prepare_catalogs")
        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: crossmatch_catalogs id=crossmatch_catalogs")
        logger.info("🔍 Combining catalogs with crossmatch (no tie-breaking) and downstream deduplication")

        def _cleanup_previous_step(step_index: int) -> None:
            """Remove previous intermediate artifacts to save space."""
            prev_step = step_index - 1
            prev_merged = os.path.join(temp_dir, f"merged_step{prev_step}")
            if os.path.exists(prev_merged):
                shutil.rmtree(prev_merged, ignore_errors=True)
                logger.info("🗑️ Deleted merged parquet: %s", prev_merged)
            if 0 <= prev_step < len(prepared_catalogs_info):
                prev_info = prepared_catalogs_info[prev_step]
                core = [prev_info.get("prepared_path"), prev_info.get("collection_path")]
                for path in core:
                    if path and os.path.exists(path):
                        try:
                            if os.path.isdir(path):
                                shutil.rmtree(path, ignore_errors=True)
                            else:
                                os.remove(path)
                            logger.info("🧹 Deleted previous artifact: %s", path)
                        except Exception as e:
                            logger.warning("⚠️ Could not delete %s: %s", path, e)

        if combine_mode == "concatenate":
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: consolidate id=consolidate")
            logger.info("🔗 Combining catalogs by simple concatenation (concatenate mode)")
            dfs = [dd.read_parquet(info["prepared_path"]) for info in prepared_catalogs_info]
            df_final = dd.concat(dfs).compute()

        elif combine_mode in ("concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"):
            init = prepared_catalogs_info[0]
            cat_prev = lsdb.open_catalog(init["collection_path"])  # guaranteed *_hats_auto

            start_i = 1
            for i in range(start_i, len(prepared_catalogs_info)):
                tag = f"crossmatch_step{i}"
                info_i = prepared_catalogs_info[i]
                if tag in completed:
                    logger.info("⏩ Skipping already completed step: %s", tag)
                    continue

                cat_curr = lsdb.open_catalog(info_i["collection_path"])  # guaranteed *_hats_auto
                target_name = info_i["internal_name"]

                logger.info("🔄 Crossmatching previous result with: %s", target_name)

                is_last = (i == len(prepared_catalogs_info) - 1)
                crossmatch_result = crossmatch_tiebreak_safe(
                    left_cat=cat_prev,
                    right_cat=cat_curr,
                    logs_dir=logs_dir,
                    temp_dir=temp_dir,
                    step=i,
                    client=client,                    # IMPORTANT: needed by importer in crossmatch module
                    translation_config=translation_config,
                    do_import=not is_last,            # last step returns parquet
                )

                if is_last:
                    final_merged_path = crossmatch_result   # parquet path
                else:
                    cat_prev = lsdb.open_catalog(crossmatch_result)  # collection path from importer

                log_step(resume_log, tag)
                if delete_temp_files:
                    _cleanup_previous_step(i)

            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: crossmatch_catalogs id=crossmatch_catalogs")
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: deduplicate id=deduplicate")
            
            # Ensure final merged parquet exists
            final_step = len(prepared_catalogs_info) - 1
            final_merged = os.path.join(temp_dir, f"merged_step{final_step}")
            n_before_dedup = int(dd.read_parquet(final_merged).shape[0].compute())
            logger.info("📏 Rows before dedup (final_merged): %d", n_before_dedup)
            if not os.path.exists(final_merged):
                logger.error("❌ Final merged Parquet folder not found: %s", final_merged)
                client.close(); cluster.close()
                return
            
            # Dedup
            logger.info("🧮 Running graph-based deduplication on final merged catalog")
            
            # --- Load config safely (no variable shadowing) ---
            tiebreaking_priority_cfg = translation_config.get("tiebreaking_priority")
            if isinstance(tiebreaking_priority_cfg, (str, bytes)):
                tiebreaking_priority_cfg = [str(tiebreaking_priority_cfg)]
            if not isinstance(tiebreaking_priority_cfg, (list, tuple)) or not tiebreaking_priority_cfg:
                raise TypeError("tiebreaking_priority must be a non-empty list of column names.")
            
            instrument_type_priority_cfg = translation_config.get("instrument_type_priority")
            # Only required if instrument_type is used in priority:
            if "instrument_type_homogenized" in set(tiebreaking_priority_cfg) and not isinstance(instrument_type_priority_cfg, dict):
                raise TypeError(
                    "instrument_type_priority must be a mapping/dict when "
                    "'instrument_type_homogenized' is used in tiebreaking_priority."
                )
            
            delta_z_threshold_cfg = translation_config.get("delta_z_threshold", 0.0)
            delta_z_threshold_cfg = float(delta_z_threshold_cfg)
            
            # 1) Open the dataset to inspect its schema (no data read yet)
            base = dd.read_parquet(final_merged, engine="pyarrow", split_row_groups=True)
            available = set(map(str, base.columns))
            
            # 2) Core required columns (independent of tiebreaking)
            required_base = {"CRD_ID", "compared_to", "z"}
            missing_base = sorted(required_base - available)
            if missing_base:
                raise KeyError(
                    f"The following required columns are missing from the parquet: {missing_base}"
                )
            
            # 3) Tiebreaking columns: ALL must exist
            priority_set = set(tiebreaking_priority_cfg or [])
            missing_priority = sorted([c for c in priority_set if c not in available])
            if missing_priority:
                raise KeyError(
                    "Columns listed in 'tiebreaking_priority' do not exist in the parquet: "
                    f"{missing_priority}. Fix your config or generate these columns before dedup."
                )
            
            # 4) Optional columns: only optional if NOT in tiebreaking_priority
            optional_candidates = {"z_flag_homogenized", "instrument_type_homogenized"}
            # If included in priority_set, they were already validated above and are required
            optional_present = (optional_candidates - priority_set) & available
            
            # 5) 'tie_result' is optional on read (it may not exist yet)
            maybe_tie_result = {"tie_result"} & available
            
            # 6) Final list of columns to read (deduped)
            needed = sorted(required_base | priority_set | optional_present | maybe_tie_result)
            
            # 7) Read only what's needed and materialize as pandas
            pdf = base[needed].compute()
            
            # 8) Run driver-side pandas dedup (fast)
            pdf_out = deduplicate_pandas(
                pdf,
                tiebreaking_priority=tiebreaking_priority_cfg,
                instrument_type_priority=instrument_type_priority_cfg if isinstance(instrument_type_priority_cfg, dict) else None,
                delta_z_threshold=delta_z_threshold_cfg,
            )
            
            # 9) Bring back to Dask and update only rows that belong to any component
            df_all = dd.read_parquet(final_merged)  # all columns
            
            # Robust neighbor parsing (works in pandas before we go back to Dask)
            def _parse_compared_to_cell(val):
                if pd.isna(val):
                    return []
                s = str(val).strip()
                if not s:
                    return []
                return [t for t in (tok.strip() for tok in s.split(",")) if t]
            
            cmp_lists = pdf_out["compared_to"].apply(_parse_compared_to_cell)
            
            # Who declared neighbors?
            declared_mask = cmp_lists.map(lambda L: len(L) > 0)
            ids_with_neighbors = set(pdf_out.loc[declared_mask, "CRD_ID"].astype(str))
            
            # Neighbors cited by someone
            neighbor_ids = set(nb for lst in cmp_lists[declared_mask] for nb in lst)
            
            # All ids that belong to any component (declared or cited)
            to_update_ids = ids_with_neighbors | neighbor_ids
            
            # RHS to merge back (only component members)
            rhs_pdf = (
                pdf_out.loc[pdf_out["CRD_ID"].astype(str).isin(to_update_ids), ["CRD_ID", "tie_result"]]
                       .rename(columns={"tie_result": "tie_result_new"})
            )
            rhs_dd = dd.from_pandas(rhs_pdf, npartitions=max(1, df_all.npartitions))
            
            merged = df_all.merge(rhs_dd, on="CRD_ID", how="left")
            
            # Coalesce: prefer new values where present; otherwise keep original
            if "tie_result" in merged.columns:
                merged["tie_result"] = merged["tie_result_new"].fillna(merged["tie_result"])
            else:
                merged = merged.assign(tie_result=merged["tie_result_new"])
            merged = merged.drop(columns=["tie_result_new"])
            
            # ================== Consistency & clamps (Dask-safe) ==================
            # Use floats para evitar booleans "nullable" com pd.NA
            tie_f = dd.to_numeric(merged["tie_result"], errors="coerce").astype("float64")          # NaN ok
            zf_f  = dd.to_numeric(merged["z_flag_homogenized"], errors="coerce").astype("float64")  # NaN ok
            
            # 1) Estrela: flag==6 => tie=3
            is_star = zf_f.eq(6.0)                         # bool puro (NaN->False)
            merged["tie_result"] = merged["tie_result"].mask(is_star, 3)
            
            # 2) Proibir tie=3 quando NÃO for estrela
            wrong_3 = tie_f.eq(3.0) & ~is_star             # bool puro
            merged["tie_result"] = merged["tie_result"].mask(wrong_3, 0)
            
            # 3) Guard extra: se é isolado (compared_to vazio) E flag é NaN => restaura tie original
            cmp_str   = merged["compared_to"].astype("string")
            cmp_empty = cmp_str.isna() | (cmp_str.str.strip().str.len().fillna(0) == 0)   # bool puro
            guard_mask = cmp_empty & zf_f.isna()                                          # bool puro
            
            n_restore = guard_mask.astype("int8").sum().compute()
            if int(n_restore) > 0:
                orig_tr = dd.read_parquet(final_merged, columns=["CRD_ID", "tie_result"])\
                           .rename(columns={"tie_result": "tie_result_orig"})
                merged = merged.merge(orig_tr, on="CRD_ID", how="left")
                merged["tie_result"] = merged["tie_result"].mask(guard_mask, merged["tie_result_orig"])
                merged = merged.drop(columns=["tie_result_orig"])
            
            # 4) Sanity logs (somente contagem; sem NA ambíguo)
            invalid_star3 = tie_f.eq(3.0) & ~zf_f.eq(6.0)   # tie==3 mas flag!=6
            n_invalid_star3 = invalid_star3.astype("int64").sum().compute()
            if int(n_invalid_star3) > 0:
                # head(5) já retorna Pandas -> NÃO chame .compute() aqui
                ex = merged.loc[invalid_star3, ["CRD_ID","compared_to","z_flag_homogenized","tie_result"]].head(5)
                #logger.warning(
                #    "Found %d rows with tie_result==3 but flag!=6. Examples:\n%s",
                #    int(n_invalid_star3), ex.to_string(index=False)
                #)
            
            invalid_flag6 = zf_f.eq(6.0) & ~tie_f.eq(3.0)   # flag==6 mas tie!=3
            n_invalid_flag6 = invalid_flag6.astype("int64").sum().compute()
            if int(n_invalid_flag6) > 0:
                # idem: sem .compute()
                ex = merged.loc[invalid_flag6, ["CRD_ID","compared_to","z_flag_homogenized","tie_result"]].head(5)
                #logger.warning(
                #    "Found %d rows with flag==6 but tie_result!=3. Examples:\n%s",
                #    int(n_invalid_flag6), ex.to_string(index=False)
                #)

            # =====================================================================
            
            # Final dtype
            merged["tie_result"] = merged["tie_result"].astype("Int8")
            df_final_dd = merged
            df_final = df_final_dd.compute()
            
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: deduplicate id=deduplicate")
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: consolidate id=consolidate")

        else:
            logger.error("❌ Unknown combine_mode: %s", combine_mode)
            client.close(); cluster.close()
            return

    try:
        n_after_dedup = int(len(df_final))
        logger.info("📏 Rows after dedup (in-memory): %d", n_after_dedup)
    except Exception as e:
        logger.warning("⚠️ Could not compute len(df_final): %s", e)
    # -------------------------------------------------------------------
    # 4) Export + bookkeeping
    # -------------------------------------------------------------------
    if combine_mode == "concatenate" and "tie_result" in df_final.columns:
        logger.info("ℹ️ Dropping column 'tie_result' (not needed for concatenate mode)")
        df_final = df_final.drop(columns=["tie_result"])

    if USE_ARROW_TYPES:
        df_final = df_final.convert_dtypes(dtype_backend="pyarrow")
        for c in df_final.columns:
            if pd.api.types.is_string_dtype(df_final[c].dtype):
                df_final[c] = df_final[c].astype(DTYPE_STR)

    # drop all-missing columns
    to_drop = []
    for col in df_final.columns:
        dt = df_final[col].dtype
        if str(dt) == "string[pyarrow]" or str(dt) == "object":
            all_missing = df_final[col].apply(lambda x: (pd.isna(x) or str(x).strip() == ""))
        else:
            all_missing = df_final[col].isna()
        if bool(all_missing.all()):
            to_drop.append(col)
    if to_drop:
        logger.info("ℹ️ Dropping columns with all-missing values: %s", ", ".join(sorted(map(str, to_drop))))
        df_final = df_final.drop(columns=to_drop)

    final_base_path = os.path.join(output_root_dir_and_output_dir, output_name)
    save_dataframe(df_final, final_base_path, output_format)
    logger.info("✅ Final combined catalog saved at %s.%s", final_base_path, output_format)

    relative_path = os.path.join(output_dir, f"{output_name}.{output_format}")
    expected_columns = ["id", "ra", "dec", "z", "z_flag", "z_err", "survey"]
    columns_assoc = {col: col for col in expected_columns if col in df_final.columns}

    update_process_info(process_info, process_info_path, "outputs", [{
        "path": relative_path,
        "root_dir": output_root_dir,
        "role": "main",
        "columns_assoc": columns_assoc,
    }])
    update_process_info(process_info, process_info_path, "end_time", str(pd.Timestamp.now()))
    update_process_info(process_info, process_info_path, "status", "Successful")

    if delete_temp_files:
        try:
            shutil.rmtree(temp_dir)
            logger.info("🧹 Deleted entire temp_dir %s after successful pipeline completion", temp_dir)
        except Exception as e:
            logger.warning("⚠️ Could not delete temp_dir %s: %s", temp_dir, e)

    logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: consolidate id=consolidate")
    client.close(); cluster.close()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine redshift catalogs via preparation, crossmatch (no tie-breaking), and graph-based deduplication."
    )
    parser.add_argument("config_path", help="Path to YAML config file.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory (default: current dir).")
    parser.add_argument("--base_dir", required=True, help="Base directory for outputs and logs.")
    args = parser.parse_args()

    start_ts = time.time()
    main(args.config_path, args.cwd, args.base_dir)
    dur = time.time() - start_ts
    logger = get_global_logger()
    log_and_print(f"✅ Pipeline completed in {dur:.2f} seconds.", logger)