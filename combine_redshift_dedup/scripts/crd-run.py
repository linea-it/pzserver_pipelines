# -*- coding: utf-8 -*-
"""
Orchestrates the CRC pipeline (prepare → auto-crossmatch → crossmatch → deduplicate → export).
"""

from __future__ import annotations

# =====================
# Built-in
# =====================
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any
from datetime import datetime

# =====================
# Logging
# =====================
import logging
from logging.handlers import RotatingFileHandler  # noqa: F401  (kept for back-compat imports)

# =====================
# Third-party
# =====================
import dask
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, as_completed, performance_report
import lsdb

# =====================
# Project
# =====================
from combine_redshift_dedup.packages.crossmatch_auto import crossmatch_auto
from combine_redshift_dedup.packages.crossmatch_cross import crossmatch_tiebreak_safe
from combine_redshift_dedup.packages.deduplication import (
    deduplicate_pandas,
    run_dedup_with_lsdb_map_partitions,
)
from combine_redshift_dedup.packages.executor import get_executor
from combine_redshift_dedup.packages.product_handle import save_dataframe
from combine_redshift_dedup.packages.specz import (
    prepare_catalog,
    USE_ARROW_TYPES,
    DTYPE_STR,
)
from combine_redshift_dedup.packages.utils import (
    configure_exception_hook,
    configure_warning_handler,
    dump_yml,
    load_yml,
    log_step,
    read_completed_steps,
    update_process_info,
    ensure_crc_logger,
    start_crc_log_collector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase_logger(base_logger: logging.Logger, phase: str) -> logging.LoggerAdapter:
    """Return a LoggerAdapter that injects the phase into records."""
    return logging.LoggerAdapter(base_logger, {"phase": phase})


def _filesize_mb(path: str) -> float:
    """Return file size in MB. On error, inf."""
    try:
        return os.path.getsize(path) / 1024 / 1024
    except Exception:
        return float("inf")


def _is_collection_root(path: str) -> bool:
    """Return True if path contains collection.properties (HATS root)."""
    return bool(path) and os.path.isdir(path) and os.path.exists(os.path.join(path, "collection.properties"))


def _is_hats_subcatalog(path: str) -> bool:
    """Return True if path contains hats.properties (HATS subcatalog)."""
    return bool(path) and os.path.isdir(path) and os.path.exists(os.path.join(path, "hats.properties"))


def _is_hats_collection(path: str) -> bool:
    """Return True if root or subcatalog."""
    return _is_collection_root(path) or _is_hats_subcatalog(path)


def _normalize_collection_root(path: str | None) -> str | None:
    """Return normalized collection root given a root or subcatalog path."""
    if not path:
        return path
    p = path.rstrip("/")
    if _is_collection_root(p):
        return p
    if _is_hats_subcatalog(p):
        parent = os.path.dirname(p)
        return parent if _is_collection_root(parent) else p
    return p


def _guess_collection_for_step(temp_dir: str, step: int) -> str | None:
    """Heuristically discover the imported collection root for a given step."""
    candidates = [
        os.path.join(temp_dir, f"merged_step{step}_hats"),
        os.path.join(temp_dir, f"merged_step{step}.hats"),
    ]
    for cand in candidates:
        if _is_collection_root(cand):
            return cand

    patterns = [
        os.path.join(temp_dir, f"merged_step{step}", "*_hats"),
        os.path.join(temp_dir, f"merged_step{step}", "*.hats"),
    ]
    for pat in patterns:
        for cand in glob.glob(pat):
            cand_root = _normalize_collection_root(cand)
            if cand_root and _is_collection_root(cand_root):
                return cand_root

    for props in glob.glob(os.path.join(temp_dir, "**", "collection.properties"), recursive=True):
        root = os.path.dirname(props)
        if f"step{step}" in root and _is_collection_root(root):
            return root

    return None


def _resume_set(resume_log_path: str, key: str, value: str, lg: logging.LoggerAdapter) -> None:
    """Append a key/value checkpoint entry into the resume log."""
    try:
        data = {}
        if os.path.exists(resume_log_path):
            with open(resume_log_path, "r") as f:
                for line in f:
                    if line.startswith("{"):
                        d = json.loads(line)
                        data.update(d)
        data[key] = value
        with open(resume_log_path, "a") as f:
            f.write(json.dumps({key: value}) + "\n")
        lg.info("Resume checkpoint saved: %s = %s", key, value)
    except Exception as e:
        lg.warning("Could not update resume log: %s", e)


def _resume_get(resume_log_path: str, key: str) -> str | None:
    """Return last value for a key from the resume log."""
    try:
        if not os.path.exists(resume_log_path):
            return None
        val = None
        with open(resume_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                d = json.loads(line)
                if key in d:
                    val = d[key]
        return val
    except Exception:
        return None


def _cleanup_previous_step(step_index: int, prepared_info: list[dict[str, Any]], temp_dir: str, lg: logging.LoggerAdapter) -> None:
    """Delete artifacts from the previous step to save disk space."""
    prev_step = step_index - 1
    prev_merged = os.path.join(temp_dir, f"merged_step{prev_step}")
    if os.path.exists(prev_merged):
        shutil.rmtree(prev_merged, ignore_errors=True)
        lg.info("Deleted merged parquet: %s", prev_merged)
    if 0 <= prev_step < len(prepared_info):
        prev_info = prepared_info[prev_step]
        core = [prev_info.get("prepared_path"), prev_info.get("collection_path")]
        for path in core:
            if path and os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                    lg.info("Deleted previous artifact: %s", path)
                except Exception as e:
                    lg.warning("Could not delete %s: %s", path, e)


# ---------------------------------------------------------------------------
# Worker task for auto self-crossmatch
# ---------------------------------------------------------------------------

def _auto_cross_worker(info: dict, logs_dir: str, translation_config: dict):
    """Run self-crossmatch from prepared_*_hats and write prepared_*_hats_auto."""
    ensure_crc_logger(logs_dir)
    hats_path = info["prepared_path"] + "_hats"
    if not os.path.isdir(hats_path):
        raise FileNotFoundError(f"Expected prepared collection not found: {hats_path}")
    cat = lsdb.open_catalog(hats_path)
    out_auto = crossmatch_auto(
        catalog=cat,
        collection_path=hats_path,  # base; writes "<base>_auto"
        logs_dir=logs_dir,
        translation_config=translation_config,
    )
    return out_auto


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def main(config_path: str, cwd: str = ".", base_dir_override: str | None = None) -> None:
    """Run the CRC pipeline end-to-end."""
    delete_temp_files = True  # set True to aggressively clean intermediates

    # --- Load config ---
    config = load_yml(config_path)
    param_config = config.get("param", {})
    if base_dir_override is None:
        raise ValueError("You must specify --base_dir via the command line.")
    base_dir = base_dir_override

    output_root_dir = config["output_root_dir"]
    output_dir = config["output_dir"]
    out_root_and_dir = os.path.join(output_root_dir, output_dir)
    output_name = config["output_name"]
    output_format = config.get("output_format", "parquet").lower()

    logs_dir = os.path.join(base_dir, "process_info")
    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(out_root_and_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # --- process.yml bookkeeping ---
    process_info_path = os.path.join(base_dir, "process.yml")
    if not os.path.exists(process_info_path):
        dump_yml(process_info_path, {})
    process_info = load_yml(process_info_path) or {}
    update_process_info(process_info, process_info_path, "status", "Failed")
    update_process_info(process_info, process_info_path, "start_time", str(pd.Timestamp.now()))

    # --- Logger & warnings (driver) ---
    base_logger = ensure_crc_logger(logs_dir)
    log_init = _phase_logger(base_logger, "init")

    # Optional collector on the driver if CRC_LOG_COLLECTOR is set
    collector_env = os.getenv("CRC_LOG_COLLECTOR", "").strip()
    if collector_env:
        try:
            host, port_str = collector_env.split(":")
            port = int(port_str)
            bind_host = "0.0.0.0" if host not in ("127.0.0.1", "localhost") else host
            start_crc_log_collector(host=bind_host, port=port)
            log_init.info("CRC log collector listening on udp://%s:%d", bind_host, port)
        except Exception as e:
            log_init.warning("Failed to start CRC log collector from '%s': %s", collector_env, e)

    log_init.info("START init: pipeline bootstrap")
    configure_warning_handler(base_logger)
    warnings.filterwarnings(
        "ignore",
        message=".*Sending large graph of size.*",
        category=UserWarning,
        module="distributed",
    )
    configure_exception_hook(base_logger, process_info, process_info_path)

    # --- Translation file ---
    path_to_translation_file = param_config.get("flags_translation_file")
    if path_to_translation_file is None:
        log_init.error("Missing 'flags_translation_file' in config!")
        return
    translation_config = load_yml(path_to_translation_file)

    # --- Inputs sorted by size ---
    catalogs_unsorted = config["inputs"]["specz"]
    catalogs = sorted(catalogs_unsorted, key=lambda e: _filesize_mb(e["path"]))
    log_init.info("Catalogs sorted by disk size:")
    for entry in catalogs:
        log_init.info(" - %s: %.1f MB", entry["internal_name"], _filesize_mb(entry["path"]))

    combine_mode = param_config.get("combine_type", "concatenate_and_mark_duplicates").lower()
    completed = read_completed_steps(os.path.join(temp_dir, "process_resume.log"))
    log_init.info("END init: pipeline bootstrap")

    # --- Dask cluster/client ---
    cluster = get_executor(config["executor"], logs_dir=logs_dir)
    client = Client(cluster)

    # Dask perf report (global)
    global_report_path = os.path.join(logs_dir, "main_dask_report.html")
    with performance_report(filename=global_report_path):

        # ---------------------------------------------------------------
        # 1) PREPARATION
        # ---------------------------------------------------------------
        log_prep = _phase_logger(base_logger, "preparation")
        log_prep.info("START preparation: reading inputs and building prepared collections (temp=%s)", temp_dir)

        max_inflight = int(param_config.get("prepare_max_inflight", 5))

        def _prebuilt_result_tuple(entry: dict) -> tuple[str, str, str, str, str]:
            base = os.path.join(temp_dir, f"prepared_{entry['internal_name']}")
            hats = f"{base}_hats"
            if os.path.isdir(hats):
                return (hats, "ra", "dec", entry["internal_name"], "")
            hats_auto = f"{hats}_auto"
            if os.path.isdir(hats_auto):
                return (hats_auto, "ra", "dec", entry["internal_name"], "")
            log_prep.warning(
                "Prepared collection not found for %s. Expected %s or %s",
                entry["internal_name"], hats, hats_auto
            )
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
                log_prep.info("Skip already prepared: %s", entry["internal_name"])
                results.append(_prebuilt_result_tuple(entry))
            else:
                queue.append(entry)

        log_prep.info(
            "Concurrency for prepare: max_inflight=%d, to_prepare=%d, already_prepared=%d",
            max_inflight, len(queue), len(results),
        )

        ac = as_completed()
        inflight = []
        for _ in range(min(max_inflight, len(queue))):
            fut = _submit_prepare(queue.pop(0))
            ac.add(fut)
            inflight.append(fut)

        prepared_count = 0
        while inflight:
            fut = next(ac)
            inflight.remove(fut)
            results.append(fut.result())
            prepared_count += 1
            if queue:
                fut2 = _submit_prepare(queue.pop(0))
                ac.add(fut2)
                inflight.append(fut2)

        log_prep.info(
            "Prepared %d new catalogs; total results now %d / %d",
            prepared_count, len(results), len(catalogs),
        )
        if len(results) != len(catalogs):
            raise RuntimeError("Internal error: prepared results count mismatch.")

        prepared_info = [
            {
                "collection_path": r[0],
                "prepared_path": os.path.join(temp_dir, f"prepared_{r[3]}"),
                "ra": r[1],
                "dec": r[2],
                "internal_name": r[3],
            }
            for r in results
        ]

        # Mark prepares as done
        resume_log = os.path.join(temp_dir, "process_resume.log")
        for entry in catalogs:
            tag = f"prepare_{entry['internal_name']}"
            if tag not in completed:
                log_step(resume_log, tag)

        log_prep.info("END preparation: finished prepared collections")

        # ---------------------------------------------------------------
        # 2) AUTO MATCH (self crossmatch over each prepared)
        # ---------------------------------------------------------------
        log_auto = _phase_logger(base_logger, "automatch")
        if combine_mode in ("concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"):
            log_auto.info("START automatch: generating *_hats_auto from prepared collections")

            queue_auto: list[dict] = []
            for info in prepared_info:
                tag = f"autocross_{info['internal_name']}"
                if tag in completed:
                    log_auto.info("Skip auto for %s (already done)", info["internal_name"])
                else:
                    queue_auto.append(info)

            max_inflight_auto = int(param_config.get("auto_cross_max_inflight", 5))
            log_auto.info(
                "Concurrency for auto-cross: max_inflight=%d, to_run=%d, already_done=%d",
                max_inflight_auto, len(queue_auto), len(prepared_info) - len(queue_auto),
            )

            ac2 = as_completed()
            inflight2 = []
            fut2info: dict[Any, dict] = {}

            for _ in range(min(max_inflight_auto, len(queue_auto))):
                info = queue_auto.pop(0)
                fut = client.submit(_auto_cross_worker, info, logs_dir, translation_config, pure=False)
                ac2.add(fut)
                inflight2.append(fut)
                fut2info[fut] = info

            auto_done = 0
            while inflight2:
                fut = next(ac2)
                inflight2.remove(fut)
                try:
                    ret = fut.result()
                except Exception as e:
                    info_err = fut2info.pop(fut)
                    log_auto.error("Auto crossmatch failed for %s: %s", info_err["internal_name"], e)
                    raise
                info_ok = fut2info.pop(fut)
                info_ok["collection_path"] = ret  # prepared_*_hats_auto
                log_step(resume_log, f"autocross_{info_ok['internal_name']}")
                auto_done += 1
                if queue_auto:
                    info = queue_auto.pop(0)
                    fut2 = client.submit(_auto_cross_worker, info, logs_dir, translation_config, pure=False)
                    ac2.add(fut2)
                    inflight2.append(fut2)
                    fut2info[fut2] = info

            log_auto.info("Auto crossmatch completed for %d catalogs", auto_done)
            log_auto.info("END automatch: all *_hats_auto generated")

        # Enforce *_hats_auto for all
        for info in prepared_info:
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
        log_cross = _phase_logger(base_logger, "crossmatch")
        log_cross.info("START crossmatch: iterative merges over prepared *_hats_auto")

        if combine_mode == "concatenate":
            # Simple concat mode still passes through consolidation phase later.
            log_cross.info("Concatenate mode selected (no crossmatch).")
            df_final = dd.concat([dd.read_parquet(i["prepared_path"]) for i in prepared_info]).compute()
            log_cross.info("END crossmatch: concatenate mode (no crossmatch performed)")

            # Consolidation will run below
            start_consolidate = True

        elif combine_mode in ("concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"):
            init = prepared_info[0]
            cat_prev = lsdb.open_catalog(init["collection_path"])  # *_hats_auto
            start_i = 1
            final_collection_path = None
            final_step = len(prepared_info) - 1

            for i in range(start_i, len(prepared_info)):
                tag = f"crossmatch_step{i}"
                info_i = prepared_info[i]

                if tag in completed:
                    log_cross.info("Skip completed step: %s", tag)
                    resume_key = f"{tag}.collection_path"
                    resumed_col = _resume_get(resume_log, resume_key)
                    resumed_col = _normalize_collection_root(resumed_col)
                    if resumed_col and _is_collection_root(resumed_col):
                        log_cross.info("Resume collection root for %s: %s", tag, resumed_col)
                        cat_prev = lsdb.open_catalog(resumed_col)
                        if i == final_step:
                            final_collection_path = resumed_col
                    elif i == final_step and final_collection_path is None:
                        guessed = _guess_collection_for_step(temp_dir, i)
                        if guessed:
                            log_cross.info("Guessed collection root for %s: %s", tag, guessed)
                            final_collection_path = guessed
                    continue

                cat_curr = lsdb.open_catalog(info_i["collection_path"])  # *_hats_auto
                target_name = info_i["internal_name"]
                log_cross.info("Crossmatching previous result with: %s", target_name)

                is_last = (i == final_step)
                crossmatch_result = crossmatch_tiebreak_safe(
                    left_cat=cat_prev,
                    right_cat=cat_curr,
                    logs_dir=logs_dir,
                    temp_dir=temp_dir,
                    step=i,
                    client=client,
                    translation_config=translation_config,
                    do_import=True,  # returns collection path (root or subcat)
                )

                log_cross.info("Raw crossmatch path returned: %s", crossmatch_result)
                norm = _normalize_collection_root(crossmatch_result)
                if norm != crossmatch_result:
                    log_cross.info("Normalized to collection root: %s", norm)

                if not _is_hats_collection(norm):
                    log_cross.warning("Returned path does not look like a HATS collection: %s", crossmatch_result)

                cat_prev = lsdb.open_catalog(norm)
                _resume_set(resume_log, f"{tag}.collection_path", norm, log_cross)
                if is_last:
                    final_collection_path = norm

                log_step(resume_log, tag)
                if delete_temp_files:
                    _cleanup_previous_step(i, prepared_info, temp_dir, log_cross)

            log_cross.info("END crossmatch: graph merge completed")
            start_consolidate = False  # consolidation runs after dedup

            # -----------------------------------------------------------
            # 4) DEDUPLICATION
            # -----------------------------------------------------------
            log_dedup = _phase_logger(base_logger, "deduplication")
            log_dedup.info("START deduplication: LSDB graph labeling and tie consolidation")

            # Ensure final collection root for dedup
            if not final_collection_path:
                resumed = _resume_get(resume_log, f"crossmatch_step{final_step}.collection_path")
                resumed = _normalize_collection_root(resumed)
                if resumed and _is_collection_root(resumed):
                    final_collection_path = resumed
                    log_dedup.info("Recovered final collection root from resume: %s", final_collection_path)
                else:
                    guessed = _guess_collection_for_step(temp_dir, final_step)
                    if guessed:
                        final_collection_path = guessed
                        log_dedup.info("Guessed final collection root from disk: %s", final_collection_path)

            final_merged_path = os.path.join(temp_dir, f"merged_step{final_step}")
            if not os.path.exists(final_merged_path):
                log_dedup.error("Final merged Parquet folder not found: %s", final_merged_path)
                client.close()
                cluster.close()
                return

            n_before = int(dd.read_parquet(final_merged_path).shape[0].compute())
            log_dedup.info("Rows before dedup (final_merged): %d", n_before)

            use_distributed = final_collection_path is not None and _is_collection_root(final_collection_path)

            # --- Safe config parsing ---
            tiebreaking_priority_cfg = translation_config.get("tiebreaking_priority")
            if isinstance(tiebreaking_priority_cfg, (str, bytes)):
                tiebreaking_priority_cfg = [str(tiebreaking_priority_cfg)]
            if not isinstance(tiebreaking_priority_cfg, (list, tuple)) or not tiebreaking_priority_cfg:
                raise TypeError("tiebreaking_priority must be a non-empty list of column names.")

            instrument_type_priority_cfg = translation_config.get("instrument_type_priority")
            if "instrument_type_homogenized" in set(tiebreaking_priority_cfg) and not isinstance(instrument_type_priority_cfg, dict):
                raise TypeError(
                    "instrument_type_priority must be a mapping when "
                    "'instrument_type_homogenized' is present in tiebreaking_priority."
                )
            delta_z_threshold_cfg = float(translation_config.get("delta_z_threshold", 0.0))

            if use_distributed:
                log_dedup.info("Running graph-based dedup on final merged collection (map_partitions)")
                final_collection_path = _normalize_collection_root(final_collection_path)
                log_dedup.info("final_collection_root: %s", final_collection_path)

                if not _is_collection_root(final_collection_path):
                    raise RuntimeError(f"Expected collection root at: {final_collection_path}")

                # Discover subcatalogs
                base = os.path.basename(final_collection_path.rstrip("/"))
                main_subcat_path = os.path.join(final_collection_path, base)
                log_dedup.info("Expected main subcatalog path: %s", main_subcat_path)
                if not _is_hats_subcatalog(main_subcat_path):
                    raise RuntimeError(f"Main subcatalog not found: {main_subcat_path}")

                final_cat = lsdb.open_catalog(main_subcat_path)
                log_dedup.info("Opened main subcatalog.")

                # Optional margin (*arcs), prefer 5arcs if present
                arcs_candidates = [p for p in glob.glob(os.path.join(final_collection_path, "*arcs")) if _is_hats_subcatalog(p)]
                if arcs_candidates:
                    def _arc_val(p):
                        m = re.search(r"([\d.]+)\s*arcs$", os.path.basename(p))
                        return float(m.group(1)) if m else float("inf")
                    if any(abs(_arc_val(p) - 5.0) < 1e-12 for p in arcs_candidates):
                        pick = min(arcs_candidates, key=lambda p: abs(_arc_val(p) - 5.0))
                    else:
                        pick = min(arcs_candidates, key=lambda p: _arc_val(p))
                    log_dedup.info("Attaching margin subcatalog: %s", pick)
                    final_cat.margin = lsdb.open_catalog(pick)
                else:
                    log_dedup.info("No *arcs margin found; proceeding without margin.")
                    final_cat.margin = None

                if not hasattr(final_cat, "_ddf"):
                    raise RuntimeError("Main subcatalog does not expose _ddf.")

                labels = run_dedup_with_lsdb_map_partitions(
                    final_cat,
                    tiebreaking_priority=tiebreaking_priority_cfg,
                    instrument_type_priority=(instrument_type_priority_cfg if isinstance(instrument_type_priority_cfg, dict) else None),
                    delta_z_threshold=delta_z_threshold_cfg,
                    crd_col="CRD_ID",
                    compared_col="compared_to",
                    z_col="z",
                    tie_col="tie_result",
                )

                df_all = dd.read_parquet(final_merged_path)
                try:
                    df_all = df_all.assign(CRD_ID=df_all["CRD_ID"].astype("string[pyarrow]"))
                    labels = labels.assign(CRD_ID=labels["CRD_ID"].astype("string[pyarrow]"))
                except Exception:
                    df_all = df_all.assign(CRD_ID=df_all["CRD_ID"].astype("string"))
                    labels = labels.assign(CRD_ID=labels["CRD_ID"].astype("string"))

                rhs_dd = labels.rename(columns={"tie_result": "tie_result_new"})

                for obj in (df_all, rhs_dd):
                    try:
                        obj = obj.clear_divisions()
                    except Exception:
                        pass

                with dask.config.set({"dataframe.shuffle.method": "tasks"}):
                    merged = dd.merge(df_all, rhs_dd, on="CRD_ID", how="left")

            else:
                log_dedup.warning(
                    "final_collection_path missing or not a collection root; falling back to driver-side pandas dedup."
                )
                base_dd = dd.read_parquet(final_merged_path, engine="pyarrow", split_row_groups=True)
                available = set(map(str, base_dd.columns))
                required_base = {"CRD_ID", "compared_to", "z"}
                missing_base = sorted(required_base - available)
                if missing_base:
                    raise KeyError(f"Missing required columns: {missing_base}")

                priority_set = set(tiebreaking_priority_cfg or [])
                missing_priority = sorted([c for c in priority_set if c not in available])
                if missing_priority:
                    raise KeyError(f"Missing priority columns: {missing_priority}")

                optional_candidates = {"z_flag_homogenized", "instrument_type_homogenized"}
                optional_present = (optional_candidates - priority_set) & available
                maybe_tie_result = {"tie_result"} & available
                needed = sorted(required_base | priority_set | optional_present | maybe_tie_result)
                pdf = base_dd[needed].compute()

                pdf_out = deduplicate_pandas(
                    pdf,
                    tiebreaking_priority=tiebreaking_priority_cfg,
                    instrument_type_priority=instrument_type_priority_cfg if isinstance(instrument_type_priority_cfg, dict) else None,
                    delta_z_threshold=delta_z_threshold_cfg,
                )

                df_all = dd.read_parquet(final_merged_path)
                rhs_pdf = pdf_out.loc[:, ["CRD_ID", "tie_result"]].rename(columns={"tie_result": "tie_result_new"})
                rhs_dd = dd.from_pandas(rhs_pdf, npartitions=max(1, df_all.npartitions))

                for obj in (df_all, rhs_dd):
                    try:
                        obj = obj.clear_divisions()
                    except Exception:
                        pass

                with dask.config.set({"dataframe.shuffle.method": "tasks"}):
                    merged = dd.merge(df_all, rhs_dd, on="CRD_ID", how="left")

            # Coalesce tie_result
            if "tie_result" in merged.columns:
                merged["tie_result"] = merged["tie_result_new"].fillna(merged["tie_result"])
            else:
                merged = merged.assign(tie_result=merged["tie_result_new"])
            merged = merged.drop(columns=["tie_result_new"])

            try:
                merged = merged.clear_divisions()
            except Exception:
                pass

            tie_f = dd.to_numeric(merged["tie_result"], errors="coerce").astype("float64")
            zf_f = dd.to_numeric(merged["z_flag_homogenized"], errors="coerce").astype("float64")

            try:
                merged = merged.clear_divisions()
            except Exception:
                pass

            # Stars: flag==6 => tie=3
            is_star = zf_f.eq(6.0)
            merged["tie_result"] = merged["tie_result"].mask(is_star, 3)

            # Forbid tie=3 when not a star
            wrong_3 = tie_f.eq(3.0) & ~is_star
            merged["tie_result"] = merged["tie_result"].mask(wrong_3, 0)

            # Guard: isolated & flag NaN => restore original tie_result
            cmp_str = merged["compared_to"].astype("string")
            cmp_empty = cmp_str.isna() | (cmp_str.str.strip().str.len().fillna(0) == 0)
            guard_mask = cmp_empty & zf_f.isna()

            n_restore = (
                guard_mask.map_partitions(lambda s: s.astype("int8").sum(), meta=("sum", "int64"))
                .sum()
                .compute()
            )
            log_dedup.info("Guard restore count (isolated & flag NaN): %d", int(n_restore))
            if int(n_restore) > 0:
                orig_tr = (
                    dd.read_parquet(final_merged_path, columns=["CRD_ID", "tie_result"])
                    .rename(columns={"tie_result": "tie_result_orig"})
                )
                try:
                    merged = merged.clear_divisions()
                except Exception:
                    pass
                try:
                    orig_tr = orig_tr.clear_divisions()
                except Exception:
                    pass
                with dask.config.set({"dataframe.shuffle.method": "tasks"}):
                    merged = dd.merge(merged, orig_tr, on="CRD_ID", how="left")
                merged["tie_result"] = merged["tie_result"].mask(guard_mask, merged["tie_result_orig"])
                merged = merged.drop(columns=["tie_result_orig"])

            _ = (tie_f.eq(3.0) & ~zf_f.eq(6.0)).map_partitions(lambda s: s.astype("int64").sum(), meta=("sum", "int64")).sum().compute()
            _ = (zf_f.eq(6.0) & ~tie_f.eq(3.0)).map_partitions(lambda s: s.astype("int64").sum(), meta=("sum", "int64")).sum().compute()

            merged["tie_result"] = merged["tie_result"].astype("Int8")

            try:
                df_final_dd = merged.clear_divisions()
            except Exception:
                df_final_dd = merged
            df_final = df_final_dd.compute()

            log_dedup.info("END deduplication: labels merged back into final dataframe")

            # Next phase runs below:
            start_consolidate = True

        else:
            base_logger.error("Unknown combine_mode: %s", combine_mode, extra={"phase": "crossmatch"})
            client.close()
            cluster.close()
            return

    # ---------------------------------------------------------------
    # 5) CONSOLIDATION / EXPORT
    # ---------------------------------------------------------------
    log_cons = _phase_logger(base_logger, "consolidation")
    log_cons.info("START consolidation: post-processing and export (out=%s)", out_root_and_dir)

    try:
        n_after_dedup = int(len(df_final))
        log_cons.info("Rows after dedup (in-memory): %d", n_after_dedup)
    except Exception as e:
        log_cons.warning("Could not compute len(df_final): %s", e)

    if combine_mode == "concatenate" and "tie_result" in df_final.columns:
        log_cons.info("Dropping 'tie_result' (concatenate mode)")
        df_final = df_final.drop(columns=["tie_result"])

    if USE_ARROW_TYPES:
        df_final = df_final.convert_dtypes(dtype_backend="pyarrow")
        for c in df_final.columns:
            if pd.api.types.is_string_dtype(df_final[c].dtype):
                df_final[c] = df_final[c].astype(DTYPE_STR)

    # Drop all-empty columns
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
        log_cons.info("Dropping all-missing columns: %s", ", ".join(sorted(map(str, to_drop))))
        df_final = df_final.drop(columns=to_drop)

    final_base_path = os.path.join(out_root_and_dir, output_name)
    save_dataframe(df_final, final_base_path, output_format)
    log_cons.info("Final combined catalog saved at %s.%s", final_base_path, output_format)

    relative_path = os.path.join(output_dir, f"{output_name}.{output_format}")
    expected_columns = ["id", "ra", "dec", "z", "z_flag", "z_err", "survey"]
    columns_assoc = {col: col for col in expected_columns if col in df_final.columns}

    update_process_info(
        process_info,
        process_info_path,
        "outputs",
        [
            {
                "path": relative_path,
                "root_dir": output_root_dir,
                "role": "main",
                "columns_assoc": columns_assoc,
            }
        ],
    )
    update_process_info(process_info, process_info_path, "end_time", str(pd.Timestamp.now()))
    update_process_info(process_info, process_info_path, "status", "Successful")

    if delete_temp_files:
        try:
            shutil.rmtree(temp_dir)
            log_cons.info("Deleted entire temp_dir after successful pipeline completion: %s", temp_dir)
        except Exception as e:
            log_cons.warning("Could not delete temp_dir %s: %s", temp_dir, e)

    log_cons.info("END consolidation: export complete")
    client.close()
    cluster.close()


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
    try:
        main(args.config_path, args.cwd, args.base_dir)
    finally:
        dur = time.time() - start_ts
        lg = logging.getLogger("crc")
        if lg.handlers:
            # keep phase to help time-profiler catch the trailer as 'consolidation'
            logging.LoggerAdapter(lg, {"phase": "consolidation"}).info("Pipeline completed in %.2f seconds.", dur)
        else:
            print(f"✅ Pipeline completed in {dur:.2f} seconds.")
