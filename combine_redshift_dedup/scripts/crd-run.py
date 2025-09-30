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
def _log_remote_future_exception(
    lg: logging.LoggerAdapter, fut, msg_prefix: str, extra: dict | None = None
) -> None:
    """
    Logs the remote (worker-side) traceback carried by a Dask Future.
    """
    import traceback as _tb
    try:
        err = fut.exception()
        tb = fut.traceback()  # remote TB; can be None
        if tb is not None:
            lg.error("%s: %r", msg_prefix, err, exc_info=(type(err), err, tb), extra=extra)
        else:
            # Fallback: driver-side traceback (should still exist)
            lg.error("%s: %r\n%s", msg_prefix, err, _tb.format_exc(), extra=extra)
    except Exception:
        lg.error("%s (and failed to render remote traceback)\n%s", msg_prefix, _tb.format_exc(), extra=extra)

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


def _cleanup_previous_step(
    step_index: int,
    prepared_info: list[dict[str, Any]],
    temp_dir: str,
    lg: logging.LoggerAdapter
) -> None:
    """Delete artifacts from *all* previous steps to save disk space.

    For every step k < step_index, removes:
      - prepared_<internal_name>
      - prepared_<internal_name>_hats
      - prepared_<internal_name>_hats_auto
      - merged_step<k>
      - merged_step<k>_hats
    Also removes any recorded collection_path for previous prepared entries.
    """

    def _rm_path(p: str) -> None:
        if not p:
            return
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.isfile(p):
                os.remove(p)
            else:
                return
            lg.info("Deleted artifact: %s", p)
        except Exception as e:
            lg.warning("Could not delete %s: %s", p, e)

    # 1) Remove all merged_step<k> and merged_step<k>_hats for k < step_index
    try:
        for entry in os.listdir(temp_dir):
            full = os.path.join(temp_dir, entry)
            if not os.path.isdir(full):
                continue

            # Accept both patterns:
            #   - merged_step<k>
            #   - merged_step<k>_hats
            name = entry.strip()
            if not name.startswith("merged_step"):
                continue

            tail = name.replace("merged_step", "", 1)
            # tail can be like "5" or "5_hats"
            num_str = tail.split("_", 1)[0].strip()
            try:
                k = int(num_str)
            except Exception:
                continue

            if k < step_index:
                _rm_path(full)
    except Exception as e:
        lg.warning("Could not list merged_step folders under %s: %s", temp_dir, e)

    # 2) Remove prepared artifacts for all previous prepared entries
    for i in range(0, min(step_index, len(prepared_info))):
        prev = prepared_info[i]

        # Base prepared_<internal_name>
        base_prepared = prev.get("prepared_path")
        if base_prepared:
            _rm_path(base_prepared)
            _rm_path(base_prepared + "_hats")
            _rm_path(base_prepared + "_hats_auto")

        # Recorded collection_path (may point to hats/auto variants)
        coll = prev.get("collection_path")
        if coll:
            _rm_path(coll)


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
# Parallel crossmatch worker
# ---------------------------------------------------------------------------
def _xmatch_worker(
    left_collection_path: str,
    right_collection_path: str,
    logs_dir: str,
    temp_dir: str,
    step: int,
    translation_config: dict,
) -> str:
    """
    Open two HATS collections and run crossmatch_tiebreak_safe.
    Returns a collection path (root or subcatalog).
    """
    try:
        ensure_crc_logger(logs_dir)
    except Exception:
        pass

    left_cat = lsdb.open_catalog(left_collection_path)
    right_cat = lsdb.open_catalog(right_collection_path)

    return crossmatch_tiebreak_safe(
        left_cat=left_cat,
        right_cat=right_cat,
        logs_dir=logs_dir,
        temp_dir=temp_dir,
        step=step,
        client=None,  # not needed on LSDB path
        translation_config=translation_config,
        do_import=True,
    )

# ---------------------------------------------------------------------------
# Cleanup for crossmatch tournment
# ---------------------------------------------------------------------------
def _cleanup_inputs_of_merge(
    left_root: str,
    right_root: str,
    lg: logging.LoggerAdapter,
) -> None:
    """
    Remove only the two input collections that were just merged.
    Accepts either a collection root or a subcatalog and normalizes to root.
    """
    def _to_root(p: str) -> str:
        n = _normalize_collection_root(p) or p
        return n

    def _rm_root(p: str) -> None:
        if not p:
            return
        try:
            if os.path.isdir(p) and _is_collection_root(p):
                shutil.rmtree(p, ignore_errors=True)
                lg.info("Deleted input collection: %s", p)
        except Exception as e:
            lg.warning("Could not delete input collection %s: %s", p, e)

    lroot = _to_root(left_root)
    rroot = _to_root(right_root)
    if lroot and rroot and (lroot != rroot):
        if _is_collection_root(lroot):
            _rm_root(lroot)
        if _is_collection_root(rroot):
            _rm_root(rroot)

# ---------------------------------------------------------------------------
# Publish / copy helpers
# ---------------------------------------------------------------------------
def _copy_file(src: str, dst: str, lg: logging.LoggerAdapter) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    abs_src, abs_dst = os.path.abspath(src), os.path.abspath(dst)
    if abs_src == abs_dst:
        lg.info("Skip copy: source equals destination (%s).", abs_dst)
        return
    try:
        if os.path.exists(abs_dst):
            os.remove(abs_dst)
        os.link(abs_src, abs_dst)  # hardlink se for mesmo FS
        lg.info("Hardlinked: %s -> %s", abs_src, abs_dst)
    except Exception:
        shutil.copy2(abs_src, abs_dst)
        lg.info("Copied: %s -> %s", abs_src, abs_dst)

def _copy_tree(src_dir: str, dst_dir: str, lg: logging.LoggerAdapter) -> None:
    """Copia recursivamente src_dir → dst_dir, sobrescrevendo se existir."""
    if not os.path.isdir(src_dir):
        lg.warning("Source dir not found for copy: %s", src_dir)
        return
    os.makedirs(dst_dir, exist_ok=True)
    for root, dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        tgt_root = os.path.join(dst_dir, rel) if rel != "." else dst_dir
        os.makedirs(tgt_root, exist_ok=True)
        for f in files:
            _copy_file(os.path.join(root, f), os.path.join(tgt_root, f), lg)

def _snapshot_shell_logs(src_dir: str, dst_dir: str, lg: logging.LoggerAdapter, max_size_mb: int = 100) -> None:
    """Copy *.log/*.err/*.out (and *.N rotations) from src_dir to dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    patterns = ["*.log", "*.log.*", "*.err", "*.err.*", "*.out", "*.out.*"]
    seen = set()
    for pat in patterns:
        for path in glob.glob(os.path.join(src_dir, pat)):
            if path in seen:
                continue
            seen.add(path)
            try:
                if os.path.getsize(path) > max_size_mb * 1024 * 1024:
                    lg.info("Skip (>%d MB): %s", max_size_mb, path)
                    continue
            except Exception:
                pass
            _copy_file(path, os.path.join(dst_dir, os.path.basename(path)), lg)

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

    # --- Paths ---
    output_root_dir = config["output_root_dir"]
    output_dir = config["output_dir"]
    out_root_and_dir = os.path.join(output_root_dir, output_dir)
    output_name = config["output_name"]
    output_format = config.get("output_format", "parquet").lower()

    logs_dir = os.path.join(base_dir, "process_info")
    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    launch_dir = os.environ.get("CRC_LAUNCH_DIR", ".")
    launch_snap_dir = os.path.join(logs_dir, "launch_dir_files")
    print(f"Pipeline was called from: {launch_dir}")

    try:
        _copy_file(config_path, os.path.join(base_dir, "config.yaml"), logging.LoggerAdapter(logging.getLogger("crc"), {"phase": "init"}))
    except Exception:
        pass

    # --- process.yml bookkeeping ---
    try:
        main_process_info = os.path.join(cwd, "process.yml")
        if os.path.exists(main_process_info):
            _copy_file(
                main_process_info,
                os.path.join(base_dir, "process.yml"),
                logging.LoggerAdapter(logging.getLogger("crc"), {"phase": "init"}),
            )
    except Exception:
        pass

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
 
    # --- Dask cluster/client ---
    cluster = get_executor(config["executor"], logs_dir=logs_dir)
    client = Client(cluster)
    
    # Ensure the minimum number of workers start within 10 seconds
    exec_args = config.get("executor", {}).get("args", {}) or {}
    instance_cfg = exec_args.get("instance", {}) or {}
    scale_cfg = exec_args.get("scale", {}) or {}
    
    procs = int(instance_cfg.get("processes", 1) or 1)
    min_jobs = scale_cfg.get("minimum_jobs")
    min_workers = 0 if min_jobs is None else int(min_jobs) * procs
    
    if min_workers > 0:
        log_init.info("Waiting up to 10s for minimum_workers=%d to start...", min_workers)
        try:
            client.wait_for_workers(min_workers, timeout=10)
        except Exception:
            current_workers = len(client.scheduler_info().get("workers", {}))
            log_init.warning(
                "Timeout: waited 10 seconds but minimum_workers=%d did not start "
                "(current=%d). Proceeding anyway.",
                min_workers,
                current_workers,
            )
        else:
            current_workers = len(client.scheduler_info().get("workers", {}))
            log_init.info(
                "Confirmed: minimum_workers=%d started within 10s (current=%d).",
                min_workers,
                current_workers,
            )

    current_workers = len(client.scheduler_info().get("workers", {}))
    log_init.info(
        "WORKERS STILL RUNNING=%d.",
        current_workers,
    )
    
    log_init.info("END init: pipeline bootstrap")

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
                param_config,
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

        current_workers = len(client.scheduler_info().get("workers", {}))
        log_prep.info(
            "WORKERS STILL RUNNING=%d.",
            current_workers,
        )

        log_prep.info("END preparation: finished prepared collections")

        # ---------------------------------------------------------------
        # Detect pipeline state for resume (is crossmatch already done?)
        # ---------------------------------------------------------------
        final_step = len(prepared_info) - 1
        resume_log = os.path.join(temp_dir, "process_resume.log")

        def _recover_final_collection_path() -> str | None:
            #1) Try to take from the resume of the last step
            resumed = _resume_get(resume_log, f"crossmatch_step{final_step}.collection_path")
            resumed = _normalize_collection_root(resumed)
            if resumed and _is_collection_root(resumed):
                return resumed
            #2) Try to guess on the disk
            guessed = _guess_collection_for_step(temp_dir, final_step)
            if guessed and _is_collection_root(guessed):
                return guessed
            return None

        merged_final_path = os.path.join(temp_dir, f"merged_step{final_step}")
        crossmatch_already_done = (
            (f"crossmatch_step{final_step}" in completed) or
            os.path.isdir(merged_final_path)
        )

        # If you have already finished crossmatching, retrieve the collection root (if it exists) for distributed dedup.
        final_collection_path = _recover_final_collection_path() if crossmatch_already_done else None

        # ---------------------------------------------------------------
        # 2) AUTO MATCH (self crossmatch over each prepared)
        # ---------------------------------------------------------------
        log_auto = _phase_logger(base_logger, "automatch")
        if combine_mode in ("concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"):

            if crossmatch_already_done:
                # No auto needed: merges already completed and merged_step{final} present
                log_auto.info(
                    "Skip automatch: crossmatch already finalized (merged_step%d exists or logged).",
                    final_step,
                )
            else:
                log_auto.info("START automatch: generating *_hats_auto from prepared collections")

                queue_auto: list[dict] = []
                already_done = 0
                for info in prepared_info:
                    tag = f"autocross_{info['internal_name']}"
                    hats_auto = info["prepared_path"] + "_hats_auto"
                    if tag in completed and os.path.isdir(hats_auto):
                        already_done += 1
                    else:
                        queue_auto.append(info)

                max_inflight_auto = int(param_config.get("auto_cross_max_inflight", 5))
                log_auto.info(
                    "Concurrency for auto-cross: max_inflight=%d, to_run=%d, already_done=%d",
                    max_inflight_auto, len(queue_auto), already_done,
                )

                if queue_auto:
                    ac2 = as_completed()
                    inflight2 = []
                    fut2info: dict[Any, dict] = {}

                    def _submit_auto(i: dict):
                        return client.submit(_auto_cross_worker, i, logs_dir, translation_config, pure=False)

                    for _ in range(min(max_inflight_auto, len(queue_auto))):
                        info = queue_auto.pop(0)
                        fut = _submit_auto(info)
                        ac2.add(fut)
                        inflight2.append(fut)
                        fut2info[fut] = info

                    auto_done_now = 0
                    while inflight2:
                        fut = next(ac2)
                        inflight2.remove(fut)
                        try:
                            out_path = fut.result()
                        except Exception as e:
                            info_err = fut2info.pop(fut)
                            log_auto.error("Auto crossmatch failed for %s: %s", info_err["internal_name"], e)
                            raise
                        info_ok = fut2info.pop(fut)
                        info_ok["collection_path"] = out_path
                        log_step(resume_log, f"autocross_{info_ok['internal_name']}")
                        auto_done_now += 1
                        if queue_auto:
                            nxt = queue_auto.pop(0)
                            fut2 = _submit_auto(nxt)
                            ac2.add(fut2)
                            inflight2.append(fut2)
                            fut2info[fut2] = nxt

                    log_auto.info("Auto crossmatch completed for %d catalogs (re/computed)", auto_done_now)
                else:
                    log_auto.info("No auto-crossmatch needed (all *_hats_auto present).")

                current_workers = len(client.scheduler_info().get("workers", {}))
                log_auto.info(
                    "WORKERS STILL RUNNING=%d.",
                    current_workers,
                )

                log_auto.info("END automatch: all *_hats_auto guaranteed on disk")

            # -----------------------------------------------------------
            # Enforcement of *_hats_auto ONLY if we are still going to run crossmatch
            # -----------------------------------------------------------
            if not crossmatch_already_done:
                missing_auto: list[str] = []
                for info in prepared_info:
                    hats_auto = info["prepared_path"] + "_hats_auto"
                    if os.path.isdir(hats_auto):
                        info["collection_path"] = hats_auto
                    else:
                        missing_auto.append(info["internal_name"])
                if missing_auto:
                    raise FileNotFoundError(
                        "Auto-cross outputs still missing after recovery: "
                        + ", ".join(missing_auto)
                        + ". Check disk/permissions/logs."
                    )


        # ---------------------------------------------------------------
        # 3) CROSSMATCH (parallel tournament; no per-step resume)
        # ---------------------------------------------------------------
        log_cross = _phase_logger(base_logger, "crossmatch")
        log_cross.info("START crossmatch (parallel tournament) over prepared *_hats_auto")

        if combine_mode == "concatenate":
            # Simple concat mode still passes through consolidation phase later.
            log_cross.info("Concatenate mode selected (no crossmatch).")
            try:
                log_cross.info("Concatenating %d prepared catalogs.", len(prepared_info))
                for _pi in prepared_info[:5]:
                    log_cross.debug("Prepared path candidate: %s", _pi.get("prepared_path"))
                df_final = dd.concat([dd.read_parquet(i["prepared_path"]) for i in prepared_info])
                _ = df_final.head(1, compute=True)  # small schema sanity
                df_final = df_final.compute()
                log_cross.info("Concatenate compute finished: shape=%s", tuple(df_final.shape))
            except Exception as e:
                import traceback as _tb
                log_cross.error("FAILED while concatenating prepared catalogs: %s\n%s", repr(e), _tb.format_exc())
                raise

            try:
                current_workers = len(client.scheduler_info().get("workers", {}))
                log_cross.info("WORKERS STILL RUNNING=%d.", current_workers)
            except Exception as e:
                log_cross.debug("Could not query scheduler_info workers: %s", e)

            log_cross.info("END crossmatch: concatenate mode (no crossmatch performed)")
            start_consolidate = True

        elif combine_mode in ("concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"):

            if crossmatch_already_done:
                # Crossmatch chain already finished previously (by merged_step{final} presence)
                log_cross.info(
                    "Skip crossmatch: already finalized at step %d (merged_step present).",
                    final_step,
                )
                start_consolidate = False  # dedup still runs before consolidation

            else:
                # === Parallel tournament without per-step resume ===
                try:
                    _resume_set(resume_log, "phase.crossmatch", "started", log_cross)

                    # Ensure all prepared_info[i]["collection_path"] are *_hats_auto
                    for info in prepared_info:
                        hats_auto = info["prepared_path"] + "_hats_auto"
                        if os.path.isdir(hats_auto):
                            info["collection_path"] = hats_auto
                        else:
                            raise FileNotFoundError(f"Missing *_hats_auto for: {info['internal_name']}")

                    max_inflight_pairs = int(param_config.get("cross_max_inflight", 5))
                    log_cross.info("Max inflight crossmatch pairs: %d", max_inflight_pairs)

                    # Build initial queue of nodes (label, root_path)
                    from collections import deque
                    nodes = deque()
                    for info in prepared_info:
                        root = _normalize_collection_root(info["collection_path"]) or info["collection_path"]
                        if not _is_hats_collection(root):
                            raise RuntimeError(f"Not a HATS collection: {root}")
                        root = _normalize_collection_root(root) or root  # normalize to root if subcatalog
                        nodes.append((info["internal_name"], root))

                    # Edge case: only one catalog
                    if len(nodes) == 1:
                        final_collection_path = nodes[0][1]
                        _resume_set(resume_log, "phase.crossmatch", "completed", log_cross)
                        log_cross.info("Only one prepared collection; skipping crossmatch. final=%s", final_collection_path)
                        start_consolidate = False
                    else:
                        # Futures pipeline
                        ready = deque(nodes)
                        carry: tuple[str, str] | None = None

                        # Use future.key (string) to avoid identity KeyError
                        inflight_meta: dict[str, tuple] = {}  # fut.key -> (l_lab, l_path, r_lab, r_path, step_id)
                        step_id = 1  # diagnostic numbering only

                        # as_completed instance where we add futures as we submit them
                        ac3 = as_completed()

                        def _submit_pair() -> bool:
                            nonlocal step_id
                            if len(ready) < 2:
                                return False
                            if len(inflight_meta) >= max_inflight_pairs:
                                return False
                            l_lab, l_path = ready.popleft()
                            r_lab, r_path = ready.popleft()
                            fut = client.submit(
                                _xmatch_worker,
                                l_path,
                                r_path,
                                logs_dir,
                                temp_dir,
                                step_id,
                                translation_config,
                                pure=False,
                            )
                            inflight_meta[fut.key] = (l_lab, l_path, r_lab, r_path, step_id)
                            ac3.add(fut)  # add immediately to as_completed
                            log_cross.info("Submitted crossmatch pair: step=%d | %s VS %s", step_id, l_lab, r_lab)
                            step_id += 1
                            return True

                        # Prime the pump
                        while _submit_pair():
                            pass

                        final_collection_path: str | None = None

                        while inflight_meta or (len(ready) + (1 if carry else 0) > 1):
                            # Process the next completed pair if any
                            if inflight_meta:
                                fut = next(ac3)
                                meta = inflight_meta.pop(fut.key, None)
                                if meta is None:
                                    # Received a completion for an unknown/detached future; log and continue.
                                    _log_remote_future_exception(
                                        log_cross, fut,
                                        "Received completion for an unknown Future"
                                    )
                                    continue

                                l_lab, l_path, r_lab, r_path, sid = meta
                                try:
                                    raw_out = fut.result()
                                except Exception:
                                    _log_remote_future_exception(
                                        log_cross, fut,
                                        f"Crossmatch failed (step {sid}: {l_lab} vs {r_lab})"
                                    )
                                    raise

                                out_root = _normalize_collection_root(raw_out) or raw_out
                                if not _is_hats_collection(out_root):
                                    log_cross.warning("crossmatch returned a non-HATS path (step %d): %s", sid, raw_out)

                                # Delicate cleanup — delete only the two inputs consumed by this merge
                                if delete_temp_files:
                                    try:
                                        _cleanup_inputs_of_merge(l_path, r_path, log_cross)
                                    except Exception as e:
                                        log_cross.warning("Cleanup of inputs for step %d failed (non-fatal): %s", sid, e)

                                # Winner goes back into the queue
                                ready.append((f"merged_step{sid}", out_root))
                                final_collection_path = out_root  # keep last seen

                                # Keep the pipeline saturated
                                while _submit_pair():
                                    pass

                            # If we have a carry and something ready, try to form a new pair
                            if carry and ready:
                                ready.appendleft(carry)
                                carry = None
                                while _submit_pair():
                                    pass

                            # If odd and nothing inflight, stash one as carry
                            if not inflight_meta and len(ready) > 1 and (len(ready) % 2 == 1):
                                carry = ready.pop()

                        # Finalization
                        if final_collection_path is None:
                            if ready:
                                final_collection_path = ready[0][1]
                            elif carry:
                                final_collection_path = carry[1]

                        final_collection_path = _normalize_collection_root(final_collection_path) or final_collection_path
                        if not _is_collection_root(final_collection_path):
                            raise RuntimeError(f"Final path is not a HATS collection root: {final_collection_path}")

                        _resume_set(resume_log, "phase.crossmatch", "completed", log_cross)
                        log_cross.info("END crossmatch: parallel tournament done; final root: %s", final_collection_path)
                        start_consolidate = False

                except Exception:
                    # Catch ANY driver-side error in the tournament
                    log_cross.exception("Crossmatch tournament FAILED with an unhandled error")
                    try:
                        _resume_set(resume_log, "phase.crossmatch", "failed", log_cross)
                    except Exception:
                        pass
                    raise

        else:
            base_logger.error("Unknown combine_mode: %s", combine_mode, extra={"phase": "crossmatch"})
            client.close()
            cluster.close()
            return


        # -----------------------------------------------------------
        # 4) DEDUPLICATION
        # -----------------------------------------------------------
        log_dedup = _phase_logger(base_logger, "deduplication")

        # In "concatenate" mode there is no LSDB collection root; skip dedup entirely.
        if combine_mode == "concatenate":
            log_dedup.info("Skip deduplication: combine_mode='concatenate' (no LSDB collection root).")
            # Next phase runs below:
            start_consolidate = True

        else:
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

            #######################################################################
            # Diagnostics / outputs
            # - edge_log: enable edge diagnostics (warn on star-neighbor exclusions)
            # - group_col: set to None to disable exporting group labels
            edge_log = True
            group_col = "group_id"  # None to deactivate
            #######################################################################

            if use_distributed:
                log_dedup.info("Running graph-based dedup on final merged collection (Dask merge with catalog ._ddf)")
                final_collection_path = _normalize_collection_root(final_collection_path)
                log_dedup.info("final_collection_root: %s", final_collection_path)

                if not _is_collection_root(final_collection_path):
                    raise RuntimeError(f"Expected collection root at: {final_collection_path}")

                # Open the LSDB collection directly (no subcatalog hunting, no margins here)
                try:
                    final_cat = lsdb.open_catalog(final_collection_path)
                    log_dedup.info("Opened LSDB collection at root path.")
                except Exception as e:
                    raise RuntimeError(f"Could not open LSDB collection at: {final_collection_path}") from e

                # Build labels lazily over the catalog
                try:
                    labels_dd = run_dedup_with_lsdb_map_partitions(
                        final_cat,
                        tiebreaking_priority=tiebreaking_priority_cfg,
                        instrument_type_priority=(instrument_type_priority_cfg if isinstance(instrument_type_priority_cfg, dict) else None),
                        delta_z_threshold=delta_z_threshold_cfg,
                        crd_col="CRD_ID",
                        compared_col="compared_to",
                        z_col="z",
                        tie_col="tie_result",
                        edge_log=edge_log,
                        group_col=group_col,
                    )
                    log_dedup.info("Labels graph built (lazy). Preparing RHS for Dask merge...")
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED while building labels graph: %s\n%s", repr(e), _tb.format_exc())
                    raise

                # Prepare RHS minimal schema (Dask DataFrame)
                try:
                    rhs_dd = labels_dd.rename(columns={"tie_result": "tie_result_new"})
                    keep_cols = ["CRD_ID", "tie_result_new"]
                    if group_col and (group_col in rhs_dd.columns):
                        keep_cols.append(group_col)
                    rhs_dd = rhs_dd[keep_cols]
                    log_dedup.info("Prepared RHS columns: %s", keep_cols)
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED while preparing RHS labels: %s\n%s", repr(e), _tb.format_exc())
                    raise

                # Base Dask DataFrame from the LSDB collection
                try:
                    df_all = final_cat._ddf
                    # Align join key dtype defensively
                    try:
                        df_all = df_all.assign(CRD_ID=df_all["CRD_ID"].astype("string[pyarrow]"))
                        rhs_dd = rhs_dd.assign(CRD_ID=rhs_dd["CRD_ID"].astype("string[pyarrow]"))
                    except Exception:
                        df_all = df_all.assign(CRD_ID=df_all["CRD_ID"].astype("string"))
                        rhs_dd = rhs_dd.assign(CRD_ID=rhs_dd["CRD_ID"].astype("string"))
                    log_dedup.info("Aligned CRD_ID dtype on both sides.")
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED while preparing base from catalog: %s\n%s", repr(e), _tb.format_exc())
                    raise

                # Clear divisions and perform the distributed merge
                try:
                    try:
                        df_all = df_all.clear_divisions()
                    except Exception:
                        pass
                    try:
                        rhs_dd = rhs_dd.clear_divisions()
                    except Exception:
                        pass
                    with dask.config.set({"dataframe.shuffle.method": "tasks"}):
                        merged = dd.merge(df_all, rhs_dd, on="CRD_ID", how="left")
                    log_dedup.info("Dask merge graph built (lazy).")
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED during Dask merge: %s\n%s", repr(e), _tb.format_exc())
                    raise

                # Coalesce tie_result in Dask (still lazy)
                try:
                    if "tie_result" in merged.columns:
                        merged["tie_result"] = merged["tie_result_new"].fillna(merged["tie_result"])
                    else:
                        merged = merged.assign(tie_result=merged["tie_result_new"])
                    if "tie_result_new" in merged.columns:
                        merged = merged.drop(columns=["tie_result_new"])
                    log_dedup.info("Coalesced tie_result (lazy).")
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED while coalescing tie_result: %s\n%s", repr(e), _tb.format_exc())
                    raise

                # Final materialization: compute only once, at the very end
                try:
                    # Small sanity check without pulling everything
                    _ = merged.head(1)
                    df_final = merged.compute()

                    if "group_id" in df_final.columns:
                        dup = df_final["group_id"].value_counts(dropna=True)
                        n_big = int((dup > 1).sum())
                        log_dedup.info("Sanity group_id: uniques=%d, ids_with_>1_occurrences=%d", int(dup.size), n_big)
                    else:
                        log_dedup.info("No group_id column present after dedup; skipping sanity counts.")

                    current_workers = len(client.scheduler_info().get("workers", {}))
                    log_dedup.info("WORKERS STILL RUNNING=%d.", current_workers)
                    log_dedup.info("END deduplication: labels merged back into final dataframe (Dask)")
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED while computing final dataframe: %s\n%s", repr(e), _tb.format_exc())
                    raise

            else:
                # We require a valid LSDB collection root; Parquet fallback is no longer supported.
                raise RuntimeError("Cannot run dedup: missing or invalid LSDB collection root; Parquet fallback removed.")

            # Next phase runs below:
            start_consolidate = True


    # ---------------------------------------------------------------
    # 5) CONSOLIDATION / EXPORT
    # ---------------------------------------------------------------
    log_cons = _phase_logger(base_logger, "consolidation")
    log_cons.info("START consolidation: staging artifacts into process dir (base_dir=%s)", base_dir)

    # Snapshot about df_final to aid debugging
    try:
        log_cons.info("df_final shape: %s", tuple(df_final.shape))
        try:
            mem_bytes = int(df_final.memory_usage(deep=True).sum())
            log_cons.info("df_final memory footprint: %.2f MB", mem_bytes / (1024 * 1024))
        except Exception as e_mem:
            log_cons.debug("Could not compute memory footprint: %s", e_mem)
        try:
            dtypes_preview = {str(k): str(v) for k, v in list(df_final.dtypes.items())[:20]}
            log_cons.info("df_final dtypes (first 20): %s", dtypes_preview)
        except Exception as e_dt:
            log_cons.debug("Could not collect dtype preview: %s", e_dt)
    except Exception as e_snap:
        log_cons.warning("Could not snapshot df_final: %s", e_snap)

    try:
        n_rows_final = int(len(df_final))
        log_cons.info("Rows in final dataframe (in-memory): %d", n_rows_final)
    except Exception as e:
        log_cons.warning("Could not compute len(df_final): %s", e)


    if combine_mode == "concatenate" and "tie_result" in df_final.columns:
        log_cons.info("Dropping 'tie_result' (concatenate mode)")
        try:
            df_final = df_final.drop(columns=["tie_result"])
        except Exception as e:
            log_cons.error("FAILED dropping 'tie_result' in concatenate mode: %s", e)
            raise

    if combine_mode == "concatenate_and_remove_duplicates":
        if "tie_result" not in df_final.columns:
            log_cons.warning("Expected 'tie_result' column for removal mode, but it is missing; skipping row filter.")
        else:
            log_cons.info("Filtering winners by tie_result == 1 (remove-duplicates mode)")
            try:
                tie_num = pd.to_numeric(df_final["tie_result"], errors="coerce").fillna(0).astype("int8")
                keep_mask = tie_num.eq(1)
                kept = int(keep_mask.sum())
                dropped = int(len(df_final) - kept)
                df_final = df_final.loc[keep_mask].copy()
                log_cons.info("Removed duplicates by tie_result==1: kept=%d rows, dropped=%d rows.", kept, dropped)
            except Exception as e:
                log_cons.error("FAILED while filtering by tie_result==1: %s", e)
                raise

    if USE_ARROW_TYPES:
        log_cons.info("Converting dtypes with dtype_backend='pyarrow' (USE_ARROW_TYPES=True)")
        try:
            df_final = df_final.convert_dtypes(dtype_backend="pyarrow")
            for c in df_final.columns:
                try:
                    if pd.api.types.is_string_dtype(df_final[c].dtype):
                        df_final[c] = df_final[c].astype(DTYPE_STR)
                except Exception as e_cast:
                    log_cons.debug("Could not cast column '%s' to Arrow string: %s", c, e_cast)
        except Exception as e:
            log_cons.error("FAILED during convert_dtypes(dtype_backend='pyarrow'): %s", e)
            raise

    # Drop all-empty columns
    try:
        to_drop = []
        for col in df_final.columns:
            dt = df_final[col].dtype
            try:
                if str(dt) == "string[pyarrow]" or str(dt) == "object":
                    all_missing = df_final[col].apply(lambda x: (pd.isna(x) or str(x).strip() == ""))
                else:
                    all_missing = df_final[col].isna()
                if bool(all_missing.all()):
                    to_drop.append(col)
            except Exception as e_col:
                log_cons.debug("Skip emptiness check for column '%s' (dtype=%s): %s", col, dt, e_col)
        if to_drop:
            log_cons.info("Dropping all-missing columns: %s", ", ".join(sorted(map(str, to_drop))))
            df_final = df_final.drop(columns=to_drop)
        else:
            log_cons.info("No all-missing columns to drop.")
    except Exception as e:
        log_cons.error("FAILED while dropping all-missing columns: %s", e)
        raise

    # Stage final output with your save_dataframe
    staged_output_base = os.path.join(base_dir, output_name)
    log_cons.info("About to call save_dataframe(base=%s, format=%s)", staged_output_base, output_format)
    try:
        try:
            log_cons.debug("df_final head(3):\n%s", df_final.head(3))
        except Exception as e_head:
            log_cons.debug("Could not preview df_final.head(3): %s", e_head)

        save_dataframe(df_final, staged_output_base, output_format)
        log_cons.info("Staged final output at %s.%s", staged_output_base, output_format)
    except Exception as e:
        import traceback as _tb
        log_cons.error("FAILED in save_dataframe: %s\n%s", repr(e), _tb.format_exc())
        raise

    relative_path = os.path.join(output_dir, f"{output_name}.{output_format}")

    expected_columns = ["id", "ra", "dec", "z", "z_flag", "z_err", "survey"]
    columns_assoc = {}

    # Special handling for id
    try:
        if "CRD_ID" in df_final.columns:
            columns_assoc["id"] = "CRD_ID"
        elif "id" in df_final.columns:
            columns_assoc["id"] = "id"
    except Exception as e:
        log_cons.debug("While mapping 'id' column: %s", e)

    # Special handling for z_flag
    try:
        if "z_flag_homogenized" in df_final.columns:
            columns_assoc["z_flag"] = "z_flag_homogenized"
        elif "z_flag" in df_final.columns:
            columns_assoc["z_flag"] = "z_flag"
    except Exception as e:
        log_cons.debug("While mapping 'z_flag' column: %s", e)

    # Identity mapping for the others
    try:
        for col in expected_columns:
            if col not in ("id", "z_flag") and col in df_final.columns:
                columns_assoc[col] = col
        log_cons.info("columns_assoc: %s", columns_assoc)
    except Exception as e:
        log_cons.debug("While building columns_assoc: %s", e)

    # Update process info
    try:
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
        log_cons.info("Process info updated with output: %s", relative_path)
    except Exception as e:
        import traceback as _tb
        log_cons.error("FAILED to update process_info: %s\n%s", e, _tb.format_exc())
        raise

    # ----------------- PUBLISH STEP -----------------
    publish_logger = _phase_logger(base_logger, "register")
    publish_logger.info("START publish: copying staged artifacts from process dir to out_root_and_dir (%s)", out_root_and_dir)

    try:
        _snapshot_shell_logs(str(launch_dir), launch_snap_dir, publish_logger, max_size_mb=100)
        publish_logger.info("Snapshotted shell logs from %s into %s", launch_dir, launch_snap_dir)
    except Exception as e:
        publish_logger.warning("Snapshot of shell logs failed: %s", e)

    # Create publish dir only now
    try:
        os.makedirs(out_root_and_dir, exist_ok=True)
        publish_logger.info("Ensured publish dir exists: %s", out_root_and_dir)
    except Exception as e:
        publish_logger.error("FAILED to create publish dir '%s': %s", out_root_and_dir, e)
        raise

    # 1) process_info/
    try:
        _copy_tree(os.path.join(base_dir, "process_info"), os.path.join(out_root_and_dir, "process_info"), publish_logger)
    except Exception as e:
        publish_logger.error("FAILED to copy process_info/: %s", e)
        raise

    # 2) process.yml and process.yaml
    try:
        _copy_file(os.path.join(base_dir, "process.yml"),  os.path.join(out_root_and_dir, "process.yml"),  publish_logger)
        if os.path.exists(os.path.join(base_dir, "process.yaml")):
            _copy_file(os.path.join(base_dir, "process.yaml"), os.path.join(out_root_and_dir, "process.yaml"), publish_logger)
    except Exception as e:
        publish_logger.error("FAILED to copy process.yml/.yaml: %s", e)
        raise

    # 3) config.yaml
    try:
        if os.path.exists(os.path.join(base_dir, "config.yaml")):
            _copy_file(os.path.join(base_dir, "config.yaml"), os.path.join(out_root_and_dir, "config.yaml"), publish_logger)
    except Exception as e:
        publish_logger.error("FAILED to copy config.yaml: %s", e)
        raise

    # 4) final output
    try:
        src_out = f"{staged_output_base}.{output_format}"
        dst_out = os.path.join(out_root_and_dir, f"{output_name}.{output_format}")
        publish_logger.info("Copying final output: %s -> %s", src_out, dst_out)
        _copy_file(src_out, dst_out, publish_logger)
    except Exception as e:
        publish_logger.error("FAILED to copy final output to publish dir: %s", e)
        raise

    publish_logger.info("END publish: artifacts copied to %s", out_root_and_dir)

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
# Helpers
# ---------------------------------------------------------------------------
def _pick_next_process_dir(root: str) -> str:
    """Return first non-existing 'processNNN' path under root."""
    i = 1
    while True:
        name = f"process{i:03d}"
        path = os.path.join(root, name)
        if not os.path.exists(path):
            return path
        i += 1

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine redshift catalogs via preparation, crossmatch (no tie-breaking), and graph-based deduplication."
    )
    parser.add_argument("config_path", help="Path to YAML config file.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory (default: current dir).")
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Base directory for outputs and logs. If omitted, picks processNNN under --cwd."
    )
    args = parser.parse_args()

    # Resolve working dir and base_dir
    workdir = os.path.abspath(args.cwd)
    os.makedirs(workdir, exist_ok=True)

    base_dir = args.base_dir
    if not base_dir:
        base_dir = _pick_next_process_dir(workdir)

    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # Optional: echo the chosen run dir early for visibility when logs aren't wired yet
    print(f"▶ Using run directory: {base_dir}")

    start_ts = time.time()
    ok = False
    try:
        # Pass resolved paths
        main(args.config_path, workdir, base_dir)
        ok = True
    finally:
        dur = time.time() - start_ts
        lg = logging.getLogger("crc")
        msg = f"Pipeline {'completed successfully' if ok else 'terminated with errors'} in {dur:.2f} seconds. (run dir: {base_dir})"
        if lg.handlers:
            logging.LoggerAdapter(lg, {"phase": "consolidation"}).info(msg)
        else:
            print(("✅ " if ok else "❌ ") + msg)
        if not ok:
            sys.exit(1)
