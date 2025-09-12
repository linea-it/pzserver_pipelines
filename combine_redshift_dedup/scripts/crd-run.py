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

            if crossmatch_already_done:
                # Crossmatch chain already finished previously (by resume log or presence of merged_step{final})
                log_cross.info(
                    "Skip crossmatch loop: already finalized at step %d (resume/merged present).",
                    final_step,
                )
                # No df_final here; dedup will read merged_step{final} directly.
                start_consolidate = False  # dedup still runs before consolidation

            else:
                # === Normal iterative crossmatch flow ===
                init = prepared_info[0]
                cat_prev = lsdb.open_catalog(init["collection_path"])  # *_hats_auto
                start_i = 1
                final_collection_path = None  # set when last step finishes

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

                    # Open next *_hats_auto catalog and crossmatch
                    cat_curr = lsdb.open_catalog(info_i["collection_path"])
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
                start_consolidate = False


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

            #######################################################################
            # Diagnostics / outputs
            # - edge_log: enable edge diagnostics (warn on star-neighbor exclusions)
            # - group_col: set to None to disable exporting group labels
            edge_log = True
            group_col = "group_id"  # None to deactivate
            #######################################################################

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

                # Compute labels per partition. Guard-restore is inside deduplicate_pandas.
                try:
                    labels = run_dedup_with_lsdb_map_partitions(
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
                    log_dedup.info("Labels Dask graph built (lazy). Preparing merge...")
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED while building labels Dask graph: %s\n%s", repr(e), _tb.format_exc())
                    raise

                # Read once for the final merge; align CRD_ID dtypes with labels.
                try:
                    df_all = dd.read_parquet(final_merged_path)
                    try:
                        df_all  = df_all.assign(CRD_ID=df_all["CRD_ID"].astype("string[pyarrow]"))
                        labels  = labels.assign(CRD_ID=labels["CRD_ID"].astype("string[pyarrow]"))
                    except Exception:
                        df_all  = df_all.assign(CRD_ID=df_all["CRD_ID"].astype("string"))
                        labels  = labels.assign(CRD_ID=labels["CRD_ID"].astype("string"))
                    log_dedup.info("Aligned CRD_ID dtypes between base and labels.")
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED while reading base parquet or aligning dtypes: %s\n%s", repr(e), _tb.format_exc())
                    raise

                # Keep only the columns needed for the merge; rename tie_result.
                try:
                    rhs_dd = labels.rename(columns={"tie_result": "tie_result_new"})
                    keep_cols = ["CRD_ID", "tie_result_new"]
                    if group_col and (group_col in rhs_dd.columns):
                        keep_cols.append(group_col)
                    rhs_dd = rhs_dd[keep_cols]
                    log_dedup.info("Prepared labels rhs_dd with columns: %s", keep_cols)
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED while preparing rhs_dd: %s\n%s", repr(e), _tb.format_exc())
                    raise

                # Clear divisions before shuffle-based merge (assign back!).
                try:
                    try:
                        df_all = df_all.clear_divisions()
                    except Exception:
                        pass
                    try:
                        rhs_dd = rhs_dd.clear_divisions()
                    except Exception:
                        pass
                    log_dedup.info("Cleared divisions for shuffle merge.")
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED while clearing divisions: %s\n%s", repr(e), _tb.format_exc())
                    raise

                # Merge and compute labels to catch issues early
                try:
                    with dask.config.set({"dataframe.shuffle.method": "tasks"}):
                        merged = dd.merge(df_all, rhs_dd, on="CRD_ID", how="left")
                    log_dedup.info("Dask merge graph built (lazy). Computing a small sample for sanity...")
                    # Compute a small sample to validate schema early
                    _ = merged.head(1)
                    log_dedup.info("Merge sample computed successfully.")
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED during merge (distributed branch): %s\n%s", repr(e), _tb.format_exc())
                    raise

            else:
                log_dedup.warning(
                    "final_collection_path missing or not a collection root; falling back to driver-side pandas dedup."
                )
                try:
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
                    maybe_tie_result = {"tie_result"} & available  # bring original tie_result if present
                    needed = sorted(required_base | priority_set | optional_present | maybe_tie_result)
                    pdf = base_dd[needed].compute()
                    log_dedup.info("Driver-side dataframe computed for dedup: shape=%s", tuple(pdf.shape))
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED while loading base data for driver-side dedup: %s\n%s", repr(e), _tb.format_exc())
                    raise

                # Run solver on driver (guard-restore happens inside).
                try:
                    pdf_out = deduplicate_pandas(
                        pdf,
                        tiebreaking_priority=tiebreaking_priority_cfg,
                        instrument_type_priority=instrument_type_priority_cfg if isinstance(instrument_type_priority_cfg, dict) else None,
                        delta_z_threshold=delta_z_threshold_cfg,
                        crd_col="CRD_ID",
                        compared_col="compared_to",
                        z_col="z",
                        tie_col="tie_result",
                        edge_log=edge_log,
                        partition_tag="[driver]",
                        logger=_phase_logger(),
                        group_col=group_col,
                    )
                    log_dedup.info("Driver-side labels computed: shape=%s", tuple(pdf_out.shape))
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED inside deduplicate_pandas (driver branch): %s\n%s", repr(e), _tb.format_exc())
                    raise

                # Merge new labels back (driver branch)
                try:
                    df_all = dd.read_parquet(final_merged_path)
                    cols = ["CRD_ID", "tie_result"]
                    if group_col and (group_col in pdf_out.columns):
                        cols.append(group_col)

                    rhs_pdf = pdf_out.loc[:, cols].rename(columns={"tie_result": "tie_result_new"})
                    rhs_dd = dd.from_pandas(rhs_pdf, npartitions=max(1, df_all.npartitions))

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
                    log_dedup.info("Driver-side merge graph built.")
                except Exception as e:
                    import traceback as _tb
                    log_dedup.error("FAILED during merge (driver branch): %s\n%s", repr(e), _tb.format_exc())
                    raise

            # Coalesce tie_result with the newly computed labels.
            try:
                if "tie_result" in merged.columns:
                    merged["tie_result"] = merged["tie_result_new"].fillna(merged["tie_result"])
                else:
                    merged = merged.assign(tie_result=merged["tie_result_new"])
                merged = merged.drop(columns=["tie_result_new"])
                log_dedup.info("Coalesced tie_result with new labels.")
            except Exception as e:
                import traceback as _tb
                log_dedup.error("FAILED while coalescing tie_result: %s\n%s", repr(e), _tb.format_exc())
                raise

            # Stable dtypes for final output.
            try:
                try:
                    merged = merged.clear_divisions()
                except Exception:
                    pass

                merged["tie_result"] = merged["tie_result"].astype("Int8")
                if "group_id" in merged.columns:
                    merged["group_id"] = merged["group_id"].astype("Int64")
                log_dedup.info("Final dtypes stabilized for tie_result/group_id.")
            except Exception as e:
                import traceback as _tb
                log_dedup.error("FAILED while stabilizing final dtypes: %s\n%s", repr(e), _tb.format_exc())
                raise

            # Materialize result.
            try:
                try:
                    df_final_dd = merged.clear_divisions()
                except Exception:
                    df_final_dd = merged

                # Extra sanity before compute
                head_sample = df_final_dd.head(1, compute=True)
                log_dedup.info("Pre-compute sample OK: columns=%s", list(head_sample.columns))

                df_final = df_final_dd.compute()

                dup = df_final["group_id"].value_counts(dropna=True)
                n_big = int((dup > 1).sum())
                log_dedup.info("Sanity group_id: uniques=%d, ids_com_>1_ocorr=%d", int(dup.size), n_big)

                log_dedup.info("END deduplication: labels merged back into final dataframe")
            except Exception as e:
                import traceback as _tb
                log_dedup.error("FAILED while computing final dataframe: %s\n%s", repr(e), _tb.format_exc())
                raise

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
    log_cons.info("START consolidation: staging artifacts into process dir (base_dir=%s)", base_dir)

    try:
        n_after_dedup = int(len(df_final))
        log_cons.info("Rows after dedup (in-memory): %d", n_after_dedup)
    except Exception as e:
        log_cons.warning("Could not compute len(df_final): %s", e)

    if combine_mode == "concatenate" and "tie_result" in df_final.columns:
        log_cons.info("Dropping 'tie_result' (concatenate mode)")
        df_final = df_final.drop(columns=["tie_result"])

    if combine_mode == "concatenate_and_remove_duplicates":
        if "tie_result" not in df_final.columns:
            log_cons.warning("Expected 'tie_result' column for removal mode, but it is missing; skipping row filter.")
        else:
            # Keep only rows marked as winners (tie_result == 1)
            tie_num = pd.to_numeric(df_final["tie_result"], errors="coerce").fillna(0).astype("int8")
            keep_mask = tie_num.eq(1)
            kept = int(keep_mask.sum())
            dropped = int(len(df_final) - kept)
            df_final = df_final.loc[keep_mask].copy()
            log_cons.info("Removed duplicates by tie_result==1: kept=%d rows, dropped=%d rows.", kept, dropped)

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

    staged_output_base = os.path.join(base_dir, output_name)
    save_dataframe(df_final, staged_output_base, output_format)
    log_cons.info("Staged final output at %s.%s", staged_output_base, output_format)

    relative_path = os.path.join(output_dir, f"{output_name}.{output_format}")

    expected_columns = ["id", "ra", "dec", "z", "z_flag", "z_err", "survey"]
    columns_assoc = {}

    # Special handling for id
    if "CRD_ID" in df_final.columns:
        columns_assoc["id"] = "CRD_ID"
    elif "id" in df_final.columns:
        columns_assoc["id"] = "id"

    # Special handling for z_flag
    if "z_flag_homogenized" in df_final.columns:
        columns_assoc["z_flag"] = "z_flag_homogenized"
    elif "z_flag" in df_final.columns:
        columns_assoc["z_flag"] = "z_flag"

    # Identity mapping for the others
    for col in expected_columns:
        if col not in ("id", "z_flag") and col in df_final.columns:
            columns_assoc[col] = col


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

    # ----------------- PUBLISH STEP (finalzinho) -----------------
    publish_logger = _phase_logger(base_logger, "register")
    publish_logger.info("START publish: copying staged artifacts from process dir to out_root_and_dir (%s)", out_root_and_dir)

    try:
        _snapshot_shell_logs(str(launch_dir), launch_snap_dir, publish_logger, max_size_mb=100)
        publish_logger.info("Snapshotted shell logs from %s into %s", launch_dir, launch_snap_dir)
    except Exception as e:
        publish_logger.warning("Snapshot of shell logs failed: %s", e)    

    # Cria pasta de publicação só agora
    os.makedirs(out_root_and_dir, exist_ok=True)

    # 1) process_info/
    _copy_tree(os.path.join(base_dir, "process_info"), os.path.join(out_root_and_dir, "process_info"), publish_logger)

    # 2) process.yml e alias process.yaml
    _copy_file(os.path.join(base_dir, "process.yml"),  os.path.join(out_root_and_dir, "process.yml"),  publish_logger)
    if os.path.exists(os.path.join(base_dir, "process.yaml")):
        _copy_file(os.path.join(base_dir, "process.yaml"), os.path.join(out_root_and_dir, "process.yaml"), publish_logger)

    # 3) config.yaml
    if os.path.exists(os.path.join(base_dir, "config.yaml")):
        _copy_file(os.path.join(base_dir, "config.yaml"), os.path.join(out_root_and_dir, "config.yaml"), publish_logger)

    # 4) output final
    _copy_file(f"{staged_output_base}.{output_format}",
               os.path.join(out_root_and_dir, f"{output_name}.{output_format}"),
               publish_logger)

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
