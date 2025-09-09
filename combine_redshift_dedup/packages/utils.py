# combine_redshift_dedup/packages/utils.py
# -*- coding: utf-8 -*-

"""Shared utilities for CRC pipeline: logging, YAML IO, process tracking, and hooks.

This module centralizes logging configuration for the whole pipeline and provides
a lightweight UDP log collector to mirror logs from Dask workers back to the
driver's console.

Idempotency:
    It is safe to call `ensure_crc_logger()` multiple times in the same process;
    configuration is guarded per-PID and avoids duplicate handlers.

Environment variables:
    CRC_LOG_COLLECTOR: Optional "<host>:<port>" for the UDP collector running on
        the driver. When set, each process adds a JSON `DatagramHandler` that
        forwards every `LogRecord` to the collector, which then re-emits into
        the unified "crc" logger (appearing in the driver's terminal).
    CRC_LOG_FORCE_LOCAL_FILE: "1" forces file logging also in workers.
    CRC_LOG_FORCE_RECONFIG: "1" forces reconfiguration even if already configured
        within the same PID (useful if you need to change log_dir at runtime).
"""

from __future__ import annotations

import json
import logging
import os
import socketserver
import sys
import threading
import traceback
import warnings
from contextlib import contextmanager
from logging.handlers import DatagramHandler, RotatingFileHandler
from pathlib import Path
from typing import Any, Tuple

import pandas as pd  # kept for other utilities in this module
import yaml          # kept for other utilities in this module


# =============================================================================
# Globals
# =============================================================================

# Single-process lock to make handler setup atomic.
_CRC_LOG_LOCK = threading.RLock()


# =============================================================================
# Formatters & Filters
# =============================================================================

class CRCFormatter(logging.Formatter):
    """Formatter with microsecond timestamps and phase support.

    Time format example: 2025-09-08-15:11:51.780823
    """

    default_msec_format = "%s.%03d"  # unused; we override formatTime

    def formatTime(self, record, datefmt=None):
        from datetime import datetime
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime("%Y-%m-%d-%H:%M:%S.") + f"{int(dt.microsecond):06d}"


class _EnsurePhase(logging.Filter):
    """Ensures `record.phase` exists; defaults to '-'."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "phase"):
            record.phase = "-"
        return True


class _OnlyCRC(logging.Filter):
    """Allows only records from logger 'crc' and its children."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "crc" or record.name.startswith("crc.")


# =============================================================================
# Logger helpers
# =============================================================================

def get_phase_logger(phase: str, base: logging.Logger | None = None) -> logging.LoggerAdapter:
    """Return a LoggerAdapter that injects `phase` into every record.

    Args:
        phase: Phase label to inject (e.g., "preparation", "crossmatch").
        base: Base logger; defaults to 'crc'.

    Returns:
        logging.LoggerAdapter: Adapter that enriches records with `phase`.
    """
    if base is None:
        base = logging.getLogger("crc")
    return logging.LoggerAdapter(base, {"phase": phase})


@contextmanager
def log_phase(phase: str, step_id: str | None = None, base: logging.Logger | None = None):
    """Context manager to log START/END markers with duration.

    Args:
        phase: Phase label to inject into logs.
        step_id: Optional label (e.g., a step name or index) to include.
        base: Base logger; defaults to 'crc'.
    """
    log = get_phase_logger(phase, base)
    label = f"{step_id}" if step_id else ""
    import time as _t
    t0 = _t.time()
    log.info("▶ START %s", label)
    try:
        yield log
    finally:
        dt = _t.time() - t0
        log.info("✅ END %s (%.2fs)", label, dt)


# =============================================================================
# UDP collector (driver-side)
# =============================================================================

class _CRCUDPHandler(socketserver.BaseRequestHandler):
    """Receives JSON-encoded LogRecords and re-emits them into 'crc'."""

    def handle(self) -> None:
        data = self.request[0]
        try:
            payload = json.loads(data.decode("utf-8"))

            # Normalize args for logging formatter compatibility.
            if "args" in payload and isinstance(payload["args"], list):
                payload["args"] = tuple(payload["args"])

            # Anti-echo: ignore messages emitted by this same PID.
            if payload.get("emitter_pid") == os.getpid():
                return

            record = logging.makeLogRecord(payload)
        except Exception:
            return
        logging.getLogger("crc").handle(record)


def start_crc_log_collector(host: str = "127.0.0.1", port: int = 19997) -> Tuple[socketserver.UDPServer, threading.Thread]:
    """Start a UDP log collector and re-emit incoming records into logger 'crc'.

    Args:
        host: Interface to bind (use driver's reachable IP for multi-host clusters).
        port: UDP port to listen on.

    Returns:
        Tuple[socketserver.UDPServer, threading.Thread]: The server and its daemon thread.
    """
    server = socketserver.UDPServer((host, port), _CRCUDPHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t


# =============================================================================
# UDP JSON handler (worker/driver-side sender)
# =============================================================================

class _JSONDatagramHandler(DatagramHandler):
    """Sends LogRecords as JSON text to the UDP collector."""

    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        # Mark for identification without relying solely on isinstance.
        self._crc_json_udp = True

    def makePickle(self, record: logging.LogRecord) -> bytes:  # noqa: N802
        """Serialize a LogRecord to JSON bytes suitable for UDP transport."""
        d = record.__dict__.copy()
        d["emitter_pid"] = os.getpid()

        # Preserve final message to avoid dependency on args formatting.
        try:
            formatted = record.getMessage()
        except Exception:
            formatted = str(record.msg)
        d["msg"] = formatted
        d["args"] = ()

        # Ensure exc_info is a string, if present.
        if "exc_info" in d and d["exc_info"] and not isinstance(d["exc_info"], str):
            d["exc_info"] = logging._defaultFormatter.formatException(d["exc_info"])

        # Make the payload JSON-serializable.
        for k in list(d.keys()):
            try:
                json.dumps(d[k])
            except Exception:
                d[k] = str(d[k])

        return json.dumps(d, ensure_ascii=False).encode("utf-8")


# =============================================================================
# Logger setup
# =============================================================================

def ensure_crc_logger(log_dir: str, level: int = logging.INFO) -> logging.Logger:
    """Ensure the unified 'crc' logger is configured exactly once per process.

    Behavior:
        - Driver: console + rotating file; optional UDP mirror if CRC_LOG_COLLECTOR is set.
        - Dask workers: UDP mirror; console disabled if CRC_LOG_COLLECTOR is set
          (to avoid duplicate logs on the driver's console). File handler only if
          CRC_LOG_FORCE_LOCAL_FILE=1 is set.
        - Header: "<ts> | <LEVEL> | <phase> | <name> | <message>"

    Args:
        log_dir: Directory where the rotating file log will be written.
        level: Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured 'crc' logger.
    """
    from pathlib import Path as _Path

    # Detect if we are inside a Dask worker.
    def _in_dask_worker() -> bool:
        try:
            from dask.distributed import get_worker  # type: ignore
            get_worker()
            return True
        except Exception:
            return False

    _Path(log_dir).mkdir(parents=True, exist_ok=True)
    desired_log_path = str(_Path(log_dir) / "pipeline.log")

    is_worker = _in_dask_worker()
    collector = os.getenv("CRC_LOG_COLLECTOR", "")             # "host:port"
    force_local_file = os.getenv("CRC_LOG_FORCE_LOCAL_FILE", "0") == "1"
    force_reconfig   = os.getenv("CRC_LOG_FORCE_RECONFIG",   "0") == "1"

    fmt = CRCFormatter("%(asctime)s | %(levelname)s | %(phase)s | %(name)s | %(message)s")
    only_crc = _OnlyCRC()
    ensure_phase = _EnsurePhase()

    with _CRC_LOG_LOCK:
        logger = logging.getLogger("crc")
        logger.setLevel(level)
        logger.propagate = False

        current_pid = os.getpid()
        already_configured_pid = getattr(logger, "_crc_configured_pid", None)
        already_configured = (already_configured_pid == current_pid) and bool(logger.handlers)

        # Idempotent guard: if already configured in this PID and not forced, return.
        if already_configured and not force_reconfig:
            return logger

        # Ensure child loggers propagate and have no handlers of their own.
        mgr = logging.Logger.manager
        for name, obj in list(mgr.loggerDict.items()):
            if isinstance(obj, logging.Logger) and name.startswith("crc.") and name != "crc":
                obj.propagate = True
                if obj.handlers:
                    obj.handlers.clear()

        # Fresh handler set.
        if logger.handlers:
            logger.handlers.clear()

        # Console handler:
        # - Driver: always add console.
        # - Worker: add console only if no collector is set (local debug scenario).
        add_console = True
        if is_worker and collector:
            add_console = False

        if add_console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(fmt)
            ch.addFilter(only_crc)
            ch.addFilter(ensure_phase)
            logger.addHandler(ch)

        # File handler:
        # - Driver only by default.
        # - Workers only if forced via CRC_LOG_FORCE_LOCAL_FILE=1.
        if (not is_worker) or force_local_file:
            fh = RotatingFileHandler(desired_log_path, maxBytes=20_000_000, backupCount=5, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            fh.addFilter(only_crc)
            fh.addFilter(ensure_phase)
            logger.addHandler(fh)

        # Optional UDP mirror (driver and workers).
        if collector:
            try:
                host, port_str = collector.split(":")
                port = int(port_str)
                if not any(isinstance(h, _JSONDatagramHandler) for h in logger.handlers):
                    dh = _JSONDatagramHandler(host, port)
                    dh.setLevel(level)
                    dh.addFilter(only_crc)
                    dh.addFilter(ensure_phase)
                    logger.addHandler(dh)
            except Exception as e:
                logger.warning(
                    "CRC_LOG_COLLECTOR invalid or failed to attach: %s",
                    e,
                    extra={"phase": "logging"},
                )

        # Reduce third-party verbosity once per PID (or if reconfiguring).
        if already_configured_pid != current_pid or force_reconfig:
            logging.getLogger().setLevel(logging.WARNING)
            logging.getLogger("dask").setLevel(logging.ERROR)
            logging.getLogger("distributed").setLevel(logging.ERROR)
            logging.getLogger("lsdb").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.ERROR)

        # Mark configured for this PID.
        logger._crc_configured_pid = current_pid
        return logger


# ============================================================
# Back-compat helpers (kept minimal on purpose)
# ============================================================


def setup_logger(name: str = "combine_redshift_dedup", logdir: str = ".") -> logging.Logger:
    """[DEPRECATED] Return the unified 'crc' logger.

    Args:
        name: Ignored. Present for backward compatibility.
        logdir: Ignored. Present for backward compatibility.

    Returns:
        The centralized "crc" logger.
    """
    warnings.warn(
        "utils.setup_logger is deprecated. Use ensure_crc_logger(log_dir) once per process.",
        DeprecationWarning,
        stacklevel=2,
    )
    return logging.getLogger("crc")


# ============================================================
# Global logger access
# ============================================================

_global_logger: logging.Logger | None = None


def set_global_logger(logger: logging.Logger) -> None:
    """Set a global logger reference for convenience.

    Args:
        logger: Logger to be reused by helpers that don't receive a logger.
    """
    global _global_logger
    _global_logger = logger


def get_global_logger() -> logging.Logger:
    """Return the global pipeline logger or the unified 'crc' logger as fallback.

    Returns:
        A logger instance.
    """
    if _global_logger is not None:
        return _global_logger
    return logging.getLogger("crc")


def log_and_print(message: str, logger: logging.Logger) -> None:
    """Print a message to stdout and log it as INFO.

    Args:
        message: Message to display and log.
        logger: Logger instance.
    """
    print(message)
    logger.info(message)


# ============================================================
# YAML utilities
# ============================================================


def load_yml(filepath: str) -> dict:
    """Load and parse a YAML file.

    Args:
        filepath: Path to the YAML file.

    Returns:
        Parsed YAML content.
    """
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yml(filepath: str, content: dict, encoding: str = "utf-8") -> None:
    """Save a dictionary as a YAML file.

    Args:
        filepath: Path to the output YAML file.
        content: Data to be saved.
        encoding: File encoding.
    """
    with open(filepath, "w", encoding=encoding) as f:
        yaml.dump(content, f)


# ============================================================
# Process tracking
# ============================================================


def log_step(log_file: str, step_name: str) -> None:
    """Record a completed pipeline step in the resume log.

    Args:
        log_file: Path to the log file.
        step_name: Step identifier to record.
    """
    with open(log_file, "a") as f:
        f.write(f"{step_name}\n")


def read_completed_steps(log_file: str) -> list[str]:
    """Read the list of completed steps from the resume log.

    Args:
        log_file: Path to the log file.

    Returns:
        List of completed step names.
    """
    if not os.path.exists(log_file):
        return []
    with open(log_file) as f:
        return [line.strip() for line in f if line.strip()]


def update_process_info(process_info: dict, process_info_path: str, key: str, value: Any) -> None:
    """Update a key in process.yml and persist the changes.

    Args:
        process_info: Current process info dictionary.
        process_info_path: Path to process.yml.
        key: Key to update.
        value: Value to assign.
    """
    process_info[key] = value
    dump_yml(process_info_path, process_info)


# ============================================================
# Error and warning handling
# ============================================================


def configure_warning_handler(logger: logging.Logger) -> None:
    """Redirect Python warnings to the logger (and suppress default stderr prints).

    Args:
        logger: Logger to handle warnings.
    """

    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        warning_msg = f"{category.__name__}: {message} ({filename}:{lineno})"
        logger.warning(warning_msg)

    warnings.showwarning = custom_warning_handler


def configure_exception_hook(logger: logging.Logger, process_info: dict, process_info_path: str) -> None:
    """Redirect uncaught exceptions to the logger and update process.yml.

    Args:
        logger: Logger instance.
        process_info: Current process info.
        process_info_path: Path to process.yml.
    """

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        traceback.print_exception(exc_type, exc_value, exc_traceback)
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        try:
            update_process_info(process_info, process_info_path, "status", "Failed")
            update_process_info(process_info, process_info_path, "end_time", str(pd.Timestamp.now()))
        except Exception as e:
            logger.warning(f"⚠️ Could not update process.yml after failure: {e}")

    sys.excepthook = handle_exception
