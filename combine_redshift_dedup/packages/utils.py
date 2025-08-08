# combine_redshift_dedup/packages/utils.py

import logging
import os
import pathlib
import sys
import traceback
import warnings

import pandas as pd
import yaml

# ============================================================
# Logger setup and global access
# ============================================================


def setup_logger(name="combine_redshift_dedup", logdir="."):
    """
    Set up and return a logger that writes to <logdir>/<name>.log
    and also prints messages to the console.

    Args:
        name (str): Logger name and base for log filename.
        logdir (str): Directory where the log file will be saved.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(pathlib.Path(logdir, "pipeline.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


# === Global logger access ===

_global_logger = None


def set_global_logger(logger):
    global _global_logger
    _global_logger = logger


def get_global_logger():
    """
    Retrieve the globally stored logger.
    Falls back to `logging.getLogger(__name__)` if not set.
    """
    return _global_logger or logging.getLogger(__name__)


def log_and_print(message, logger):
    """
    Print a message and log it as INFO.

    Args:
        message (str): Message to display and log.
        logger (logging.Logger): Logger instance.
    """
    print(message)
    logger.info(message)


# ============================================================
# YAML utilities
# ============================================================


def load_yml(filepath):
    """
    Load and parse a YAML file.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yml(filepath, content, encoding="utf-8"):
    """
    Save a dictionary as a YAML file.

    Args:
        filepath (str): Path to the output YAML file.
        content (dict): Data to be saved.
        encoding (str): File encoding.
    """
    with open(filepath, "w", encoding=encoding) as f:
        yaml.dump(content, f)


# ============================================================
# Process tracking
# ============================================================


def log_step(log_file, step_name):
    """
    Record a completed pipeline step in the resume log.

    Args:
        log_file (str): Path to the log file.
        step_name (str): Step identifier to record.
    """
    with open(log_file, "a") as f:
        f.write(f"{step_name}\n")


def read_completed_steps(log_file):
    """
    Read the list of completed steps from the log file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        list[str]: List of completed step names.
    """
    if not os.path.exists(log_file):
        return []
    with open(log_file) as f:
        return [line.strip() for line in f if line.strip()]


def update_process_info(process_info, process_info_path, key, value):
    """
    Update a key in process.yml and persist the changes.

    Args:
        process_info (dict): Current process info dictionary.
        process_info_path (str): Path to process.yml.
        key (str): Key to update.
        value (Any): Value to assign.
    """
    process_info[key] = value
    dump_yml(process_info_path, process_info)


# ============================================================
# Error and warning handling
# ============================================================


def configure_warning_handler(logger):
    """
    Redirect Python warnings to the logger (no stderr printing).

    Args:
        logger (logging.Logger): Logger to handle warnings.
    """

    def custom_warning_handler(
        message, category, filename, lineno, file=None, line=None
    ):
        warning_msg = f"{category.__name__}: {message} ({filename}:{lineno})"
        logger.warning(warning_msg)

    warnings.showwarning = custom_warning_handler


def configure_exception_hook(logger, process_info, process_info_path):
    """
    Redirect uncaught exceptions to the logger and update process.yml.

    Args:
        logger (logging.Logger): Logger instance.
        process_info (dict): Current process info.
        process_info_path (str): Path to process.yml.
    """

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        traceback.print_exception(exc_type, exc_value, exc_traceback)
        logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

        try:
            update_process_info(process_info, process_info_path, "status", "Failed")
            update_process_info(
                process_info, process_info_path, "end_time", str(pd.Timestamp.now())
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not update process.yml after failure: {e}")

    sys.excepthook = handle_exception
