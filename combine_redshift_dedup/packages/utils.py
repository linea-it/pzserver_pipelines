# combine_redshift_dedup/packages/utils.py

import logging
import pathlib
import yaml
import os

def setup_logger(name="combine_redshift_dedup", logdir="."):
    """
    Set up and return a logger that writes to <logdir>/<name>.log
    and also outputs messages to the console.
    
    Args:
        name (str): Logger name and base for log filename.
        logdir (str): Directory where the log file will be saved.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler (logs everything)
    file_handler = logging.FileHandler(pathlib.Path(logdir, f"{name}.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler (INFO and above)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Prevent duplicate handlers in repeated runs
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger

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
    Write a dictionary as YAML to a file.

    Args:
        filepath (str): Path to the output YAML file.
        content (dict): Data to save.
        encoding (str): Encoding for the file.
    """
    with open(filepath, 'w', encoding=encoding) as f:
        yaml.dump(content, f)

def log_step(log_file, step_name):
    """
    Record a completed step to the process log file.

    Args:
        log_file (str): Path to the log file.
        step_name (str): Step identifier to record.
    """
    with open(log_file, "a") as f:
        f.write(f"{step_name}\n")

def read_completed_steps(log_file):
    """
    Read the list of completed steps from the process log file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        list: List of completed step names.
    """
    if not os.path.exists(log_file):
        return []
    with open(log_file) as f:
        return [line.strip() for line in f if line.strip()]
