import logging
import pathlib
from typing import Any

import yaml


def create_logdir(cwd: str) -> pathlib.Path:
    """Create a log directory if it does not exist.
    Args:
        cwd (str): Current working directory where the log directory will be created.
    """
    logdir = pathlib.Path(cwd, "logs")
    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)
    return logdir


def setup_logger(name="tsm", logdir="."):
    """
    Configures the logger for recording events and messages.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(pathlib.Path(logdir, f"{name}.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def load_yml(filepath: str) -> Any:
    """Load yaml file

    Args:
        filepath (str): filepath

    Returns:
        Any: yaml file content
    """

    with open(filepath, encoding="utf-8") as _file:
        content = yaml.safe_load(_file)

    return content


def dump_yml(filepath, content, encoding="utf-8"):
    """Dump yaml file

    Args:
        filepath (str): filepath output
        content (dict): yaml content
    """

    with open(filepath, "w", encoding=encoding) as _file:
        yaml.dump(content, _file)
