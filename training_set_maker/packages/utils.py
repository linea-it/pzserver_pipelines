import logging
import pathlib
import shutil
from typing import Any

import yaml


def create_logdir(cwd: str) -> pathlib.Path:
    """Create a log directory if it does not exist.
    Args:
        cwd (str): Current working directory where the log directory will be created.
    """
    logdir = pathlib.Path(cwd, "process_info")
    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)
    return logdir


def setup_logger(name="training_set_maker", logdir="."):
    """
    Configures the logger for recording events and messages.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logname = "pipeline.log"

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    filename = pathlib.Path(logdir, logname)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger, logname


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


def copy_file(src: str, dst: str):
    """Copy a file from src to dst

    Args:
        src (str): source file path
        dst (str): destination file path
    """

    shutil.copy2(src, dst)


def copy_files_by_extensions(src_dir: str, dst_dir: str, extensions: list):
    """Copy files with specific extensions from src_dir to dst_dir

    Args:
        src_dir (str): source directory path
        dst_dir (str): destination directory path
        extensions (list): list of file extensions to filter by (e.g., ['.txt', '.csv'])
    """

    pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)

    for ext in extensions:
        for file in pathlib.Path(src_dir).glob(f"*{ext}"):
            shutil.copy2(file, dst_dir)


def copy_directory(src_dir: str, dst_dir: str):
    """Copy a directory from src_dir to dst_dir

    Args:
        src_dir (str): source directory path
        dst_dir (str): destination directory path
    """

    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
