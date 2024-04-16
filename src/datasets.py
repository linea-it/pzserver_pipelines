import os
import logging
import pathlib
import yaml


class MetaDatasets(object):
    def __init__(self) -> None:
        self.logger = logging.getLogger()

        try:
            datasets_dir = os.environ["DATASETS_DIR"]
        except KeyError:
            self.logger.exception("DATASETS_DIR not defined.")

        with open(
            pathlib.Path(datasets_dir, "datasets.yml"), encoding="utf-8"
        ) as _file:
            self.__datasets = yaml.safe_load(_file)

    def get_dataset(self, name: str) -> dict:
        """get dataset by name

        Args:
            name (str): dataset name

        Returns:
            dict: dataset info
        """
        return self.__datasets.get(name)

    def get_dataset_path(self, name: str) -> pathlib.Path:
        """get dataset path by name

        Args:
            name (str): dataset name

        Returns:
            pathlib.Path: dataset path
        """
        return self.get_dataset(name).get("path")
