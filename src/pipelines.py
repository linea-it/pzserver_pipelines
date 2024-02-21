import os
import logging
import pathlib
import yaml


class MetaPipelines(object):
    def __init__(self) -> None:
        self.logger = logging.getLogger()

        try:
            pipelines_dir = os.environ["PIPELINES_DIR"]
        except KeyError:
            self.logger.exception("PIPELINES_DIR not defined.")

        with open(
            pathlib.Path(pipelines_dir, "pipelines.yml"), encoding="utf-8"
        ) as _file:
            self.__pipelines = yaml.safe_load(_file)

    def get_pipeline(self, name: str) -> dict:
        """get pipeline by name

        Args:
            name (str): pipeline name

        Returns:
            dict: pipeline info
        """
        return self.__pipelines.get(name)

    def get_pipeline_path(self, name: str) -> pathlib.Path:
        """get pipeline path by name

        Args:
            name (str): pipeline name

        Returns:
            pathlib.Path: pipeline path
        """
        return self.get_pipeline(name).get("dir")
