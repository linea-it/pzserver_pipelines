import logging
from typing import Union

from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster


def get_executor(executor: str) -> Union[LocalCluster, SLURMCluster]:
    """ Returns the configuration of where the pipeline will be run

    Args:
        executor (dict): executor dict 

    Returns:
        Union[LocalCluster, SLURMCluster]: Executor object
    """

    executor_name = executor.get("name", "local") # type: ignore

    logger = logging.getLogger()
    logger.info("Getting executor config: %s", executor_name)

    try:
        config = executor.get("args") # type: ignore
    except KeyError:
        logger.warning("The executor not found. Using minimal local config.")
        executor_name = "minimal"

    match executor_name:
        case "local":
            cluster = LocalCluster(**config)
        case "slurm":
            icfg = config["instance"]
            cluster = SLURMCluster(**icfg)
            cluster.adapt(**config["adapt"])
        case _:
            cluster = LocalCluster(
                n_workers=1,
                threads_per_worker=1,
            )

    return cluster
