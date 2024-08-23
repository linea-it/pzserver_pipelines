import logging
from typing import Union

from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster


def get_executor(
    executor_key: str, executors: str
) -> Union[LocalCluster, SLURMCluster]:
    """ Returns the configuration of where the pipeline will be run

    Args:
        executor_key (str): executor key
        executors (dict): executors dict 

    Returns:
        Union[LocalCluster, SLURMCluster]: Executor object
    """

    logger = logging.getLogger()
    logger.info("Getting executor config: %s", executor_key)

    try:
        config = executors[executor_key]
    except KeyError:
        logger.warning("The executor key not found. Using minimal local config.")
        executor_key = "minimal"

    match executor_key:
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
