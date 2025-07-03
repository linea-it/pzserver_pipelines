# combine_redshift_dedup/packages/executor.py

import logging
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster

def get_executor(executor_config):
    """
    Create and return a Dask cluster based on the provided executor configuration.

    Args:
        executor_config (dict): Configuration dictionary with 'name' and 'args' keys.

    Returns:
        dask.distributed.LocalCluster or dask_jobqueue.SLURMCluster:
        A configured Dask cluster.
    """
    logger = logging.getLogger("combine_redshift_dedup")
    
    executor_name = executor_config.get("name", "local")
    args = executor_config.get("args", {})

    logger.info(f"Setting up executor: {executor_name}")

    if executor_name == "local":
        # Set up a local cluster with provided arguments
        cluster = LocalCluster(**args)
    
    elif executor_name == "slurm":
        # Set up a SLURM cluster with provided instance and adapt arguments
        instance_cfg = args.get("instance", {})
        adapt_cfg = args.get("adapt", {})

        cluster = SLURMCluster(
            **instance_cfg
        )
        cluster.adapt(**adapt_cfg)
    
    else:
        logger.warning(f"Unknown executor '{executor_name}'. Falling back to minimal local cluster.")
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1
        )

    return cluster
