# combine_redshift_dedup/packages/executor.py

import logging
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster

def get_executor(executor_config, logs_dir=None):
    """
    Create and return a Dask cluster based on the provided executor configuration.

    Args:
        executor_config (dict): Configuration dictionary with 'name' and 'args' keys.
        logs_dir (str, optional): Directory to save SLURM job logs.

    Returns:
        dask.distributed.LocalCluster or dask_jobqueue.SLURMCluster:
        A configured Dask cluster.
    """
    logger = logging.getLogger("combine_redshift_dedup")
    
    executor_name = executor_config.get("name", "local")
    args = executor_config.get("args", {})

    logger.info(f"Setting up executor: {executor_name}")

    if executor_name == "local":
        cluster = LocalCluster(**args)
    
    elif executor_name == "slurm":
        instance_cfg = args.get("instance", {})
        scale_cfg = args.get("scale", {}) or {}
    
        if logs_dir:
            extra_directives = instance_cfg.get("job_extra_directives", [])
            extra_directives.extend([
                f"--output={logs_dir}/slurm-%j.out",
                f"--error={logs_dir}/slurm-%j.err"
            ])
            instance_cfg["job_extra_directives"] = extra_directives
    
        cluster = SLURMCluster(**instance_cfg)
    
        # Minimal logic:
        # - If user provides minimum/maximum jobs -> use adaptive scaling
        # - Else, keep legacy fixed scaling with `jobs`
        if "minimum_jobs" in scale_cfg or "maximum_jobs" in scale_cfg:
            min_jobs = scale_cfg.get("minimum_jobs", None)
            max_jobs = scale_cfg.get("maximum_jobs", None)
    
            kw = {}
            if min_jobs is not None:
                kw["minimum"] = min_jobs
            if max_jobs is not None:
                kw["maximum"] = max_jobs
    
            cluster.adapt(**kw)
            # (Opcional) Para já subir 'minimum' imediatamente:
            # if min_jobs is not None: cluster.scale(min_jobs)
        elif "jobs" in scale_cfg:
            cluster.scale(**scale_cfg)

    else:
        logger.warning(f"Unknown executor '{executor_name}'. Falling back to minimal local cluster.")
        cluster = LocalCluster(n_workers=1, threads_per_worker=1)

    return cluster