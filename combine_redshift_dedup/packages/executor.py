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
            extra_directives = list(instance_cfg.get("job_extra_directives", []))
            extra_directives.extend([
                f"--output={logs_dir}/slurm-%j.out",
                f"--error={logs_dir}/slurm-%j.err",
            ])
            instance_cfg["job_extra_directives"] = extra_directives
    
        cluster = SLURMCluster(**instance_cfg)
    
        procs_per_job = int(instance_cfg.get("processes", 1))
    
        if "minimum_jobs" in scale_cfg or "maximum_jobs" in scale_cfg:
            min_jobs = scale_cfg.get("minimum_jobs")
            max_jobs = scale_cfg.get("maximum_jobs")
    
            # Dask adaptive usa WORKERS (processos dask), não jobs
            min_workers = None if min_jobs is None else min_jobs * procs_per_job
            max_workers = None if max_jobs is None else max_jobs * procs_per_job
    
            cluster.adapt(
                minimum=min_workers,
                maximum=max_workers,
                # opcional: ajuste conforme seu perfil de tarefas
                # target_duration="2s", wait_count=3,
            )
    
            # (Opcional, recomendado) já pedir os 'minimum_jobs' imediatamente:
            if min_jobs:
                try:
                    cluster.scale(jobs=min_jobs)  # dask-jobqueue aceita 'jobs='
                except TypeError:
                    cluster.scale(min_jobs)      # fallback p/ versões antigas
        elif "jobs" in scale_cfg:
            # neste caminho o usuário quis jobs fixos
            cluster.scale(jobs=scale_cfg["jobs"])

    else:
        logger.warning(f"Unknown executor '{executor_name}'. Falling back to minimal local cluster.")
        cluster = LocalCluster(n_workers=1, threads_per_worker=1)

    return cluster