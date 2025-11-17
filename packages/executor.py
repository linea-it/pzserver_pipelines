# combine_redshift_dedup/packages/executor.py

import logging
from typing import Dict, Optional, Any

from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster

from utils import get_phase_logger

LOGGER_NAME = "crc.executor"


def _get_logger() -> logging.LoggerAdapter:
    """Return a phase-aware logger ('crc.executor' with phase='executor')."""
    base = logging.getLogger(LOGGER_NAME)
    base.propagate = True
    return get_phase_logger("executor", base)


def get_executor(executor_config: Dict[str, Any], logs_dir: Optional[str] = None):
    """Create and return a Dask cluster from config.

    Behavior:
      - Local: start a LocalCluster(**args).
      - SLURM: start SLURMCluster with n_workers = minimum_jobs * processes.
        Each job will use the provided cores, processes, and memory.
        Optionally enable adapt() with maximum_jobs.

    Args:
        executor_config: Dictionary with:
            - name: "local" or "slurm".
            - args:
                For "local": passed to LocalCluster(**args).
                For "slurm":
                  - instance: kwargs for SLURMCluster(**instance).
                  - scale:
                      * minimum_jobs (int)
                      * maximum_jobs (int)
        logs_dir: Directory to store SLURM job logs (if provided).

    Returns:
        LocalCluster | SLURMCluster: Configured Dask cluster.
    """
    logger = _get_logger()

    executor_name = executor_config.get("name", "local")
    args = executor_config.get("args", {}) or {}

    logger.info("Setting up executor: %s", executor_name)

    # -------------------------
    # Local cluster
    # -------------------------
    if executor_name == "local":
        cluster = LocalCluster(**args)
        logger.info("LocalCluster started with args=%s", args)
        return cluster

    # -------------------------
    # SLURM cluster
    # -------------------------
    if executor_name == "slurm":
        instance_cfg = dict(args.get("instance", {}) or {})
        scale_cfg = dict(args.get("scale", {}) or {})

        # Add job log directives if requested
        if logs_dir:
            extra_directives = list(instance_cfg.get("job_extra_directives", []))
            extra_directives.extend(
                [
                    f"--output={logs_dir}/slurm-%j.out",
                    f"--error={logs_dir}/slurm-%j.err",
                ]
            )
            instance_cfg["job_extra_directives"] = extra_directives

        processes = int(instance_cfg.get("processes", 1))
        cores = instance_cfg.get("cores")
        memory = instance_cfg.get("memory")
        queue = instance_cfg.get("queue")
        account = instance_cfg.get("account")

        min_jobs = int(scale_cfg.get("minimum_jobs", 0))
        max_jobs = int(scale_cfg.get("maximum_jobs", 0) or 0)

        # n_workers in SLURMCluster = total worker processes
        n_workers_init = min_jobs * processes

        logger.info(
            "SLURM job template: cores=%s, processes=%s, memory=%s, queue=%s, account=%s",
            cores,
            processes,
            memory,
            queue,
            account,
        )
        logger.info(
            "Initial submit: minimum_jobs=%d → n_workers=%d (worker processes).",
            min_jobs,
            n_workers_init,
        )

        # Submit jobs immediately on construction
        cluster = SLURMCluster(n_workers=n_workers_init, **instance_cfg)
        logger.info("SLURMCluster started with instance args=%s", instance_cfg)

        # Optional: cap number of jobs adaptively
        if max_jobs > 0:
            cluster.adapt(minimum_jobs=min_jobs, maximum_jobs=max_jobs)
            logger.info("Adaptive ceiling enabled: maximum_jobs=%d", max_jobs)

        return cluster

    # -------------------------
    # Unknown -> minimal local
    # -------------------------
    logger.warning(
        "Unknown executor '%s'. Falling back to a minimal LocalCluster.", executor_name
    )
    return LocalCluster(n_workers=1, threads_per_worker=1)
