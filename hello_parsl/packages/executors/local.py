from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider


def create_executor():
    """
    Creates an instance of the Parsl configuration

    Returns:
        config: Parsl config instance.
    """

    return HighThroughputExecutor(
        label="local",
        worker_debug=False,
        max_workers=2,
        provider=LocalProvider(
            min_blocks=1,
            init_blocks=1,
            max_blocks=1,
            parallelism=1,
        ),
    )
