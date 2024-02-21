"""_summary_ """

from parsl.config import Config


def get_parsl_config(modname) -> Config:
    mod = __import__(f"executors.{modname}", fromlist=["create_executor"])
    _func = getattr(mod, "create_executor")
    executor_instance = _func()
    return Config(executors=[executor_instance], strategy=None)