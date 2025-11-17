import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator

MAINDIR = Path(__file__).parent
DATASETS_DIR = os.getenv("DATASETS_DIR", "/datasets")
DASK_EXECUTOR = os.getenv("DASK_EXECUTOR", "local")


class Slurm(BaseModel):

    class Instance(BaseModel):
        cores: int = 25
        processes: int = 1
        memory: str = "50GB"
        queue: str = "cpu"
        account: str = "hpc-bpglsst"
        job_extra_directives: list[str] = ["--propagate", "--time=2:00:00"]

    class Scale(BaseModel):
        minimum_jobs: int = 10
        maximum_jobs: int = 20

    instance: Instance = Instance()
    scale: Scale = Scale()


class Local(BaseModel):
    n_workers: int = 2
    threads_per_worker: int = 2
    memory_limit: str = "1GiB"


class Executor(BaseModel):

    name: str = DASK_EXECUTOR
    args: Any = {}

    @model_validator(mode="before")
    @classmethod
    def sync_args(cls, data: Any) -> Any:

        assert isinstance(data, dict), "data is not dict"
        name = data.get("name", DASK_EXECUTOR)

        match name:
            case "local":
                executor = Local(**data.get("args", {}))
            case "slurm":
                executor = Slurm(**data.get("args", {}))
            case _:
                raise ValueError(f"name '{name}' do not match")

        data["args"] = executor.model_dump()
        return data


class Inputs(BaseModel):

    class Specz(BaseModel):

        class Columns(BaseModel):
            id: str | None = None
            ra: str = "ra"
            dec: str = "dec"
            z: str = "z"
            z_flag: str | None = None
            z_err: str | None = None
            survey: str | None = None

        path: str = f"{DATASETS_DIR}/specz.parquet"
        internal_name: str = "00_specz"
        format: str = "parquet"
        columns: Columns = Columns()

    specz: list = [Specz(), Specz()]


class Param(BaseModel):
    combine_type: str = "concatenate"
    z_flag_homogenized_value_to_cut: float = 3.0
    flags_translation_file: str = str(Path(MAINDIR, "flags_translation.yaml"))


class Config(BaseModel):
    output_root_dir: str = "."
    output_dir: str = "outputs"
    output_format: str | None = "parquet"
    output_name: str = "crd"
    executor: Executor = Executor()
    inputs: Inputs = Inputs()
    param: Param = Param()


if __name__ == "__main__":
    import yaml

    cfg = Config()

    with open("config.yml", "w") as outfile:
        data_json = cfg.model_dump()
        print(data_json)
        yaml.dump(data_json, outfile)
