import os
from typing import Any

from pydantic import (
    BaseModel,
    model_validator,
)

DATASETS_DIR = os.getenv("DATASETS_DIR", "/datasets")

class Slurm(BaseModel):

  class Instance(BaseModel):
    processes: int = 1
    memory: str = "123GiB"
    queue: str = "cpu"
    job_extra_directives: list[str] = ["--propagate", "--time=2:00:00"]

  class Adapt(BaseModel):
    maximum_jobs: int = 10

  instance: Instance = Instance()
  adapt: Adapt = Adapt()

class Local(BaseModel):
  n_workers: int = 2
  threads_per_worker: int = 2
  memory_limit: str = "1GiB"


class Executor(BaseModel):

  name: str = "local"
  args: Any = {}

  @model_validator(mode='before')
  @classmethod
  def sync_args(cls, data: Any) -> Any:
    
      assert (isinstance(data, dict)), 'data is not dict'
      name = data.get("name", "local")

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

  class Dataset(BaseModel):

    class Columns(BaseModel):
      id: str = "id"

    path: str = f"{DATASETS_DIR}/mini_dataset"
    columns: Columns = Columns()


  class Specz(BaseModel):

    class Columns(BaseModel):
      ra: str = "ra"
      dec: str = "dec"
      z: str = "z"

    path: str = f"{DATASETS_DIR}/specz.parquet"
    columns: Columns = Columns()

  dataset: Dataset = Dataset()
  specz: list = [Specz()]


class Param(BaseModel):

  class Crossmatch(BaseModel):
    output_catalog_name: str = "tsm_cross_001"
    radius_arcsec: float = 1.0
    n_neighbors: int = 1

  crossmatch: Crossmatch = Crossmatch()
  duplicate_criteria: str = "closest"


class Config(BaseModel):
  output_dir: str = "./outputs"
  output_name: str = "tsm.parquet"
  executor: Executor = Executor()
  inputs: Inputs = Inputs()
  param: Param = Param()


if __name__ == "__main__":
  import yaml

  cfg = Config()

  with open('config.yml', 'w') as outfile:
    data_json = cfg.model_dump()
    print(data_json)
    yaml.dump(data_json, outfile)
