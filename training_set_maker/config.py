import os

from pydantic import BaseModel

DATASETS_DIR = os.getenv("DATASETS_DIR", "/datasets")

class Executor(BaseModel):

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

  name: str = "local"
  args: Slurm | Local = Local()


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
