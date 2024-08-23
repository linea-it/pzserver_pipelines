import pathlib
from dask.distributed import Client
import dask.dataframe as dd
from utils import setup_logger


class Specz:
    """Specz class"""

    def __init__(self, inputs: list[dict], client: Client) -> None:
        """Initialize specz class

        Args:
            inputs (list): input list
            client (Client): Client dask
        """

        self.logger = setup_logger("csc")
        self.client = client
        
        dataframes = []

        for _input in inputs:
            dataframes.append(self.__read(_input))

        ddf = self.client.submit(dd.concat, dataframes)
        self.dataframe = ddf.result()
        self.dataframe = self.dataframe.compute()


    def __read(self, _input):
        """ Read specz """

        filepath = pathlib.Path(_input.get("path"))
        extension = filepath.suffix

        match extension:
            case ".csv":
                dataframe = dd.read_csv(_input.get('path'))
            case ".h5" | ".hf5" | ".hdf5":
                dataframe = dd.read_hdf(_input.get('path'))
            case ".pq" | ".parq" | ".parquet":
                dataframe = dd.read_parquet(_input.get('path'))
            case ".fits" | ".fit":
                raise NotImplemented()
            case _:
                raise NotImplemented()
                    
        columns_mapping = _input.get("columns", None)
        if columns_mapping:
            map_cols = {v: k for k, v in columns_mapping.items()}
        else:
            map_cols = {"ra": "ra", "dec": "dec", "z": "z"}

        return dataframe.rename(columns=map_cols)


if __name__ == "__main__":
    print("test Specz class")
    # TODO

