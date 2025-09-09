# combine_redshift_dedup/packages/product_handle.py

from pathlib import Path
import pandas as pd
import dask.dataframe as dd
import tables_io
from astropy.table import Table
import csv

class ProductHandle:
    """
    A unified interface to read data files of various formats into a Dask DataFrame.
    Supports Parquet, CSV, HDF5, FITS and plain text.
    """

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.extension = self.filepath.suffix.lower()
        self.delimiter = None
        self.has_header = True
        self.column_names = None

    def to_ddf(self):
        """
        Read the file and return a Dask DataFrame.

        Returns:
            dask.dataframe.DataFrame
        """
        if self.extension == ".parquet":
            return dd.read_parquet(self.filepath)

        elif self.extension == ".csv":
            self._inspect_csv()
            return dd.read_csv(
                self.filepath,
                delimiter=self.delimiter,
                header=0 if self.has_header else None,
                names=self.column_names if not self.has_header else None
            )

        elif self.extension in {".fits", ".fit"}:
            return self._read_fits_to_ddf()

        elif self.extension in {".h5", ".hdf5"}:
            return self._read_hdf5_to_ddf()

        else:
            raise ValueError(f"Unsupported file extension: {self.extension}")

    def _inspect_csv(self):
        """
        Inspect a CSV file to determine delimiter and header presence.
        """
        with open(self.filepath, "r", encoding="utf-8") as f:
            sample = f.readline()
            self.delimiter = csv.Sniffer().sniff(sample).delimiter

        # Determine if the first row is likely a header
        df_sample = pd.read_csv(self.filepath, delimiter=self.delimiter, nrows=5, header=None)
        if any(isinstance(val, str) for val in df_sample.iloc[0].values):
            self.has_header = True
        else:
            self.has_header = False
            self.column_names = [f"col_{i}" for i in range(df_sample.shape[1])]

    def _read_fits_to_ddf(self):
        """
        Read a FITS file into a Dask DataFrame via pandas.
        Note: FITS reading is not natively parallelized.
        """
        table = Table.read(self.filepath)
        df = table.to_pandas()
        return dd.from_pandas(df, npartitions=1)

    def _read_hdf5_to_ddf(self):
        """
        Read an HDF5 file into a Dask DataFrame via tables_io.
        """
        df = tables_io.read(self.filepath, tables_io.types.PD_DATAFRAME)
        return dd.from_pandas(df, npartitions=1)

def save_dataframe(df, output_path, format_):
    """Save DataFrame to disk in the requested format.

    For Parquet, writes with engine='pyarrow' to preserve Arrow-backed dtypes.
    For CSV/HDF5/FITS, uses NumPy-backed dtypes to avoid extension-type issues
    in downstream writers.

    Args:
      df: Pandas DataFrame (may contain Arrow-backed dtypes).
      output_path: Path without extension.
      format_: One of {"parquet", "csv", "hdf5", "fits"}.
    """
    ext = format_.lower()

    if ext == "parquet":
        # Keep Arrow-backed dtypes and schema (fast path).
        df.to_parquet(f"{output_path}.parquet", index=False, engine="pyarrow")
        return

    # Non-Parquet formats: convert to NumPy-backed dtypes for compatibility.
    # This avoids issues with pyarrow-backed string/boolean in third-party IO.
    df_np = df.convert_dtypes(dtype_backend="numpy_nullable")

    if ext == "csv":
        # CSV is text; NumPy-backed avoids edge cases with extension dtypes.
        df_np.to_csv(f"{output_path}.csv", index=False)
    elif ext == "hdf5":
        # Assumes `tables_io.write` expects NumPy/pandas-native dtypes.
        tables_io.write(df_np, f"{output_path}.hdf5")
    elif ext == "fits":
        # Astropy Table prefers NumPy dtypes; disable index passthrough.
        table = Table.from_pandas(df_np, preserve_index=False)
        table.write(f"{output_path}.fits", overwrite=True)
    else:
        raise ValueError(f"Unsupported output format: {format_}")

