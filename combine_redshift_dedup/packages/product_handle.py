# combine_redshift_dedup/packages/product_handle.py

from pathlib import Path
import io
import csv
import json
import numpy as np
import pandas as pd
import dask.dataframe as dd
import tables_io
from astropy.table import Table

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------
# NA tokens normalized to pd.NA after decoding from bytes/text
# ---------------------------------------------------------------------
_NA_TOKENS = {"", "na", "nan", "null", "none"}


class ProductHandle:
    """
    Unified interface to read Parquet/CSV/HDF5/FITS/plain text into a Dask DataFrame.
    """

    def __init__(self, filepath):
        self.filepath = Path(filepath)

        # Handle compound extensions like ".csv.gz"
        full_suffix = "".join(self.filepath.suffixes).lower()
        if full_suffix.endswith((".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zst")):
            self.base_ext = ".csv"
        elif full_suffix.endswith((".tsv.gz", ".tsv.bz2", ".tsv.xz", ".tsv.zst")):
            self.base_ext = ".tsv"
        elif full_suffix.endswith((".txt.gz", ".txt.bz2", ".txt.xz", ".txt.zst")):
            self.base_ext = ".txt"
        else:
            self.base_ext = self.filepath.suffix.lower()

        self.delimiter = None
        self.has_header = True
        self.column_names = None

    # -----------------------------
    # Public API
    # -----------------------------
    def to_ddf(self):
        """
        Read the file and return a Dask DataFrame.
        """
        if self.base_ext == ".parquet":
            return dd.read_parquet(self.filepath)

        elif self.base_ext in {".csv", ".tsv", ".txt"}:
            self._inspect_csv()
            return dd.read_csv(
                self.filepath,
                delimiter=self.delimiter,
                header=0 if self.has_header else None,
                names=self.column_names if not self.has_header else None,
                na_values=["", "NA", "NaN", "null", "None"],
                keep_default_na=True,
                comment="#",
                assume_missing=True,              # safer for mixed int/float columns
                dtype_backend="numpy_nullable",
                encoding_errors="replace",
                compression="infer",
            )

        elif self.base_ext in {".fits", ".fit"}:
            return self._read_fits_to_ddf()

        elif self.base_ext in {".h5", ".hdf5"}:
            return self._read_hdf5_to_ddf()

        else:
            raise ValueError(f"Unsupported file extension: {self.filepath}")

    # -----------------------------
    # CSV helpers
    # -----------------------------
    def _inspect_csv(self):
        """
        Robust delimiter + header detection:
        - Try sniffing across a larger sample.
        - Fallback to a set of common delimiters.
        - Header heuristic: first row is mostly non-numeric tokens.
        """
        with open(self.filepath, "rb") as fb:
            raw = fb.read(128 * 1024)
        sample = raw.decode("utf-8", errors="replace")

        # Delimiter
        try:
            sniffed = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
            self.delimiter = sniffed.delimiter
        except Exception:
            if "\t" in sample and "," not in sample:
                self.delimiter = "\t"
            elif ";" in sample and "," not in sample:
                self.delimiter = ";"
            else:
                self.delimiter = ","

        # Header?
        try:
            df_head = pd.read_csv(io.StringIO(sample), delimiter=self.delimiter, nrows=5, header=None)
        except Exception:
            df_head = pd.DataFrame()

        def _looks_like_header(df0: pd.DataFrame) -> bool:
            if df0.empty:
                return False
            row0 = df0.iloc[0].astype(str).tolist()

            def is_number(s: str) -> bool:
                try:
                    float(s)
                    return True
                except Exception:
                    return False

            non_numeric = sum(1 for v in row0 if not is_number(v))
            return non_numeric >= max(1, len(row0) // 2)

        if not df_head.empty and _looks_like_header(df_head):
            self.has_header = True
        else:
            self.has_header = False
            self.column_names = [f"col_{i}" for i in range(df_head.shape[1] if not df_head.empty else 0)]

    # -----------------------------
    # FITS reader (+ postprocess)
    # -----------------------------
    def _read_fits_to_ddf(self):
        """
        Read a FITS file into a Dask DataFrame via pandas.
        Post-process:
          - decode bytes -> pandas StringDtype (ASCII; NA tokens -> pd.NA)
          - auto-restore BooleanDtype ONLY for 'true'/'false'/NA text columns
        """
        table = Table.read(self.filepath)
        df = table.to_pandas()
        df = _normalize_after_read(df, source="fits")
        return dd.from_pandas(df, npartitions=1)

    # -----------------------------
    # HDF5 reader (+ postprocess)
    # -----------------------------
    def _read_hdf5_to_ddf(self):
        """
        Read an HDF5 file into a Dask DataFrame via tables_io.
        Post-process:
          - decode bytes -> pandas StringDtype (UTF-8; NA tokens -> pd.NA)
          - auto-restore BooleanDtype ONLY for 'true'/'false'/NA text columns
        """
        df = tables_io.read(self.filepath, tables_io.types.PD_DATAFRAME)
        df = _normalize_after_read(df, source="hdf5")
        return dd.from_pandas(df, npartitions=1)


# =============================
# Write-time helpers (no object)
# =============================
def _to_fixed_bytes(series: pd.Series, safety_cap: int = 1 << 16, *, encoding: str = "utf-8", errors: str = "replace") -> pd.Series:
    """
    Convert string-like series to fixed-width NumPy bytes (|S{N}) with given encoding.
    NA -> "" (then encoded).
    """
    def _to_text(x):
        if isinstance(x, (dict, list, tuple, set)):
            return json.dumps(x, ensure_ascii=False)
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        return str(x)

    s = series
    if pd.api.types.is_object_dtype(s):
        s = s.map(_to_text)
    else:
        s = s.astype("string").fillna("").astype(str)

    if len(s):
        lens = s.apply(lambda v: len(v.encode(encoding, errors=errors)) if v else 0)
        max_len = int(lens.max() or 0)
    else:
        max_len = 0

    max_len = max(1, min(safety_cap, max_len))
    dtype_s = f"|S{max_len}"

    arr = np.fromiter((v.encode(encoding, errors=errors) for v in s), dtype=dtype_s, count=len(s))
    return pd.Series(arr, index=series.index)


def _bool_to_fixed_bytes(series: pd.Series, *, encoding: str) -> pd.Series:
    """
    STRICT boolean mapping:
    Only accept boolean dtype (NumPy bool_ or pandas BooleanDtype).
    Convert True/False/pd.NA -> 'true'/'false'/'' and then to fixed-width bytes.
    """
    if not (pd.api.types.is_bool_dtype(series) or getattr(series.dtype, "name", "") == "BooleanDtype"):
        # Do NOT coerce integer 0/1 to boolean implicitly.
        return series

    s = series.astype("boolean")
    s_txt = s.map(lambda x: "true" if x is True else "false" if x is False else "")
    return _to_fixed_bytes(s_txt, encoding=encoding, errors="replace")


# =============================
# Write-time sanitizers
# =============================
def _sanitize_for_hdf5(df: pd.DataFrame) -> pd.DataFrame:
    """
    HDF5 sanitizer:
    - Strings/objects -> fixed-width UTF-8 bytes
    - Booleans -> fixed-width UTF-8 bytes 'true'/'false'/'' (strict boolean only)
    - Nullable ints -> float64 only when NA present; otherwise keep native ints
    - Categories -> int32; datetimes -> datetime64[ns]
    """
    df = df.copy()

    # Drop all-missing columns
    all_missing = [c for c in df.columns if df[c].isna().all()]
    if all_missing:
        df.drop(columns=all_missing, inplace=True)

    # Avoid accidental boolean inference
    df = df.convert_dtypes(dtype_backend="numpy_nullable", convert_boolean=False)

    for c in df.columns:
        s = df[c]

        # Datetimes
        if pd.api.types.is_datetime64_any_dtype(s):
            df[c] = pd.to_datetime(s, errors="coerce").astype("datetime64[ns]")
            continue

        # Integers
        if pd.api.types.is_integer_dtype(s):
            if pd.api.types.is_extension_array_dtype(s):  # Int64/Int32/... with <NA>
                if s.isna().any():
                    df[c] = s.astype("float64")
                else:
                    name = s.dtype.name.lower()
                    if name.endswith("8"):
                        df[c] = s.astype("int16")   # keep numeric; avoid 8-bit pitfalls
                    elif name.endswith("16"):
                        df[c] = s.astype("int16")
                    elif name.endswith("32"):
                        df[c] = s.astype("int32")
                    else:
                        df[c] = s.astype("int64")
            # Numpy plain ints are fine; but promote 8-bit just in case
            elif str(s.dtype).lower().endswith("8"):
                df[c] = s.astype("int16")
            continue

        # Booleans (strict)
        if pd.api.types.is_bool_dtype(s) or getattr(s.dtype, "name", "") == "BooleanDtype":
            df[c] = _bool_to_fixed_bytes(s, encoding="utf-8")
            continue

        # Categories
        if pd.api.types.is_categorical_dtype(s):
            df[c] = s.cat.codes.astype("int32")
            continue

        # Floats ok
        if pd.api.types.is_float_dtype(s):
            continue

        # Strings/objects/arrow-backed strings -> bytes
        dtype_name = getattr(s.dtype, "name", str(s.dtype))
        if (
            pd.api.types.is_string_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or ("string" in dtype_name)
        ):
            df[c] = _to_fixed_bytes(s, encoding="utf-8", errors="replace")
            continue

        if df[c].dtype == "object":
            df[c] = _to_fixed_bytes(s, encoding="utf-8", errors="replace")

    # No object dtype left
    bad = [c for c in df.columns if df[c].dtype == "object"]
    if bad:
        raise TypeError(f"HDF5 sanitizer: unsupported object dtypes after conversion: {bad}")

    return df


def _sanitize_for_fits(df: pd.DataFrame) -> pd.DataFrame:
    """
    FITS sanitizer:
    - Strings/objects -> fixed-width ASCII bytes
    - Booleans -> fixed-width ASCII bytes 'true'/'false'/'' (strict boolean only)
    - Nullable ints -> float64 when NA present; otherwise keep native ints
      BUT: promote any 8-bit integer (signed/unsigned) to int16 to avoid LOGICAL
    - Categories -> int32; datetimes -> datetime64[ns]
    """
    df = df.copy()

    # Drop all-missing columns
    all_missing = [c for c in df.columns if df[c].isna().all()]
    if all_missing:
        df.drop(columns=all_missing, inplace=True)

    # Avoid accidental boolean inference
    df = df.convert_dtypes(dtype_backend="numpy_nullable", convert_boolean=False)

    for c in df.columns:
        s = df[c]

        # Datetimes
        if pd.api.types.is_datetime64_any_dtype(s):
            df[c] = pd.to_datetime(s, errors="coerce").astype("datetime64[ns]")
            continue

        # Integers
        if pd.api.types.is_integer_dtype(s):
            if pd.api.types.is_extension_array_dtype(s):
                if s.isna().any():
                    df[c] = s.astype("float64")
                else:
                    name = s.dtype.name.lower()    # "int8", "int16", "int32", "int64", ...
                    if name.endswith("8"):
                        df[c] = s.astype("int16")  # promote 8-bit -> 16-bit (avoid FITS LOGICAL)
                    elif name.endswith("16"):
                        df[c] = s.astype("int16")
                    elif name.endswith("32"):
                        df[c] = s.astype("int32")
                    else:
                        df[c] = s.astype("int64")
            else:
                # NumPy ints
                npname = str(s.dtype).lower()
                if npname.endswith("8"):
                    df[c] = s.astype("int16")      # promote 8-bit -> 16-bit
                # else: keep as is
            continue

        # Booleans (strict)
        if pd.api.types.is_bool_dtype(s) or getattr(s.dtype, "name", "") == "BooleanDtype":
            df[c] = _bool_to_fixed_bytes(s, encoding="ascii")
            continue

        # Categories
        if pd.api.types.is_categorical_dtype(s):
            df[c] = s.cat.codes.astype("int32")
            continue

        # Floats ok
        if pd.api.types.is_float_dtype(s):
            continue

        # Strings/objects/pyarrow strings -> ASCII bytes
        dtype_name = getattr(s.dtype, "name", str(s.dtype))
        if (
            pd.api.types.is_string_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or ("string" in dtype_name)
        ):
            df[c] = _to_fixed_bytes(s, encoding="ascii", errors="replace")
            continue

        if df[c].dtype == "object":
            df[c] = _to_fixed_bytes(s, encoding="ascii", errors="replace")

    # No object dtype left
    bad = [c for c in df.columns if df[c].dtype == "object"]
    if bad:
        raise TypeError(f"FITS sanitizer: unsupported object dtypes after conversion: {bad}")

    return df


# =============================
# Read-time normalization
# =============================
def _is_fixed_bytes(s: pd.Series) -> bool:
    """True if dtype is fixed-width bytes (NumPy kind 'S')."""
    return getattr(s.dtype, "kind", "") == "S"

def _is_object_bytes_series(s: pd.Series, sample_size: int = 256) -> bool:
    """
    True if dtype is object and the majority of non-null values are bytes-like.
    """
    if s.dtype != "object":
        return False
    non_null = s.dropna()
    if non_null.empty:
        return True
    sample = non_null.head(sample_size)
    n = len(sample)
    cnt = sum(isinstance(v, (bytes, bytearray, np.bytes_)) for v in sample)
    return cnt / max(1, n) >= 0.8

def _normalize_na_tokens_to_pdna(s: pd.Series) -> pd.Series:
    """
    Map known NA tokens (case-insensitive) to pd.NA on a string Series.
    """
    if not pd.api.types.is_string_dtype(s):
        s = s.astype("string")
    return s.map(lambda x: pd.NA if x is None or str(x).strip().lower() in _NA_TOKENS else x)

def _decode_byteslike_to_string(series: pd.Series, *, encoding: str, errors: str = "replace") -> pd.Series:
    """
    Decode fixed-width bytes (|S{N}) or object-bytes to pandas StringDtype.
    Applies .strip() and normalizes NA tokens -> pd.NA.
    """
    if _is_fixed_bytes(series):
        it = series.astype("O")
        as_str = it.map(
            lambda b: b.decode(encoding, errors=errors)
            if isinstance(b, (bytes, bytearray))
            else ("" if b is None else str(b))
        )
    elif _is_object_bytes_series(series):
        as_str = series.map(
            lambda b: (
                b.decode(encoding, errors=errors)
                if isinstance(b, (bytes, bytearray, np.bytes_))
                else ("" if b is None else str(b))
            )
        )
    else:
        return series

    as_str = as_str.map(lambda x: x.strip() if isinstance(x, str) else x).astype("string")
    as_str = _normalize_na_tokens_to_pdna(as_str)
    return as_str

def _looks_like_bool_text(series: pd.Series) -> bool:
    """
    Strict boolean-literal check after decoding/NA-normalization:
    True iff non-NA values are subset of {'true','false'} (case-insensitive).
    """
    if series.empty:
        return False
    sample = series.dropna()
    if sample.empty:
        return True
    vals = set(sample.astype(str).str.strip().str.lower().tolist())
    return vals.issubset({"true", "false"})

def _bytes_bool_to_boolean(series: pd.Series) -> pd.Series:
    """
    Convert textual booleans 'true'/'false' + NA to pandas BooleanDtype.
    """
    s = series.astype("string").str.strip().str.lower()
    return s.map(lambda x: True if x == "true" else False if x == "false" else pd.NA).astype("boolean")

def _normalize_after_read(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """
    Post-read normalization:
      - Decode bytes-like columns (UTF-8 for HDF5, ASCII for FITS) and map NA tokens -> pd.NA
      - Auto-restore booleans ONLY when the decoded text is strictly {'true','false', NA}
      - Keep dtype_backend stable with numpy_nullable
    """
    df = df.copy()
    text_encoding = "ascii" if source == "fits" else "utf-8"

    for c in df.columns:
        s = df[c]
        was_bytes = _is_fixed_bytes(s) or _is_object_bytes_series(s)
        if was_bytes:
            decoded = _decode_byteslike_to_string(s, encoding=text_encoding, errors="replace")
            if _looks_like_bool_text(decoded):
                df[c] = _bytes_bool_to_boolean(decoded)
            else:
                df[c] = decoded

    try:
        df = df.convert_dtypes(dtype_backend="numpy_nullable")
    except Exception:
        pass
    return df


# =============================
# Save API
# =============================
def save_dataframe(df, output_path, format_):
    """Save DataFrame to disk in the requested format.

    Parquet: write via PyArrow directly.
    CSV/HDF5/FITS: sanitize dtypes for compatibility.
    """
    ext = format_.lower()

    if ext == "parquet":
        try:
            df2 = df.reset_index(drop=True)
        except Exception:
            df2 = df
        table = pa.Table.from_pandas(df2, preserve_index=False)
        pq.write_table(table, f"{output_path}.parquet")
        return

    if ext == "hdf5":
        # Strict: integers remain integers; booleans only if dtype is boolean
        df_h5 = _sanitize_for_hdf5(df)
        tables_io.write(df_h5, f"{output_path}.hdf5")
        return

    # CSV/FITS start from numpy_nullable to stabilize null semantics
    df_np = df.convert_dtypes(dtype_backend="numpy_nullable", convert_boolean=False)

    if ext == "csv":
        df_np.to_csv(f"{output_path}.csv", index=False)
        return

    if ext == "fits":
        # Promote 8-bit ints to int16; write strings/booleans as bytes/text safely
        df_fits = _sanitize_for_fits(df_np).reset_index(drop=True)
        table = Table.from_pandas(df_fits)
        table.write(f"{output_path}.fits", overwrite=True)
        return

    raise ValueError(f"Unsupported output format: {format_}")