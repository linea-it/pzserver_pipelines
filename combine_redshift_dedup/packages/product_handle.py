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


# ============================================================
# Constants
# ============================================================
# NA tokens normalized to pd.NA after decoding from bytes/text
_NA_TOKENS = {"", "na", "nan", "null", "none", "<na>"}


# ============================================================
# Reader: Unified handle + format-specific readers
# ============================================================
class ProductHandle:
    """Unified interface to read Parquet/CSV/HDF5/FITS/plain text into a Dask DataFrame."""

    def __init__(self, filepath):
        """Initialize.

        Args:
          filepath: Path to a data file.
        """
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
        """Read the file and return a Dask DataFrame."""
        if self.base_ext == ".parquet":
            return dd.read_parquet(self.filepath)

        elif self.base_ext in {".csv", ".tsv", ".txt"}:
            self._inspect_csv()

            ddf_raw = dd.read_csv(
                self.filepath,
                delimiter=self.delimiter,
                header=0 if self.has_header else None,
                names=self.column_names if not self.has_header else None,
                dtype="object",
                na_values=["", "NA", "NaN", "null", "None"],
                keep_default_na=True,
                comment="#",
                encoding_errors="replace",
                compression="infer",
            )

            sample_rows = 200_000
            pdf_sample = pd.read_csv(
                self.filepath,
                delimiter=self.delimiter,
                header=0 if self.has_header else None,
                names=self.column_names if not self.has_header else None,
                dtype="object",
                na_values=["", "NA", "NaN", "null", "None"],
                keep_default_na=True,
                comment="#",
                low_memory=False,
                nrows=sample_rows,
            )

            schema = _infer_csv_schema(pdf_sample, frac_threshold=0.995)
            meta = _meta_from_schema(ddf_raw.columns, schema)

            ddf = ddf_raw.map_partitions(_apply_csv_schema, schema, meta=meta)
            return ddf

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
        """Detect delimiter/header using robust heuristics."""
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
        """Read FITS to Dask DataFrame with nullable semantics preserved."""
        table = Table.read(self.filepath)

        # 1) Pandas DF with NAs preserved (including boolean with pd.NA where masked)
        df = _astropy_table_to_pandas_nullable(table)

        # 2) Decode bytes/strings, normalize NA tokens, and promote text-boolean to BooleanDtype
        df = _normalize_after_read(df, source="fits")

        # 3) Dynamic check: does this Dask preserve pandas BooleanDtype with NA?
        def _dask_preserves_nullable_boolean() -> bool:
            try:
                probe = pd.DataFrame({"_p": pd.Series([True, pd.NA, False], dtype="boolean")})
                res = dd.from_pandas(probe, npartitions=1).compute()
                return (str(res["_p"].dtype) == "boolean") and bool(res["_p"].isna().iloc[1])
            except Exception:
                return False

        preserves = _dask_preserves_nullable_boolean()

        if not preserves:
            # Stable fallback: re-encode booleans as Int8 (1/0/<NA>)
            for c in df.columns:
                s = df[c]
                if pd.api.types.is_bool_dtype(s) and str(s.dtype) == "boolean":
                    df[c] = s.map({True: 1, False: 0}).astype("Int8")

        # 4) Lift to Dask
        ddf = dd.from_pandas(df, npartitions=1)

        return ddf

    # -----------------------------
    # HDF5 reader (+ postprocess)
    # -----------------------------
    def _read_hdf5_to_ddf(self):
        """Read HDF5 to Dask DataFrame via tables_io with dtype normalization."""
        df = tables_io.read(self.filepath, tables_io.types.PD_DATAFRAME)
        df = _normalize_after_read(df, source="hdf5")
        return dd.from_pandas(df, npartitions=1)


# ============================================================
# Reader helpers (CSV + decode/normalize + astropy conversions)
# ============================================================
def _string_bool_to_boolean(series: pd.Series) -> pd.Series:
    """Convert 'true'/'false'/NA to pandas BooleanDtype without evaluating NA."""
    s = series.astype("string")

    def _parse(x):
        if pd.isna(x):
            return pd.NA
        t = str(x).strip().lower()
        if t == "true":
            return True
        if t == "false":
            return False
        return pd.NA

    return s.map(_parse).astype("boolean")


def _normalize_na_tokens_frame(pdf: pd.DataFrame) -> pd.DataFrame:
    """Normalize common NA tokens to pd.NA on string/object columns."""
    pdf = pdf.copy()
    for c in pdf.columns:
        s = pdf[c]
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            pdf[c] = s.map(
                lambda x: pd.NA if (x is None or (isinstance(x, str) and x.strip().lower() in _NA_TOKENS))
                else x
            )
    return pdf


def _csv_postprocess(pdf: pd.DataFrame) -> pd.DataFrame:
    """Normalize NA tokens and stabilize dtypes for CSV-loaded frames."""
    pdf = _normalize_na_tokens_frame(pdf)
    pdf = pdf.convert_dtypes(dtype_backend="numpy_nullable", convert_boolean=False)
    return pdf


def _infer_csv_schema(pdf: pd.DataFrame, *, frac_threshold: float = 0.995) -> dict:
    """Infer per-column target dtypes for CSV (boolean/Int64/Float64/string)."""
    schema = {}
    for c in pdf.columns:
        s = pdf[c]

        # Only attempt inference on text/object-like columns
        if not (s.dtype == "object" or pd.api.types.is_string_dtype(s)):
            schema[c] = "string"
            continue

        # Normalize NA tokens and ensure string dtype for checks
        s2 = s.map(lambda x: pd.NA if (x is None or (isinstance(x, str) and x.strip().lower() in _NA_TOKENS)) else x)
        s2 = s2.astype("string").map(lambda x: x.strip() if isinstance(x, str) else x)

        # Strict textual boolean detection
        s2_lower = s2.str.lower()
        non_na = s2_lower.dropna()
        if non_na.empty:
            schema[c] = "string"
            continue
        if set(non_na.unique()) <= {"true", "false"}:
            schema[c] = "boolean"
            continue

        # Numeric inference
        num = pd.to_numeric(s2, errors="coerce")
        nn_total = int(num.notna().sum())
        if nn_total == 0:
            schema[c] = "string"
            continue

        frac_numeric = nn_total / max(1, int(s2.notna().sum()))
        if frac_numeric >= frac_threshold:
            non_na_num = num.dropna()
            has_frac = (non_na_num % 1 != 0).any()
            schema[c] = "Float64" if has_frac else "Int64"
        else:
            schema[c] = "string"
    return schema


def _apply_csv_schema(pdf: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Apply inferred CSV schema column-wise without coercing 0/1 to booleans."""
    out = {}
    for c in pdf.columns:
        s = pdf[c]
        # Only transform string/object-like columns; keep others as-is
        if s.dtype != "object" and not pd.api.types.is_string_dtype(s):
            out[c] = s
            continue

        # Normalize NA tokens and trim
        s2 = s.map(lambda x: pd.NA if (x is None or (isinstance(x, str) and x.strip().lower() in _NA_TOKENS))
                   else (x.strip() if isinstance(x, str) else x))
        target = schema.get(c, "string")

        if target == "Int64":
            out[c] = pd.to_numeric(s2, errors="coerce").astype("Int64")
        elif target == "Float64":
            out[c] = pd.to_numeric(s2, errors="coerce").astype("Float64")
        elif target == "boolean":
            out[c] = _string_bool_to_boolean(s2)
        else:
            out[c] = s2.astype("string")
    return pd.DataFrame(out, columns=pdf.columns)


def _meta_from_schema(columns, schema: dict) -> pd.DataFrame:
    """Build an empty DataFrame matching the given schema for Dask meta."""
    data = {}
    for c in columns:
        t = schema.get(c, "string")
        if t == "Int64":
            data[c] = pd.Series([], dtype="Int64")
        elif t == "Float64":
            data[c] = pd.Series([], dtype="Float64")
        elif t == "boolean":
            data[c] = pd.Series([], dtype="boolean")
        else:
            data[c] = pd.Series([], dtype="string")
    return pd.DataFrame(data)


def _is_fixed_bytes(s: pd.Series) -> bool:
    """Return True if dtype is fixed-width bytes (NumPy kind 'S')."""
    return getattr(s.dtype, "kind", "") == "S"


def _is_object_bytes_series(s: pd.Series, sample_size: int = 256) -> bool:
    """Return True if dtype is object and most non-null values are bytes-like."""
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
    """Normalize common NA tokens (case-insensitive) to pd.NA on a string Series."""
    if not pd.api.types.is_string_dtype(s):
        s = s.astype("string")
    return s.map(lambda x: pd.NA if x is None or str(x).strip().lower() in _NA_TOKENS else x)


def _decode_byteslike_to_string(series: pd.Series, *, encoding: str, errors: str = "replace") -> pd.Series:
    """Decode fixed-width or object-bytes to pandas StringDtype and normalize NA tokens."""
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
    """Return True if non-NA values are a subset of {'true','false'}."""
    if series.empty:
        return False
    sample = series.dropna()
    if sample.empty:
        return False
    vals = set(sample.astype(str).str.strip().str.lower().tolist())
    return vals.issubset({"true", "false"})


def _bytes_bool_to_boolean(series: pd.Series) -> pd.Series:
    """Convert textual booleans 'true'/'false'/NA to pandas BooleanDtype safely."""
    s = series.astype("string")
    s_norm = s.str.strip().str.lower()

    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out = out.mask(s_norm == "true", True)
    out = out.mask(s_norm == "false", False)
    return out


def _normalize_after_read(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Normalize decoded bytes/strings, NA tokens, and strict text booleans after read."""
    df = df.copy()
    text_encoding = "ascii" if source == "fits" else "utf-8"

    for c in df.columns:
        s = df[c]

        # Case 1: bytes-like → decode
        was_bytes = _is_fixed_bytes(s) or _is_object_bytes_series(s)
        if was_bytes:
            decoded = _decode_byteslike_to_string(s, encoding=text_encoding, errors="replace")
            if _looks_like_bool_text(decoded):
                df[c] = _bytes_bool_to_boolean(decoded)
            else:
                df[c] = decoded
            continue

        # Case 2: string-like → normalize NA tokens and strict boolean text
        if pd.api.types.is_string_dtype(s):
            s_norm = _normalize_na_tokens_to_pdna(s.astype("string"))
            if _looks_like_bool_text(s_norm):
                df[c] = _string_bool_to_boolean(s_norm)
            else:
                df[c] = s_norm
            continue

        # Other dtypes: keep

    try:
        df = df.convert_dtypes(dtype_backend="numpy_nullable", convert_boolean=False)
    except Exception:
        pass
    return df


def _astropy_table_to_pandas_nullable(table: Table) -> pd.DataFrame:
    """Convert Astropy Table to pandas with nullable dtypes and masked values as NA."""
    out = {}
    for name in table.colnames:
        col = table[name]
        has_mask = hasattr(col, "mask")

        # FITS LOGICAL → pandas nullable boolean with NA preserved
        if getattr(col.dtype, "kind", None) == "b":
            s = pd.Series(col.astype(object))
            if has_mask:
                s.loc[np.asarray(col.mask, dtype=bool)] = pd.NA
            out[name] = s.astype("boolean")
            continue

        if has_mask:
            s = pd.Series(col.astype(object))
            s.loc[np.asarray(col.mask, dtype=bool)] = pd.NA

            # Try numeric coercion
            s_try = pd.to_numeric(s, errors="coerce")
            nn = s.notna().sum()
            if nn > 0:
                frac_num = s_try.notna().sum() / nn
                if frac_num >= 0.995:
                    non_na = s_try.dropna()
                    has_frac = (non_na % 1 != 0).any()
                    out[name] = (s_try.astype("Float64") if has_frac else s_try.astype("Int64"))
                    continue
            out[name] = s
        else:
            out[name] = pd.Series(col)

    return pd.DataFrame(out)


# ============================================================
# Writer helpers (fixed-bytes encoders + sanitizers)
# ============================================================
def _to_fixed_bytes(series: pd.Series, safety_cap: int = 1 << 16, *, encoding: str = "utf-8", errors: str = "replace") -> pd.Series:
    """Convert string-like series to fixed-width NumPy bytes (|S{N})."""
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
    """Encode BooleanDtype/boolean as fixed-width bytes 'true'/'false'/''."""
    if not (pd.api.types.is_bool_dtype(series) or getattr(series.dtype, "name", "") == "BooleanDtype"):
        return series

    s = series.astype("boolean")
    s_txt = s.map(lambda x: "true" if x is True else "false" if x is False else "")
    return _to_fixed_bytes(s_txt, encoding=encoding, errors="replace")


def _sanitize_for_hdf5(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize DataFrame for HDF5 writing (strings/booleans to bytes, stable ints)."""
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
                        df[c] = s.astype("int16")   # avoid 8-bit pitfalls
                    elif name.endswith("16"):
                        df[c] = s.astype("int16")
                    elif name.endswith("32"):
                        df[c] = s.astype("int32")
                    else:
                        df[c] = s.astype("int64")
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

    bad = [c for c in df.columns if df[c].dtype == "object"]
    if bad:
        raise TypeError(f"HDF5 sanitizer: unsupported object dtypes after conversion: {bad}")

    return df


def _sanitize_for_fits(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize DataFrame for FITS writing (strings/booleans to ASCII bytes; promote 8-bit ints)."""
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
                    name = s.dtype.name.lower()
                    if name.endswith("8"):
                        df[c] = s.astype("int16")  # promote 8-bit -> 16-bit (avoid FITS LOGICAL)
                    elif name.endswith("16"):
                        df[c] = s.astype("int16")
                    elif name.endswith("32"):
                        df[c] = s.astype("int32")
                    else:
                        df[c] = s.astype("int64")
            else:
                npname = str(s.dtype).lower()
                if npname.endswith("8"):
                    df[c] = s.astype("int16")      # promote 8-bit -> 16-bit
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

    bad = [c for c in df.columns if df[c].dtype == "object"]
    if bad:
        raise TypeError(f"FITS sanitizer: unsupported object dtypes after conversion: {bad}")

    return df


# ============================================================
# Save API (Parquet/CSV/HDF5/FITS) + placement of writer helpers
# ============================================================
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
