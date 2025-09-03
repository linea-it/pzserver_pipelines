# specz.py
from __future__ import annotations

"""Combine Redshift Catalogs – catalog preparation.

This module loads a raw spectroscopic catalog, validates/normalizes schema,
derives standardized fields (IDs, homogenized flags, DP1 flags), optionally
imports HATS + generates margin cache, and writes a prepared Parquet artifact.

Public API:
    - prepare_catalog(entry, translation_config, logs_dir, temp_dir, combine_mode)
"""

# -----------------------
# Standard library
# -----------------------
import ast
import builtins
import difflib
import glob
import json
import logging
import math
import os
import pathlib
import re
import shutil
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

# -----------------------
# Third-party
# -----------------------
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import get_client

dask.config.set({"dataframe.shuffle.method": "tasks"})
os.environ.setdefault("DASK_DISTRIBUTED__SHUFFLE__METHOD", "tasks")

# -----------------------
# Project (lsdb/hats/CRC)
# -----------------------
import hats
import lsdb
from combine_redshift_dedup.packages.product_handle import ProductHandle
from hats_import.catalog.file_readers import ParquetReader
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from hats_import.pipeline import ImportArguments, pipeline_with_client

if TYPE_CHECKING:
    # Avoid heavy imports at runtime just for typing
    from dask.distributed import Client  # noqa: F401

# -----------------------
# Arrow-backed dtypes (pandas >= 2.x)
# -----------------------
import pyarrow as pa

USE_ARROW_TYPES = True

if USE_ARROW_TYPES:
    DTYPE_STR   = pd.ArrowDtype(pa.string())
    DTYPE_FLOAT = pd.ArrowDtype(pa.float64())
    DTYPE_INT   = pd.ArrowDtype(pa.int64())
    DTYPE_BOOL  = pd.ArrowDtype(pa.bool_())
    DTYPE_INT8  = pd.ArrowDtype(pa.int8())
else:
    DTYPE_STR   = "string"
    DTYPE_FLOAT = "Float64"
    DTYPE_INT   = "Int64"
    DTYPE_BOOL  = "boolean"
    DTYPE_INT8  = "Int8"

# -----------------------
# Module exports & constants
# -----------------------
__all__ = ["prepare_catalog"]

LOGGER_NAME = "prepare_catalog_logger"

DP1_REGIONS = [
    (6.02, -72.08, 2.5),   # 47 Tuc
    (37.86, 6.98, 2.5),    # Rubin SV 38 7
    (40.00, -34.45, 2.5),  # Fornax dSph
    (53.13, -28.10, 2.5),  # ECDFS
    (59.10, -48.73, 2.5),  # EDFS
    (95.00, -25.00, 2.5),  # Rubin SV 95 -25
    (106.23, -10.51, 2.5), # Seagull
]

JADES_LETTER_TO_SCORE = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "E": 0.0}
VIMOS_FLAG_TO_SCORE = {
    "LRB_X": 0.0, "MR_X": 0.0,
    "LRB_B": 1.0, "LRB_C": 1.0, "MR_C": 1.0,
    "MR_B": 3.0,
    "LRB_A": 4.0, "MR_A": 4.0,
}

# -----------------------
# Logging utilities
# -----------------------
def _build_logger(logs_dir: str, name: str, file_name: str) -> logging.Logger:
    """Create and configure a file‐based logger for this module.

    The logger writes INFO+ messages to ``<logs_dir>/<file_name>`` using a
    timestamped single-line format. Existing handlers with the same logger
    name are cleared to prevent duplicate outputs when the function is
    called multiple times (e.g., in notebooks or retries).

    Args:
        logs_dir (str): Directory where the log file will be written. The
            directory is created if it does not exist.
        name (str): Logger name (e.g., ``LOGGER_NAME``).
        file_name (str): Log file name (e.g., ``"prepare_all.log"``).

    Returns:
        logging.Logger: Configured logger instance with a single FileHandler.

    Raises:
        OSError: If the log directory cannot be created or the file handler
            cannot be initialized.

    Notes:
        - ``propagate`` is disabled to avoid duplicate messages on root handlers.
        - The file handler uses ``encoding='utf-8'`` and ``delay=True`` so the
          file is created only on first write.
    """
    log_dir = pathlib.Path(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # keep logs local to this file-based handler

    # Remove previous handlers to avoid duplicates in repeated runs.
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_dir / file_name, encoding="utf-8", delay=True)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    logger.addHandler(fh)
    return logger


# -----------------------
# YAML mapping validation
# -----------------------
def _validate_and_rename(df: dd.DataFrame, entry: dict, logger: logging.Logger) -> dd.DataFrame:
    """Validate column mapping from YAML and perform safe renaming.

    This function verifies that all non-null source columns declared in the YAML
    mapping exist in the input dataframe, suggests close matches for missing
    sources, and then applies a *conflict-safe* rename:
    if the target name already exists in the dataframe and differs from the
    mapped source, the existing target is temporarily parked as
    ``<target>__orig`` (or ``__orig1``, ``__orig2``, ...) before renaming.

    It also tags the catalog ``source`` with the product internal name and
    ensures a minimal base schema is present with pandas *nullable dtypes*.

    Args:
        df (dd.DataFrame): Input Dask DataFrame read from the product handle.
        entry (dict): YAML entry for the product. Must include:
            - ``internal_name`` (str)
            - ``columns`` (mapping of standard->source; null/''/'null' ignored)
        logger (logging.Logger): Module logger.

    Returns:
        dd.DataFrame: DataFrame with validated/renamed columns and base schema.

    Raises:
        ValueError: If any required source column is missing, with suggestions.

    Notes:
        - Only non-null mappings are enforced; null/empty entries are ignored.
        - The rename map is built as {source -> standard}.
        - Base schema ensures downstream stability when certain fields are absent.
    """
    product_name = entry["internal_name"]
    columns_cfg = (entry.get("columns") or {})
    non_null_map = {std: src for std, src in columns_cfg.items() if src not in (None, "", "null")}
    input_cols = list(map(str, df.columns))

    # Check that all declared sources exist in the input
    missing_sources = [src for src in non_null_map.values() if src not in input_cols]
    if missing_sources:
        suggestions = {src: difflib.get_close_matches(src, input_cols, n=3, cutoff=0.6) for src in missing_sources}
        raise ValueError(
            f"[{product_name}] Missing mapped source columns in input parquet: {missing_sources}\n"
            f"Configured (non-null) mapping: {non_null_map}\n"
            f"Closest matches: {suggestions}\n"
            f"Available columns (sample): {sorted(input_cols)[:30]} ..."
        )

    # Resolve rename target collisions proactively:
    # if a target already exists (and differs from its source), park it aside.
    for std, src in non_null_map.items():
        tgt = std
        if src != tgt and tgt in df.columns:
            base = f"{tgt}__orig"
            parked = base
            i = 1
            existing = set(map(str, df.columns))
            while parked in existing:
                parked = f"{base}{i}"
                i += 1
            logger.info(
                f"{product_name} Resolving rename collision: target '{tgt}' already exists; "
                f"renaming existing '{tgt}' → '{parked}' so '{src}' can become '{tgt}'."
            )
            df = df.rename(columns={tgt: parked})

    # Apply the YAML rename: source -> standard
    col_map = {src: std for std, src in non_null_map.items()}
    if col_map:
        logger.info(f"{product_name} rename map OK (up to 6): {list(col_map.items())[:6]}")
        df = df.rename(columns=col_map)
    else:
        logger.info(f"{product_name} no non-null column mappings; proceeding without renaming.")

    # Tag the catalog source and ensure a minimal base schema with nullable dtypes
    df["source"] = product_name

    base_schema = {
        "id": DTYPE_STR,
        "instrument_type": DTYPE_STR,
        "survey": DTYPE_STR,
        "ra": DTYPE_FLOAT,
        "dec": DTYPE_FLOAT,
        "z": DTYPE_FLOAT,
        "z_flag": DTYPE_FLOAT,
        "z_err": DTYPE_FLOAT,
    }

    for col, pd_dtype in base_schema.items():
        if col not in df.columns:
            df = _add_missing_with_dtype(df, col, pd_dtype)

    return df


def _rename_duplicate_columns_dd(df: dd.DataFrame, logger: logging.Logger,
                                 protected: set[str] | None = None) -> dd.DataFrame:
    """
    Ensures unique names per partition. Keeps the first occurrence of each name
    (including 'protected'); renames the others to <name>__dupN.
    """
    protected = protected or set()

    def _renamer(pdf: pd.DataFrame) -> pd.DataFrame:
        cols = list(map(str, pdf.columns))
        seen: dict[str, int] = {}
        new_cols: list[str] = []
        renamed_pairs: list[tuple[str, str]] = []

        for c in cols:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_name = f"{c}__dup{seen[c]}"
                while new_name in seen:
                    seen[c] += 1
                    new_name = f"{c}__dup{seen[c]}"
                seen[new_name] = 0
                new_cols.append(new_name)
                renamed_pairs.append((c, new_name))

        if renamed_pairs:
            sample = ", ".join([f"{a}->{b}" for a, b in renamed_pairs[:6]])
            logger.warning(f"⚠️ Renamed duplicate columns in partition: {sample}"
                           f"{' ...' if len(renamed_pairs) > 6 else ''}")

        out = pdf.copy()
        out.columns = new_cols
        return out

    meta_fixed = _renamer(df._meta)
    return df.map_partitions(_renamer, meta=meta_fixed)


def _rename_duplicate_columns_pd(pdf: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    cols = pd.Index(map(str, pdf.columns))
    if cols.has_duplicates:
        seen = {}
        new_cols = []
        renamed = []
        for c in cols:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new = f"{c}__dup{seen[c]}"
                while new in seen:
                    seen[c] += 1
                    new = f"{c}__dup{seen[c]}"
                seen[new] = 0
                new_cols.append(new)
                renamed.append((c, new))
        if renamed:
            sample = ", ".join([f"{a}->{b}" for a,b in renamed[:6]])
            logger.warning(f"⚠️ Renamed duplicate columns (pandas): {sample}"
                           f"{' ...' if len(renamed) > 6 else ''}")
        pdf = pdf.copy()
        pdf.columns = new_cols
    return pdf

# -----------------------
# Type helpers
# -----------------------
def _add_missing_with_dtype(_df: dd.DataFrame, col: str, pd_dtype: Any) -> dd.DataFrame:
    """Add a missing column with a nullable/extension dtype and correct Dask meta.

    If `col` already exists, returns `_df` unchanged.

    Args:
      _df: Input Dask DataFrame.
      col: Column name to add if missing.
      pd_dtype: Target pandas dtype (e.g., "Float64", "Int64", "string", "boolean",
        or a pandas ExtensionDtype such as `pd.ArrowDtype(pa.float64())`).

    Returns:
      dd.DataFrame: `_df` if `col` exists; otherwise a new DataFrame with `col`
      filled with <NA> and typed as `pd_dtype`.
    """
    if col in _df.columns:
        return _df

    # Accurate meta via empty Series of the requested dtype.
    meta_added = _df._meta.assign(**{col: pd.Series(pd.array([], dtype=pd_dtype))})

    def _adder(part: pd.DataFrame) -> pd.DataFrame:
        p = part.copy()
        p[col] = pd.Series(pd.NA, index=p.index, dtype=pd_dtype)
        return p

    return _df.map_partitions(_adder, meta=meta_added)


def _normalize_string_series_to_na(s: pd.Series) -> pd.Series:
    """Normalize to StringDtype and coerce placeholders to <NA>.

    Operations:
      * cast to string dtype
      * strip whitespace
      * map '', None, 'none', 'null', 'nan' (case-insensitive) to <NA>

    Args:
      s: Input Series.

    Returns:
      pd.Series: StringDtype series with canonical missing values.
    """
    s = s.astype(DTYPE_STR).str.strip()
    low = s.fillna("").str.lower()
    mask_empty = (low == "") | low.isin(["none", "null", "nan"])
    return s.mask(mask_empty, pd.NA).astype(DTYPE_STR)


def _to_nullable_boolean_strict(s: pd.Series) -> pd.Series:
    """Convert to nullable boolean with strict semantics.

    Only `True`/`False` (bool or numpy.bool_) are preserved. Other values,
    including string equivalents, become <NA>.

    Args:
      s: Input Series.

    Returns:
      pd.Series: Boolean nullable series with values in {True, False, <NA>}.
    """
    if s.dtype == object or str(s.dtype).startswith("string"):
        s = s.astype(DTYPE_STR).str.strip()
        low = s.fillna("").str.lower()
        s = s.mask((low == "") | low.isin(["none", "null", "nan"]), pd.NA)

    vals = s.astype("object")
    mask_true = vals.apply(lambda v: isinstance(v, (bool, np.bool_)) and v is True)
    mask_false = vals.apply(lambda v: isinstance(v, (bool, np.bool_)) and v is False)

    out = pd.Series(pd.array([pd.NA] * len(vals), dtype=DTYPE_BOOL), index=vals.index)
    out[mask_true] = True
    out[mask_false] = False
    return out


def _normalize_schema_hints(hints: dict | None) -> dict:
    """Normalize YAML dtype hints to {'int','float','str','bool'}.

    Args:
      hints: Mapping column -> dtype string.

    Returns:
      dict: Normalized mapping; unknown dtypes are ignored.
    """
    if not hints:
        return {}
    norm = {}
    for col, dt in hints.items():
        if dt is None:
            continue
        k = str(col)
        v = str(dt).strip().lower()
        if v in {"int", "int64"}:
            norm[k] = "int"
        elif v in {"float", "float64", "double"}:
            norm[k] = "float"
        elif v in {"str", "string"}:
            norm[k] = "str"
        elif v in {"bool", "boolean"}:
            norm[k] = "bool"
    return norm


def _normalize_types(df: dd.DataFrame, product_name: str, logger: logging.Logger) -> tuple[dd.DataFrame, bool]:
    """Normalize core dtypes and clean values.

    Steps:
      1) String-like cols -> normalized string dtype with canonical <NA>.
      2) Optional `type` -> normalized string, lower-cased.
      3) z_flag special-cases for JADES/VIMOS (string encodings -> numeric).
      4) Float-like cols -> numeric, cast to target float dtype with checks.

    Args:
      df: Dask DataFrame after YAML rename.
      product_name: Catalog identifier for log context.
      logger: Logger.

    Returns:
      Tuple[dd.DataFrame, bool]: (normalized df, type_cast_ok).

    Raises:
      ValueError: On string normalization failure or non-numeric residues during
        float coercion.

    Notes:
      - `survey` is upper-cased after normalization to stabilize rule lookups.
      - Non-numeric residues are detected via `dd.to_numeric(..., errors="coerce")`
        and reported with a small value sample.
    """
    # 1) Normalize string-like columns
    string_like = ["id", "instrument_type", "survey", "instrument_type_homogenized", "source"]
    for col in string_like:
        if col in df.columns:
            try:
                df[col] = df[col].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
                )
                if col == "survey":
                    df[col] = df[col].str.upper()
            except Exception as e:
                raise ValueError(
                    f"[{product_name}] Failed to normalize string column '{col}'. "
                    f"Original error: {e}"
                ) from e

    # 2) Optional auxiliary 'type'
    type_cast_ok = False
    if "type" in df.columns:
        try:
            df["type"] = df["type"].map_partitions(
                _normalize_string_series_to_na,
                meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
            ).str.lower()
            type_cast_ok = True
        except Exception as e:
            logger.warning(f"⚠️ {product_name} Failed to normalize 'type' to lower-case: {e}")
            type_cast_ok = False

    # 3) z_flag mapping for JADES/VIMOS
    def _map_special_partition(partition: pd.DataFrame) -> pd.DataFrame:
        p = partition.copy()
        if "survey" not in p or "z_flag" not in p:
            return p

        survey_uc = p["survey"].astype(str).str.upper()
        mask_jades = survey_uc == "JADES"
        mask_vimos = survey_uc == "VIMOS"
        if not (mask_jades.any() or mask_vimos.any()):
            return p

        zf_num = pd.to_numeric(p["z_flag"], errors="coerce")
        nonnum_mask = zf_num.isna()
        z_flag_str = p["z_flag"].astype(str)

        def map_jades(val):
            s = str(val).strip().upper()
            return JADES_LETTER_TO_SCORE.get(s, np.nan)

        def map_vimos(val):
            s = str(val).strip().upper()
            return VIMOS_FLAG_TO_SCORE.get(s, np.nan)

        idx_j = nonnum_mask & mask_jades
        if idx_j.any():
            z_flag_str.loc[idx_j] = z_flag_str.loc[idx_j].map(map_jades)

        idx_v = nonnum_mask & mask_vimos
        if idx_v.any():
            z_flag_str.loc[idx_v] = z_flag_str.loc[idx_v].map(map_vimos)

        mapped_numeric = pd.to_numeric(z_flag_str, errors="coerce")
        zf = zf_num.copy()
        zf[nonnum_mask] = mapped_numeric[nonnum_mask]
        p["z_flag"] = zf
        return p

    if "z_flag" in df.columns and "survey" in df.columns:
        df = df.map_partitions(_map_special_partition)

    # 4) Coerce float-like columns with diagnostics
    float_like = ["ra", "dec", "z", "z_err", "z_flag", "z_flag_homogenized"]
    for col in float_like:
        if col in df.columns:
            try:
                coerced = dd.to_numeric(df[col], errors="coerce")
                invalid_mask = dd.isna(coerced) & ~dd.isna(df[col])
                invalid_count = dask.compute(invalid_mask.sum())[0]
                if invalid_count > 0:
                    sample_vals = df[col].loc[invalid_mask].head(5, compute=True).tolist()
                    raise ValueError(
                        f"[{product_name}] Failed to convert '{col}' to numeric: "
                        f"{invalid_count} non-numeric value(s). Examples: {sample_vals}. "
                        f"Tip: fix or drop non-numeric entries before running."
                    )
                df[col] = coerced.map_partitions(
                    lambda s: s.astype(DTYPE_FLOAT),
                    meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
                )
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(
                    f"[{product_name}] Unexpected failure converting '{col}' to numeric. "
                    f"Original error: {e}"
                ) from e

    return df, type_cast_ok


# -----------------------
# CRD_ID generation
# -----------------------
def _generate_crd_ids(df: dd.DataFrame, product_name: str, temp_dir: str) -> tuple[dd.DataFrame, str]:
    """Assign stable, catalog-scoped CRD_IDs and return the `compared_to` path.

    The CRD_ID has the form ``CRD{NNN}_{i}``, where ``NNN`` is the numeric
    prefix extracted from ``product_name`` (``entry['internal_name']``) and
    ``i`` is a 1-based, contiguous counter across all partitions. This function
    computes partition lengths, derives running offsets, and assigns IDs
    deterministically per partition to avoid a single large shuffle.

    It also prepares the file path where the per-catalog pair graph
    (``compared_to``) will be persisted later in the pipeline.

    Args:
        df (dd.DataFrame): Input Dask DataFrame after schema normalization.
        product_name (str): Product internal name, expected to start with a
            numeric prefix (e.g., ``"492_vltvimos_v201"``).
        temp_dir (str): Directory where intermediate artifacts are written.

    Returns:
        tuple[dd.DataFrame, str]: The DataFrame with a new ``CRD_ID`` column
        (pandas ``StringDtype``) and the absolute path to the JSON file where
        this catalog's ``compared_to`` dictionary should be saved.

    Raises:
        ValueError: If a numeric prefix cannot be extracted from ``product_name``.

    Notes:
        - The assignment strategy is partition-local to avoid reindexing/shuffling
          the entire Dask collection.
        - CRD_IDs are stable for a given partitioning and row order; if the input
          partitioning/order changes, IDs may change accordingly.
    """
    m = re.match(r"(\d+)_", product_name)
    if not m:
        raise ValueError(f"❌ Could not extract numeric prefix from internal_name '{product_name}'")
    catalog_prefix = m.group(1)

    # Compute partition sizes on the driver to derive contiguous offsets.
    sizes = df.map_partitions(len).compute().tolist()
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)

    def _add_crd(part: pd.DataFrame, start: int) -> pd.DataFrame:
        # Partition-local ID assignment using a contiguous range seeded by `start`.
        p = part.copy()
        n = len(p)
        p["CRD_ID"] = [f"CRD{catalog_prefix}_{start + i + 1}" for i in range(n)]
        return p

    parts = [
        df.get_partition(i).map_partitions(
            _add_crd, offset,
            meta=df._meta.assign(CRD_ID=pd.Series(pd.array([], dtype=DTYPE_STR)))
        )
        for i, offset in enumerate(offsets)
    ]
    df = dd.concat(parts)

    compared_to_path = os.path.join(temp_dir, f"compared_to_dict_{catalog_prefix}.json")
    return df, compared_to_path

# -----------------------
# Homogenization (respect user-provided columns)
# -----------------------
def _honor_user_homogenized_mapping(
    df: dd.DataFrame,
    entry: dict,
    product_name: str,
    logger: logging.Logger,
) -> dd.DataFrame:
    """Recreate canonical homogenized columns when YAML maps to homogenized sources.

    Some inputs may already provide homogenized fields and choose to expose
    them through the standard names via YAML mapping. This function detects
    those cases and mirrors the values to the canonical homogenized columns
    expected by downstream logic:

      - If YAML declares ``z_flag <- z_flag_homogenized``, then create/overwrite
        ``z_flag_homogenized`` from the (renamed) ``z_flag``.
      - If YAML declares ``instrument_type <- instrument_type_homogenized``,
        then create/overwrite ``instrument_type_homogenized`` from
        ``instrument_type``.

    A protective check prevents silent duplication if the rename step resulted
    in two columns named the same (should not happen if `_validate_and_rename`
    handled collisions).

    Args:
        df (dd.DataFrame): DataFrame after YAML rename.
        entry (dict): Product YAML node containing the ``columns`` mapping.
        product_name (str): Internal name for logging context.
        logger (logging.Logger): Logger.

    Returns:
        dd.DataFrame: DataFrame with canonical homogenized columns present
        (if derivable from the YAML mapping).

    Raises:
        ValueError: If duplicate columns with the same name are detected after
            YAML rename (indicates a misconfigured mapping or a collision).
    """

    def _ensure_single_series(df: dd.DataFrame, name: str):
        # Fail fast if a rename collision left duplicated columns with identical names.
        if list(df.columns).count(name) > 1:
            raise ValueError(
                f"[{product_name}] Detected duplicate '{name}' columns after YAML rename. "
                f"This typically happens when the file already had '{name}' and you also mapped "
                f"another source to '{name}'. The pipeline expects unique column names. "
                f"Tip: let the pipeline resolve collisions (see logs) or adjust the YAML."
            )
        return df[name]

    cols_cfg = (entry.get("columns") or {})
    norm = lambda x: (str(x).strip().lower() if x is not None else None)

    # z_flag <- z_flag_homogenized ?
    if norm(cols_cfg.get("z_flag")) == "z_flag_homogenized":
        if "z_flag" in df.columns:
            df["z_flag_homogenized"] = _ensure_single_series(df, "z_flag")
            logger.info(f"{product_name} YAML maps 'z_flag' <- 'z_flag_homogenized'; using user-provided homogenized z_flag.")
        else:
            logger.warning(f"⚠️ {product_name} YAML claims 'z_flag' comes from 'z_flag_homogenized', but 'z_flag' is missing after rename.")

    # instrument_type <- instrument_type_homogenized ?
    if norm(cols_cfg.get("instrument_type")) == "instrument_type_homogenized":
        if "instrument_type" in df.columns:
            df["instrument_type_homogenized"] = _ensure_single_series(df, "instrument_type")
            logger.info(f"{product_name} YAML maps 'instrument_type' <- 'instrument_type_homogenized'; using user-provided homogenized instrument_type.")
        else:
            logger.warning(f"⚠️ {product_name} YAML claims 'instrument_type' comes from 'instrument_type_homogenized', but 'instrument_type' is missing after rename.")

    return df


def _homogenize(
    df: dd.DataFrame,
    translation_config: dict,
    product_name: str,
    logger: logging.Logger,
    type_cast_ok: bool,
) -> tuple[dd.DataFrame, bool, list, dict, dict]:
    """Compute homogenized columns for tie-breaking.

    Strategy:
      1) Fast-paths: reuse quality-like `z_flag` and validated `type` when possible.
      2) Otherwise, apply YAML translations per partition (vectorized).

    Args:
      df: Frame after type normalization.
      translation_config: Config with priorities and translation rules.
      product_name: Catalog identifier for log context.
      logger: Logger.
      type_cast_ok: Whether `type` was normalized.

    Returns:
      Tuple[df, used_type_fastpath, tiebreaking_priority, instrument_type_priority, translation_rules_uc].
    """
    tiebreaking_priority = translation_config.get("tiebreaking_priority", [])
    instrument_type_priority = translation_config.get("instrument_type_priority", {})
    translation_rules_uc = {k.upper(): v for k, v in translation_config.get("translation_rules", {}).items()}

    # -----------------------
    # Vectorized translator
    # -----------------------
    def _translate_column_vectorized(df: dd.DataFrame, key: str, out_col: str, out_kind: str) -> dd.DataFrame:
        """Apply YAML translation rules per partition (no row-wise Python loops)."""
        assert key in {"z_flag", "instrument_type"}
        assert out_col in {"z_flag_homogenized", "instrument_type_homogenized"}
        assert out_kind in {"float", "str"}

        def _partition(p: pd.DataFrame) -> pd.DataFrame:
            if p.empty or ("survey" not in p.columns) or (key not in p.columns):
                q = p.copy()
                if out_kind == "float":
                    q[out_col] = pd.Series(pd.array([], dtype=DTYPE_FLOAT)).reindex(q.index)
                else:
                    q[out_col] = pd.Series(pd.array([], dtype=DTYPE_STR)).reindex(q.index)
                return q

            s = p.copy()
            survey_uc = s["survey"].astype(str).str.upper()

            # Initialize output series.
            if out_kind == "float":
                out = pd.Series(pd.array([pd.NA] * len(s), dtype=DTYPE_FLOAT), index=s.index)
            else:
                out = pd.Series(pd.array([pd.NA] * len(s), dtype=DTYPE_STR), index=s.index)

            # --- helpers for vectorized exprs ---
            class StrSeriesProxy:
                """Elementwise string slicing via .str accessors."""
                def __init__(self, ser: pd.Series):
                    self._s = ser.astype(DTYPE_STR)
                def __getitem__(self, key):
                    if isinstance(key, slice):
                        return self._s.str.slice(key.start, key.stop, key.step)
                    return self._s.str.get(key)

            def v_len(x):
                if isinstance(x, StrSeriesProxy):
                    return x._s.str.len()
                if isinstance(x, pd.Series):
                    return x.astype(DTYPE_STR).str.len()
                return builtins.len(x)

            def v_str(x):
                if isinstance(x, StrSeriesProxy):
                    return x
                if isinstance(x, pd.Series):
                    return StrSeriesProxy(x)
                return builtins.str(x)

            def v_int(x):
                if isinstance(x, StrSeriesProxy):
                    x = x._s
                if isinstance(x, pd.Series):
                    return pd.to_numeric(x, errors="coerce").astype(DTYPE_INT)
                return builtins.int(x)

            def v_float(x):
                if isinstance(x, StrSeriesProxy):
                    x = x._s
                if isinstance(x, pd.Series):
                    return pd.to_numeric(x, errors="coerce").astype(DTYPE_FLOAT)
                return builtins.float(x)

            import ast as _ast

            class _BoolToBitwise(_ast.NodeTransformer):
                """Rewrite boolean ops and chained comparisons to bitwise equivalents."""
                def visit_BoolOp(self, node):
                    self.generic_visit(node)
                    op = _ast.BitAnd() if isinstance(node.op, _ast.And) else _ast.BitOr()
                    expr = node.values[0]
                    for v in node.values[1:]:
                        expr = _ast.BinOp(left=expr, op=op, right=v)
                    return expr
                def visit_UnaryOp(self, node):
                    self.generic_visit(node)
                    if isinstance(node.op, _ast.Not):
                        return _ast.UnaryOp(op=_ast.Invert(), operand=node.operand)
                    return node
                def visit_Compare(self, node):
                    self.generic_visit(node)
                    if len(node.ops) <= 1:
                        return node
                    expr = _ast.Compare(left=node.left, ops=[node.ops[0]], comparators=[node.comparators[0]])
                    for i in range(1, len(node.ops)):
                        cmp_i = _ast.Compare(left=node.comparators[i-1], ops=[node.ops[i]], comparators=[node.comparators[i]])
                        expr = _ast.BinOp(left=expr, op=_ast.BitAnd(), right=cmp_i)
                    return expr

            # Apply rules per survey.
            for sname, ruleset in translation_rules_uc.items():
                rule = (ruleset.get(f"{key}_translation") or {})
                if not isinstance(rule, dict) or not len(rule):
                    continue

                mask_s = (survey_uc == sname)
                if not mask_s.any():
                    continue

                default_val = rule.get("default", (np.nan if out_kind == "float" else ""))

                # Default fill.
                if out_kind == "float":
                    out.loc[mask_s] = out.loc[mask_s].fillna(default_val)
                else:
                    fill_vals = pd.Series([default_val] * int(mask_s.sum()), index=out.index[mask_s], dtype=DTYPE_STR)
                    out.loc[mask_s] = out.loc[mask_s].fillna(fill_vals)

                # Direct mappings.
                direct = {k: v for k, v in rule.items() if k not in {"conditions", "default"}}
                if direct:
                    col = s.loc[mask_s, key]
                    is_num = pd.api.types.is_numeric_dtype(col)
                    if key == "z_flag":
                        is_num = True  # allow numeric path even if dtype looks object

                    if is_num:
                        col_num = pd.to_numeric(col, errors="coerce")
                        num_map = {}
                        for rk, rv in direct.items():
                            try:
                                num_map[float(rk)] = rv
                            except Exception:
                                pass
                        mapped = col_num.map(num_map)
                    else:
                        col_str = col.astype(str).str.strip().str.lower()
                        str_map = {str(k).strip().lower(): v for k, v in direct.items()}
                        mapped = col_str.map(str_map)

                    if out_kind == "float":
                        out.loc[mask_s] = mapped.fillna(out.loc[mask_s]).astype(DTYPE_FLOAT)
                    else:
                        mapped = mapped.astype("object")
                        mapped_str = pd.Series(pd.array(mapped.where(mapped.notna(), None), dtype=DTYPE_STR), index=mapped.index)
                        take = mapped_str.notna()
                        out.loc[take.index] = out.loc[take.index].where(~take, mapped_str)

                # Conditional rules.
                for cond in (rule.get("conditions") or []):
                    expr = cond.get("expr")
                    if not expr:
                        continue

                    ctx = {c: s[c] for c in s.columns if c in s.columns}
                    safe_globals = {
                        "__builtins__": {},
                        "len": v_len,
                        "int": v_int,
                        "float": v_float,
                        "str": v_str,
                        "math": math,
                        "np": np,
                        "pd": pd,
                    }

                    try:
                        tree = _ast.parse(expr, mode="eval")
                        tree2 = _BoolToBitwise().visit(tree)
                        _ast.fix_missing_locations(tree2)
                        expr_vec = _ast.unparse(tree2)
                        mlocal = eval(expr_vec, safe_globals, ctx)
                    except Exception as e:
                        raise ValueError(f"Error evaluating condition '{expr}' for survey '{sname}': {e}")

                    if isinstance(mlocal, pd.Series):
                        mlocal = mlocal.reindex(s.index)
                    elif hasattr(mlocal, "__len__") and not np.isscalar(mlocal) and len(mlocal) == len(s):
                        mlocal = pd.Series(mlocal, index=s.index)
                    else:
                        mlocal = pd.Series(bool(mlocal), index=s.index)

                    mlocal = mlocal.astype("boolean").fillna(False)
                    mask_s_aligned = mask_s.reindex(s.index).fillna(False).astype(bool)
                    mlocal = mlocal & mask_s_aligned

                    val = cond.get("value", default_val)
                    if out_kind == "float":
                        out.loc[mlocal] = pd.to_numeric(val, errors="coerce")
                    else:
                        out.loc[mlocal] = (pd.NA if pd.isna(val) else str(val))

            # Finalize dtype.
            if out_col == "z_flag_homogenized":
                s[out_col] = pd.to_numeric(out, errors="coerce").astype(DTYPE_FLOAT)
            else:
                s[out_col] = pd.Series(out, dtype=DTYPE_STR).str.lower()

            return s

        # Accurate meta for new column.
        meta = df._meta.copy()
        if out_kind == "float":
            meta[out_col] = pd.Series(pd.array([], dtype=DTYPE_FLOAT))
        else:
            meta[out_col] = pd.Series(pd.array([], dtype=DTYPE_STR))

        return df.map_partitions(_partition, meta=meta)

    # -----------------------
    # z_flag_homogenized
    # -----------------------
    def can_use_zflag_as_quality() -> bool:
        """Return True if `z_flag` looks like a [0,1] quality score with fractional values."""
        if "z_flag" not in df.columns:
            return False
        try:
            cnt = df["z_flag"].count()
            minv = df["z_flag"].min()
            maxv = df["z_flag"].max()
            frac_cnt = df[(df["z_flag"] > 0.0) & (df["z_flag"] < 1.0)]["z_flag"].count()
            cnt, minv, maxv, frac_cnt = dask.compute(cnt, minv, maxv, frac_cnt)
            if cnt == 0 or not np.isfinite(minv) or not np.isfinite(maxv):
                return False
            if not (minv >= 0.0 and maxv <= 1.0):
                return False
            return frac_cnt > 0
        except Exception as e:
            logger.warning(f"⚠️ {product_name} Could not validate 'z_flag' as quality-like: {e}")
            return False

    def quality_like_to_flag(x):
        """Map a [0,1] quality score to VVDS-like flags {0..4}."""
        if pd.isna(x):
            return np.nan
        x = float(x)
        if x == 0.0:
            return 0.0
        if 0.0 < x < 0.7:
            return 1.0
        if 0.7 <= x < 0.9:
            return 2.0
        if 0.9 <= x < 0.99:
            return 3.0
        if 0.99 <= x <= 1.0:
            return 4.0
        return np.nan

    if "z_flag_homogenized" in tiebreaking_priority:
        if "z_flag_homogenized" not in df.columns:
            if can_use_zflag_as_quality():
                logger.info(f"{product_name} Using 'z_flag' (quality-like) fast path for z_flag_homogenized.")
                df["z_flag_homogenized"] = df["z_flag"].map_partitions(
                    lambda s: s.apply(quality_like_to_flag).astype(DTYPE_FLOAT),
                    meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
                )
            else:
                logger.info(f"{product_name} z_flag not quality-like; using YAML translation for z_flag_homogenized.")
                df = _translate_column_vectorized(df, key="z_flag", out_col="z_flag_homogenized", out_kind="float")
        else:
            logger.warning(f"{product_name} Column 'z_flag_homogenized' already exists. Skipping recompute.")

    # -----------------------
    # instrument_type_homogenized
    # -----------------------
    used_type_fastpath = False

    def can_use_type_for_instrument() -> bool:
        """Reuse normalized `type` only if values are a subset of {s,g,p}."""
        if not type_cast_ok or "type" not in df.columns:
            return False
        try:
            allowed = {"s", "g", "p"}
            uniques = df["type"].dropna().unique().compute().tolist()
            uniques = [u for u in uniques if isinstance(u, (str, np.str_)) and u != ""]
            if len(uniques) == 0:
                return False
            return all(u in allowed for u in uniques)
        except Exception as e:
            logger.warning(f"⚠️ {product_name} Could not validate 'type' values: {e}")
            return False

    if "instrument_type_homogenized" in tiebreaking_priority:
        if "instrument_type_homogenized" not in df.columns:
            if can_use_type_for_instrument():
                logger.info(f"{product_name} Using 'type' fast path for instrument_type_homogenized.")
                df["instrument_type_homogenized"] = df["type"].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
                ).str.lower()
                used_type_fastpath = True
            else:
                logger.info(f"{product_name} 'type' not suitable; using YAML translation for instrument_type_homogenized.")
                df = _translate_column_vectorized(df, key="instrument_type", out_col="instrument_type_homogenized", out_kind="str")
                df["instrument_type_homogenized"] = df["instrument_type_homogenized"].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
                ).str.lower()
        else:
            logger.warning(f"{product_name} Column 'instrument_type_homogenized' already exists. Skipping recompute.")

    return df, used_type_fastpath, tiebreaking_priority, instrument_type_priority, translation_rules_uc

# -----------------------
# Tie-breaking / duplicates
# -----------------------
def _apply_tiebreaking_and_collect(
    df: dd.DataFrame,
    translation_config: dict,
    tiebreaking_priority: list,
    instrument_type_priority: dict,
    compared_to_dict_solo: dict[str, set[str]],
    product_name: str,
    logger: logging.Logger,
) -> dd.DataFrame:
    """Resolve duplicates via prioritized tie-breaking and collect undirected edges.

    Strategy:
      1) Build RA/DEC key (rounded to 1e-6 deg).
      2) Keep only keys with global duplicates (count > 1).
      3) Shuffle rows so that same key coalesces in a partition.
      4) Per-partition tie-breaking -> updates (CRD_ID, tie_result).
      5) Per-partition edge emission -> (CRD_ID_A, CRD_ID_B).
      6) Merge updates back; default non-dups to tie_result=1.
      7) Accumulate edges on the driver.

    Robustness:
      - Prefer p2p shuffle for speed; on any p2p consistency error, fall back
        to 'tasks' shuffle automatically.
    """
    import dask
    from dask.distributed import get_client

    # ---- Config ----
    delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)

    # ---- Validate tiebreaking columns ----
    if not tiebreaking_priority:
        logger.warning(f"⚠️ {product_name} tiebreaking_priority is empty. Using delta_z_threshold only.")
    else:
        for col in tiebreaking_priority:
            if col not in df.columns:
                raise ValueError(f"Tiebreaking column '{col}' is missing in catalog '{product_name}'.")
            col_dtype = df[col].dtype
            is_str = (col == "instrument_type_homogenized") or pd.api.types.is_string_dtype(col_dtype)
            empty_mask = df[col].isna()
            if is_str:
                empty_mask = empty_mask | (df[col] == "")
            if empty_mask.all().compute():
                raise ValueError(
                    f"Tiebreaking column '{col}' is invalid in catalog '{product_name}' (all NaN/empty)."
                )
            if col != "instrument_type_homogenized" and not pd.api.types.is_numeric_dtype(col_dtype):
                try:
                    df[col] = dd.to_numeric(df[col], errors="coerce").astype("float64")
                    logger.info(f"ℹ️ {product_name} cast '{col}' to float for tie-breaking.")
                except Exception as e:
                    raise ValueError(
                        f"Tiebreaking column '{col}' must be numeric (except 'instrument_type_homogenized'). "
                        f"Attempted cast failed: {e}"
                    )

    if not tiebreaking_priority and (delta_z_threshold is None or float(delta_z_threshold) == 0.0):
        raise ValueError(
            f"Cannot deduplicate '{product_name}': no tiebreaking_priority and delta_z_threshold unset/zero."
        )

    # ---- Build key and prefilter duplicates ----
    valid_coord_mask = df["ra"].notnull() & df["dec"].notnull()
    df_valid = df[valid_coord_mask].assign(
        ra_dec_key=(
            df["ra"].astype("float64").round(6).astype(str) + "_" +
            df["dec"].astype("float64").round(6).astype(str)
        )
    )

    # Contagem global para selecionar apenas chaves com duplicatas
    with dask.config.set({"dataframe.shuffle.method": "tasks"}):  # contagem é leve; 'tasks' é mais robusto
        key_counts = df_valid["ra_dec_key"].value_counts(split_out=64)
    dup_keys = key_counts[key_counts > 1]
    df_dups = df_valid.merge(dup_keys.to_frame("cnt"), left_on="ra_dec_key", right_index=True, how="inner")

    # Short-circuit: sem duplicatas
    try:
        _ = df_dups.head(1, compute=True)
    except Exception:
        df["CRD_ID"] = df["CRD_ID"].astype(DTYPE_STR)
        if "tie_result" not in df.columns:
            df["tie_result"] = pd.Series(pd.array([], dtype=DTYPE_INT8)).reindex_like(df, fill_value=1)
        else:
            df["tie_result"] = df["tie_result"].fillna(1).astype(DTYPE_INT8)
        return df

    # ---- Helpers: partition ops ----
    def _tiebreak_partition_noindex(p: pd.DataFrame) -> pd.DataFrame:
        if p.empty:
            return pd.DataFrame({
                "CRD_ID": pd.Series(pd.array([], dtype=DTYPE_STR)),
                "tie_result": pd.Series(pd.array([], dtype=DTYPE_INT8)),
            })

        updates: list[pd.DataFrame] = []

        for _, group_raw in p.groupby("ra_dec_key", sort=False):
            group = group_raw.copy()
            group["tie_result"] = 0
            surviving = group.copy()

            for priority_col in tiebreaking_priority:
                if priority_col == "instrument_type_homogenized":
                    pr = surviving["instrument_type_homogenized"].map(instrument_type_priority).astype("float64")
                    surviving["_priority_value"] = pr.fillna(-np.inf)
                else:
                    surviving["_priority_value"] = surviving[priority_col].astype("float64").fillna(-np.inf)

                # Early elimination: stars (flag 6) if z_flag_homogenized is used.
                if priority_col == "z_flag_homogenized":
                    ids_to_eliminate = surviving.loc[
                        surviving["z_flag_homogenized"] == 6, "CRD_ID"
                    ].astype(str).tolist()
                    if ids_to_eliminate:
                        group.loc[group["CRD_ID"].astype(str).isin(ids_to_eliminate), "tie_result"] = 0
                        surviving = surviving[surviving["z_flag_homogenized"] != 6]

                if surviving.empty:
                    break

                max_val = surviving["_priority_value"].max()
                surviving = surviving[surviving["_priority_value"] == max_val].drop(columns=["_priority_value"], errors="ignore")

                if len(surviving) == 1:
                    break

            group.loc[group["CRD_ID"].astype(str).isin(surviving["CRD_ID"].astype(str)), "tie_result"] = 2

            if len(surviving) > 1 and (delta_z_threshold or 0) > 0:
                z_vals = pd.to_numeric(surviving["z"], errors="coerce").to_numpy(dtype=float, copy=False)
                ids = surviving["CRD_ID"].astype(str).to_numpy(copy=False)
                remaining = set(ids)
                thr = float(delta_z_threshold)

                for i in range(len(ids)):
                    if ids[i] not in remaining:
                        continue
                    for j in range(i + 1, len(ids)):
                        if ids[j] not in remaining:
                            continue
                        zi, zj = z_vals[i], z_vals[j]
                        if not (np.isfinite(zi) and np.isfinite(zj)):
                            continue
                        if abs(zi - zj) <= thr:
                            group.loc[group["CRD_ID"].astype(str) == ids[i], "tie_result"] = 2
                            group.loc[group["CRD_ID"].astype(str) == ids[j], "tie_result"] = 0
                            remaining.discard(ids[j])

            survivors = group[group["tie_result"] == 2]
            if len(survivors) == 1:
                group.loc[group["tie_result"] == 2, "tie_result"] = 1
            elif len(survivors) == 0:
                non_elim = group[group["tie_result"] != 0]
                if len(non_elim) == 1:
                    cid = str(non_elim.iloc[0]["CRD_ID"])
                    group.loc[group["CRD_ID"].astype(str) == cid, "tie_result"] = 1

            updates.append(group[["CRD_ID", "tie_result"]])

        out = pd.concat(updates, ignore_index=True) if updates else pd.DataFrame(columns=["CRD_ID", "tie_result"])
        if not out.empty:
            out["CRD_ID"] = out["CRD_ID"].astype(DTYPE_STR)
            out["tie_result"] = out["tie_result"].astype(DTYPE_INT8)
        else:
            out = pd.DataFrame({
                "CRD_ID": pd.Series(pd.array([], dtype=DTYPE_STR)),
                "tie_result": pd.Series(pd.array([], dtype=DTYPE_INT8)),
            })
        return out

    def _edges_partition_noindex(p: pd.DataFrame) -> pd.DataFrame:
        if p.empty:
            return pd.DataFrame({
                "CRD_ID_A": pd.Series(pd.array([], dtype=DTYPE_STR)),
                "CRD_ID_B": pd.Series(pd.array([], dtype=DTYPE_STR)),
            })
        edges = []
        for _, group in p.groupby("ra_dec_key", sort=False):
            ids = group["CRD_ID"].astype(str).tolist()
            n = len(ids)
            if n <= 1:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = ids[i], ids[j]
                    if a == b:
                        continue
                    a_norm, b_norm = (a, b) if a < b else (b, a)
                    edges.append((a_norm, b_norm))
        if not edges:
            return pd.DataFrame({
                "CRD_ID_A": pd.Series(pd.array([], dtype=DTYPE_STR)),
                "CRD_ID_B": pd.Series(pd.array([], dtype=DTYPE_STR)),
            })
        out = pd.DataFrame(edges, columns=["CRD_ID_A", "CRD_ID_B"])
        out["CRD_ID_A"] = out["CRD_ID_A"].astype(DTYPE_STR)
        out["CRD_ID_B"] = out["CRD_ID_B"].astype(DTYPE_STR)
        return out

    # ---- Shuffle + fallback wrapper ----
    def run_shuffle_and_compute():
        client = get_client()
        try:
            nworkers = len(client.scheduler_info()["workers"]) or 1
        except Exception:
            nworkers = 1
        target_parts = max(2 * nworkers, df_dups.npartitions)
    
        with dask.config.set({"dataframe.shuffle.method": "tasks"}):
            df_shuf = df_dups.shuffle("ra_dec_key", shuffle="tasks", npartitions=target_parts)
    
            meta_updates = pd.DataFrame({
                "CRD_ID": pd.Series(pd.array([], dtype=DTYPE_STR)),
                "tie_result": pd.Series(pd.array([], dtype=DTYPE_INT8)),
            })
            meta_edges = pd.DataFrame({
                "CRD_ID_A": pd.Series(pd.array([], dtype=DTYPE_STR)),
                "CRD_ID_B": pd.Series(pd.array([], dtype=DTYPE_STR)),
            })
    
            tie_update_dd = df_shuf.map_partitions(_tiebreak_partition_noindex, meta=meta_updates) \
                                   .drop_duplicates(subset=["CRD_ID"], keep="last")
            edges_dd      = df_shuf.map_partitions(_edges_partition_noindex,    meta=meta_edges) \
                                   .drop_duplicates()
    
            tie_update_dd, edges_dd = dask.persist(tie_update_dd, edges_dd)
    
        edges = edges_dd.compute()
        return tie_update_dd, edges

    tie_update_dd, edges = run_shuffle_and_compute()

    # ---- Merge decisions back ----
    df["CRD_ID"] = df["CRD_ID"].astype(DTYPE_STR)
    df = (
        df.drop("tie_result", axis=1, errors="ignore")
          .merge(tie_update_dd, on="CRD_ID", how="left")
          .fillna({"tie_result": 1})
    )
    df["tie_result"] = df["tie_result"].astype(DTYPE_INT8)

    # ---- Accumulate edges on driver ----
    for a, b in zip(edges.get("CRD_ID_A", []), edges.get("CRD_ID_B", [])):
        compared_to_dict_solo.setdefault(a, set()).add(b)
        compared_to_dict_solo.setdefault(b, set()).add(a)

    return df

# -----------------------
# RA/DEC strict validation (fail fast)
# -----------------------
def _validate_ra_dec_or_fail(df: dd.DataFrame, product_name: str) -> None:
    """Validate RA/DEC for finiteness and range; fail fast with details.

    Checks (using float64 views):
      * missing values (NaN / <NA>)
      * non-finite values (±inf)
      * RA out of [0, 360)
      * DEC out of [-90, 90]

    Args:
      df: DataFrame with columns 'ra' and 'dec'.
      product_name: Catalog identifier for error context.

    Raises:
      ValueError: If any invalid RA/DEC entries are detected.
    """
    def _isfinite_series(s: pd.Series) -> pd.Series:
        arr = s.astype("float64")  # Extension -> float64 (NaN where <NA>)
        return pd.Series(np.isfinite(arr), index=s.index)

    isfinite_ra  = df["ra"].map_partitions(_isfinite_series,  meta=("ra", "bool"))
    isfinite_dec = df["dec"].map_partitions(_isfinite_series, meta=("dec", "bool"))

    ra64  = df["ra"].astype("float64")
    dec64 = df["dec"].astype("float64")

    in_range = (
        (ra64 >= 0.0) & (ra64 < 360.0) &
        (dec64 >= -90.0) & (dec64 <= 90.0)
    )
    invalid_mask = ~(isfinite_ra & isfinite_dec & in_range)

    na_ra, na_dec = df["ra"].isna().sum(), df["dec"].isna().sum()
    nonfinite_ra, nonfinite_dec = (~isfinite_ra).sum(), (~isfinite_dec).sum()
    oor_ra_low,  oor_ra_high  = (ra64 < 0.0).sum(), (ra64 >= 360.0).sum()
    oor_dec_low, oor_dec_high = (dec64 < -90.0).sum(), (dec64 > 90.0).sum()
    invalid_total = invalid_mask.sum()

    (na_ra, na_dec, nonfinite_ra, nonfinite_dec,
     oor_ra_low, oor_ra_high, oor_dec_low, oor_dec_high, invalid_total) = dask.compute(
        na_ra, na_dec, nonfinite_ra, nonfinite_dec,
        oor_ra_low, oor_ra_high, oor_dec_low, oor_dec_high, invalid_total
    )

    if invalid_total > 0:
        cols = [c for c in ["CRD_ID", "id", "source", "survey", "ra", "dec"] if c in df.columns]
        sample_records = df[invalid_mask][cols].head(5, compute=True).to_dict(orient="records")
        raise ValueError(
            f"[{product_name}] Found invalid RA/DEC rows before DP1 flagging: {invalid_total}\n"
            f"  - RA NaN: {na_ra}, DEC NaN: {na_dec}\n"
            f"  - RA non-finite (±inf): {nonfinite_ra}, DEC non-finite (±inf): {nonfinite_dec}\n"
            f"  - RA out-of-range (<0): {oor_ra_low}, RA out-of-range (>=360): {oor_ra_high}\n"
            f"  - DEC out-of-range (<-90): {oor_dec_low}, DEC out-of-range (>90): {oor_dec_high}\n"
            f"  - Sample of bad rows (up to 5): {sample_records}"
        )


# -----------------------
# DP1 flagging
# -----------------------
def _flag_dp1(df: dd.DataFrame) -> dd.DataFrame:
    """Flag rows within predefined DP1 circular fields.

    Geometry: spherical law of cosines with clipping before arccos.

    Args:
      df: DataFrame with columns 'ra' and 'dec' in degrees.

    Returns:
      dd.DataFrame: Input plus 'is_in_DP1_fields' (0/1) as nullable int dtype.
    """
    ra_centers  = np.deg2rad([r[0] for r in DP1_REGIONS])
    dec_centers = np.deg2rad([r[1] for r in DP1_REGIONS])
    radii       = [r[2] for r in DP1_REGIONS]

    def _compute(part: pd.DataFrame) -> pd.DataFrame:
        p = part.copy()
        ra_rad  = np.deg2rad(p["ra"].to_numpy(dtype=float, copy=False))
        dec_rad = np.deg2rad(p["dec"].to_numpy(dtype=float, copy=False))
        in_any = np.zeros(len(p), dtype=bool)
        for ra_c, dec_c, rdeg in zip(ra_centers, dec_centers, radii):
            cos_ang = (np.sin(dec_c) * np.sin(dec_rad) +
                       np.cos(dec_c) * np.cos(dec_rad) * np.cos(ra_rad - ra_c))
            ang_deg = np.rad2deg(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
            in_any |= (ang_deg <= rdeg)
        # Use Arrow/int nullable dtype if configured.
        p["is_in_DP1_fields"] = pd.Series(in_any, index=p.index, dtype=DTYPE_INT)
        return p

    # Accurate meta with desired dtype for the new column.
    meta = df._meta.assign(is_in_DP1_fields=pd.Series(pd.array([], dtype=DTYPE_INT)))
    return df.map_partitions(_compute, meta=meta)

# -----------------------
# Column selection
# -----------------------
def _extract_variables_from_expr(expr: str) -> set[str]:
    """Extract variable names referenced in a boolean/arithmetic expression.

    Args:
      expr: Expression string (e.g., "0 < z_err < 5e-4 and len(flag) > 1").

    Returns:
      Set of identifier names referenced in the expression.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return set()

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.vars = set()
        def visit_Name(self, node: ast.Name) -> None:  # type: ignore[override]
            self.vars.add(node.id)
            self.generic_visit(node)

    v = _Visitor()
    v.visit(tree)
    return v.vars


def _select_output_columns(
    df: dd.DataFrame,
    translation_rules_uc: dict,
    tiebreaking_priority: list,
    used_type_fastpath: bool,
    save_expr_columns: bool = False,
    schema_hints: dict | None = None,
) -> dd.DataFrame:
    """Assemble final output schema and coerce optional expr columns.

    Args:
      df: Frame after tie-breaking and DP1 flagging.
      translation_rules_uc: Upper-cased translation rules (for expr var discovery).
      tiebreaking_priority: Priority columns to append if present.
      used_type_fastpath: If True, copy normalized `type` into `instrument_type`.
      save_expr_columns: Whether to keep variables used in YAML expressions.
      schema_hints: Normalized hints {'int','float','str','bool'} for expr cols.

    Returns:
      Subset of `df` with deterministic column order and coerced dtypes.
    """
    final_cols = [
        "CRD_ID", "id", "ra", "dec", "z", "z_flag", "z_err",
        "instrument_type", "survey", "source", "tie_result", "is_in_DP1_fields",
    ]
    if "z_flag_homogenized" in df.columns:
        final_cols.append("z_flag_homogenized")
    if "instrument_type_homogenized" in df.columns:
        final_cols.append("instrument_type_homogenized")

    # Preserve normalized string semantics when fast-path was used.
    if used_type_fastpath and "type" in df.columns:
        df["instrument_type"] = df["type"].astype(DTYPE_STR)

    # Include tiebreaking columns if present.
    extra = [c for c in tiebreaking_priority if c not in final_cols and c in df.columns]
    final_cols += extra

    # Determine additional columns referenced by YAML expressions.
    extra_expr_cols = set()
    if save_expr_columns:
        for ruleset in translation_rules_uc.values():
            for key in ["z_flag_translation", "instrument_type_translation"]:
                rule = ruleset.get(key, {})
                for cond in rule.get("conditions", []):
                    expr = cond.get("expr", "")
                    vars_in_expr = _extract_variables_from_expr(expr)
                    extra_expr_cols.update({v for v in vars_in_expr if v in df.columns})

    standard = {"id", "ra", "dec", "z", "z_flag", "z_err", "instrument_type", "survey"}
    already = set(final_cols)
    needed = [c for c in extra_expr_cols if c not in standard and c not in already]
    if save_expr_columns:
        final_cols += needed

    # Coerce hinted expr columns to requested dtypes.
    schema_hints = schema_hints or {}
    if save_expr_columns and schema_hints:
        target_cols = [c for c in needed if c in schema_hints]
        for col in target_cols:
            kind = schema_hints[col]
            if col in df.columns:
                if kind == "str":
                    df[col] = df[col].map_partitions(
                        _normalize_string_series_to_na,
                        meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
                    )
                elif kind == "float":
                    coerced = dd.to_numeric(df[col], errors="coerce")
                    df[col] = coerced.map_partitions(
                        lambda s: s.astype(DTYPE_FLOAT),
                        meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
                    )
                elif kind == "int":
                    coerced = dd.to_numeric(df[col], errors="coerce")
                    df[col] = coerced.map_partitions(
                        lambda s: s.astype(DTYPE_INT),
                        meta=pd.Series(pd.array([], dtype=DTYPE_INT)),
                    )
                elif kind == "bool":
                    df[col] = df[col].map_partitions(
                        _to_nullable_boolean_strict,
                        meta=pd.Series(pd.array([], dtype=DTYPE_BOOL)),
                    )
            else:
                # Create hinted column with <NA> to stabilize schema.
                if kind == "str":
                    df = _add_missing_with_dtype(df, col, DTYPE_STR)
                elif kind == "float":
                    df = _add_missing_with_dtype(df, col, DTYPE_FLOAT)
                elif kind == "int":
                    df = _add_missing_with_dtype(df, col, DTYPE_INT)
                elif kind == "bool":
                    df = _add_missing_with_dtype(df, col, DTYPE_BOOL)

    # Deduplicate while preserving order, then subset.
    final_cols = list(dict.fromkeys(final_cols))
    df = df[[c for c in final_cols if c in df.columns]]
    return df


# -----------------------
# Save parquet + HATS/margin
# -----------------------
def _save_parquet(df: dd.DataFrame, temp_dir: str, product_name: str) -> str:
    """Write a partitioned Parquet artifact for the prepared catalog.

    Output layout:
        ``<temp_dir>/prepared_<product_name>/`` with one or more files produced
        by ``Dask.to_parquet(..., engine="pyarrow")``.

    Args:
        df (dd.DataFrame): Prepared dataframe to persist.
        temp_dir (str): Base directory for temporary artifacts.
        product_name (str): Product internal name.

    Returns:
        str: Absolute path to the output directory containing Parquet files.
    """
    out_path = os.path.join(temp_dir, f"prepared_{product_name}")
    logger = logging.getLogger(LOGGER_NAME)

    protected = {
        "CRD_ID", "id", "ra", "dec", "z", "z_err", "z_flag",
        "instrument_type", "survey", "source",
        "z_flag_homogenized", "instrument_type_homogenized",
        "tie_result", "is_in_DP1_fields",
    }

    df = _rename_duplicate_columns_dd(df, logger, protected=protected)

    with dask.config.set({"dataframe.shuffle.method": "tasks"}):
        df.to_parquet(out_path, write_index=False, engine="pyarrow")
    return out_path


# ---- Helper: build a stable Arrow schema for our prepared catalog ----
def _build_arrow_schema_for_catalog(df: pd.DataFrame) -> pa.Schema:
    """Build a robust pyarrow Schema for LSDB.from_dataframe.

    We declare explicit Arrow types for all standard columns we may output,
    so per-pixel empty partitions don’t degrade to `null[pyarrow]`.

    Args:
        df: Prepared pandas DataFrame (pandas>=2 with ArrowDtype preferred).

    Returns:
        pyarrow.Schema with appropriate nullability for each column.
    """
    # Canonical types we expect in the prepared parquet
    canonical = {
        # core ids / strings
        "CRD_ID": pa.string(),
        "id": pa.string(),
        "source": pa.string(),
        "survey": pa.string(),
        "instrument_type": pa.string(),
        "instrument_type_homogenized": pa.string(),
        # coordinates / redshift
        "ra": pa.float64(),
        "dec": pa.float64(),
        "z": pa.float64(),
        "z_err": pa.float64(),
        "z_flag": pa.float64(),
        "z_flag_homogenized": pa.float64(),
        # flags / results
        "tie_result": pa.int8(),
        "is_in_DP1_fields": pa.int64(),
    }

    fields = []
    for col in df.columns:
        # Prefer our canonical mapping when available
        if col in canonical:
            fields.append(pa.field(col, canonical[col], nullable=True))
            continue

        # Otherwise, derive a reasonable Arrow type from pandas dtype
        dt = df[col].dtype

        # Pandas ArrowDtype -> use its underlying pyarrow dtype
        if isinstance(dt, pd.ArrowDtype):
            fields.append(pa.field(col, dt.pyarrow_dtype, nullable=True))
            continue

        # Pandas extension dtypes
        s = str(dt)
        if s.startswith("Float") or s == "float64":
            fields.append(pa.field(col, pa.float64(), nullable=True))
        elif s.startswith("Int") or s in {"int64", "int32", "int16", "int8"}:
            # default to int64 unless clearly int8 already
            arr_type = pa.int8() if "8" in s else pa.int64()
            fields.append(pa.field(col, arr_type, nullable=True))
        elif s in {"boolean", "bool"}:
            fields.append(pa.field(col, pa.bool_(), nullable=True))
        else:
            # object, string, unknown -> string
            fields.append(pa.field(col, pa.string(), nullable=True))

    # Deduplicate by column name preserving order
    seen = set()
    uniq_fields = []
    for f in fields:
        if f.name not in seen:
            uniq_fields.append(f)
            seen.add(f.name)

    return pa.schema(uniq_fields)


def import_catalog(
    path: str,
    ra_col: str,
    dec_col: str,
    artifact_name: str,
    output_path: str,
    logs_dir: str,
    logger: logging.Logger,
    client,  # type: ignore[override]
    size_threshold_mb: int = 500,
) -> None:
    """Import a Parquet catalog into HATS format.

    Tries a fast path via `lsdb.from_dataframe(..., schema=...)` for small
    catalogs, providing an explicit pyarrow Schema to avoid meta drift
    (e.g., `null[pyarrow]` vs `string[pyarrow]`). On any error, falls back to
    the distributed HATS import (`pipeline_with_client`) with robust shuffle.

    Args:
        path: Directory produced by `_save_parquet` (or any folder with parquet).
        ra_col: RA column name.
        dec_col: DEC column name.
        artifact_name: Output HATS artifact name (folder).
        output_path: Destination root directory.
        logs_dir: Logs directory (informational).
        logger: Logger instance.
        client: Dask client (required for the distributed fallback).
        size_threshold_mb: Upper bound to try the fast path.
    """
    # Discover parquet files (supports optional 'base/' subdir)
    base_path = os.path.join(path, "base")
    parquet_files = glob.glob(os.path.join(base_path, "*.parquet")) if os.path.exists(base_path) \
        else glob.glob(os.path.join(path, "*.parquet"))
    if not parquet_files:
        raise ValueError(f"No Parquet files found at {path}")

    total_size_mb = sum(os.path.getsize(f) for f in parquet_files) / 1024**2
    hats_path = os.path.join(output_path, artifact_name)

    # ---- FAST PATH (small datasets): pandas -> LSDB.from_dataframe with explicit schema
    if total_size_mb <= size_threshold_mb:
        try:
            logger.info(f"⚡ Small catalog detected ({total_size_mb:.1f} MB). Trying fast `from_dataframe` with schema.")
            # Load to pandas (Arrow-backed dtypes if available)
            dfp = pd.read_parquet(parquet_files)

            dfp = _rename_duplicate_columns_pd(dfp, logger)

            # Enforce canonical dtypes on critical columns before building schema
            # (this ensures consistency even if a later pixel slice is all-NA)
            for col, pd_dtype in [
                ("CRD_ID", DTYPE_STR),
                ("id", DTYPE_STR),
                ("source", DTYPE_STR),
                ("survey", DTYPE_STR),
                ("instrument_type", DTYPE_STR),
                ("instrument_type_homogenized", DTYPE_STR),
                ("ra", DTYPE_FLOAT),
                ("dec", DTYPE_FLOAT),
                ("z", DTYPE_FLOAT),
                ("z_err", DTYPE_FLOAT),
                ("z_flag", DTYPE_FLOAT),
                ("z_flag_homogenized", DTYPE_FLOAT),
                ("tie_result", DTYPE_INT8),
                ("is_in_DP1_fields", DTYPE_INT),
            ]:
                if col in dfp.columns:
                    try:
                        dfp[col] = dfp[col].astype(pd_dtype)
                    except Exception:
                        # If cast fails (e.g., values out of range), leave as-is; schema will still force Arrow type
                        pass

            # Build an explicit Arrow schema from the *current* dfp columns
            schema = _build_arrow_schema_for_catalog(dfp)

            # Create LSDB Catalog with explicit schema and write to HATS
            catalog = lsdb.from_dataframe(
                dfp,
                catalog_name=artifact_name,
                ra_column=ra_col,
                dec_column=dec_col,
                use_pyarrow_types=True,
                schema=schema,
            )
            catalog.to_hats(hats_path, overwrite=True)
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: import_catalog id={artifact_name} (fast path)")
            return
        except Exception as e:
            # Fall back to distributed pipeline if any meta/schema drift occurs
            logger.warning(f"⚠️ Fast path failed for {artifact_name} ({type(e).__name__}: {e}). Falling back to distributed import.")

    # ---- DISTRIBUTED FALLBACK: hats-import pipeline (robust)
    if client is None:
        raise ValueError("❌ Dask client is required to import catalog into HATS (distributed fallback).")

    logger.info("🧱 Using distributed HATS import (pipeline_with_client).")
    file_reader = ParquetReader()
    args = ImportArguments(
        ra_column=ra_col,
        dec_column=dec_col,
        input_file_list=parquet_files,
        file_reader=file_reader,
        output_artifact_name=artifact_name,
        output_path=output_path,
    )

    # Use a robust shuffle for HPC environments (avoid p2p here)
    with dask.config.set({"dataframe.shuffle.method": "tasks"}):
        pipeline_with_client(args, client)

    logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: import_catalog id={artifact_name} (distributed)")


def generate_margin_cache_safe(
    hats_path: str,
    output_path: str,
    artifact_name: str,
    logs_dir: str,
    logger: logging.Logger,
    client,  # type: ignore[override]
) -> str | None:
    """Generate a margin cache for a HATS catalog when beneficial.

    Behavior:
      - If the catalog has more than one partition, runs the margin cache
        pipeline; otherwise, logs and returns ``None``.
      - If a previous run left a broken state (``intermediate/`` without a
        ``margin_pair.csv``), the directory is removed and recomputed.
      - If the HATS reader reports “no rows” for the margin, the function
        logs and returns ``None`` instead of raising.

    Args:
        hats_path (str): Path to the HATS catalog.
        output_path (str): Directory where the margin cache folder is written.
        artifact_name (str): Name of the margin cache artifact (folder).
        logs_dir (str): Directory for logs (not directly used here).
        logger (logging.Logger): Logger for progress messages.
        client: Dask client.

    Returns:
        str | None: Path to the margin cache folder, or ``None`` if skipped.

    Raises:
        OSError: If deletion of a corrupted margin directory fails.
        ValueError: Re-raised for unexpected errors during HATS reading.
    """
    margin_dir = os.path.join(output_path, artifact_name)
    intermediate_dir = os.path.join(margin_dir, "intermediate")
    critical_file = os.path.join(intermediate_dir, "margin_pair.csv")

    # Clean up broken resumption state if detected.
    if os.path.exists(intermediate_dir) and not os.path.exists(critical_file):
        logger.info(f"⚠️ Detected incomplete margin cache at {margin_dir}. Deleting to force regeneration...")
        try:
            shutil.rmtree(margin_dir)
        except Exception as e:
            logger.info(f"❌ Failed to delete corrupted margin cache at {margin_dir}: {e}")
            raise

    try:
        catalog = hats.read_hats(hats_path)
        info = catalog.partition_info.as_dataframe().astype(int)
        if len(info) > 1:
            args = MarginCacheArguments(
                input_catalog_path=hats_path,
                output_path=output_path,
                margin_threshold=1.0,
                output_artifact_name=artifact_name,
            )
            pipeline_with_client(args, client)
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: generate_margin_cache id={artifact_name}")
            return margin_dir
        else:
            logger.info(f"⚠️ Margin cache skipped: single partition for {artifact_name}")
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: generate_margin_cache id={artifact_name}")
            return None
    except ValueError as e:
        if "Margin cache contains no rows" in str(e):
            logger.info(f"⚠️ {e} Proceeding without margin cache.")
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: generate_margin_cache id={artifact_name}")
            return None
        else:
            raise


def _maybe_hats_and_margin(
    out_path: str,
    product_name: str,
    logs_dir: str,
    temp_dir: str,
    logger: logging.Logger,
    client,  # type: ignore[override]
    combine_mode: str,
) -> tuple[str, str]:
    """Optionally build HATS and margin cache artifacts depending on mode.

    If ``combine_mode != "concatenate"``, this function:
      1) Parses the numeric prefix from ``product_name`` to name artifacts.
      2) Imports the prepared Parquet into HATS (``cat{prefix}_hats``).
      3) Runs the margin cache pipeline (``cat{prefix}_margin``) if beneficial.

    Args:
        out_path (str): Path returned by :func:`_save_parquet`.
        product_name (str): Product internal name (must start with digits + underscore).
        logs_dir (str): Directory for logs (not directly used here).
        temp_dir (str): Directory for output artifacts.
        logger (logging.Logger): Logger.
        client: Dask client.
        combine_mode (str): One of
            ``"concatenate"``,
            ``"concatenate_and_mark_duplicates"``,
            ``"concatenate_and_remove_duplicates"``.

    Returns:
        tuple[str, str]: ``(hats_path, margin_cache_path)``. Note that the
        margin path is returned even if the cache was skipped; callers should
        check existence if they need to read from it.

    Raises:
        ValueError: If ``product_name`` lacks a numeric prefix or if ``client``
            is missing when ``combine_mode`` requires HATS/margin generation.
    """
    hats_path, margin_cache_path = "", ""
    if combine_mode != "concatenate":
        if client is None:
            raise ValueError(
                f"❌ Dask client is required for combine_mode='{combine_mode}' "
                f"(needed to run import_catalog and generate_margin_cache)."
            )
        m = re.match(r"(\d+)_", product_name)
        if not m:
            raise ValueError(f"❌ Could not extract numeric prefix from internal_name '{product_name}'")
        prefix = m.group(1)
        art_hats, art_margin = f"cat{prefix}_hats", f"cat{prefix}_margin"

        import_catalog(out_path, "ra", "dec", art_hats, temp_dir, logs_dir, logger, client)
        generate_margin_cache_safe(os.path.join(temp_dir, art_hats), temp_dir, art_margin, logs_dir, logger, client)

        hats_path = os.path.join(temp_dir, art_hats)
        margin_cache_path = os.path.join(temp_dir, art_margin)
    return hats_path, margin_cache_path

# =======================
# Main orchestrator
# =======================
def prepare_catalog(
    entry: dict,
    translation_config: dict,
    logs_dir: str,
    temp_dir: str,
    combine_mode: str = "concatenate_and_mark_duplicates",
) -> tuple[str, str, str, str, str, str]:
    """Prepare a spectroscopic catalog for the CRC pipeline.

    High-level flow:
      1) Load product via :class:`ProductHandle` → Dask DataFrame.
      2) YAML validation & safe renaming (base schema enforced).
      3) Respect user-provided homogenized fields if mapped in YAML.
      4) Normalize dtypes/values; map survey-specific z_flag encodings.
      5) Generate stable ``CRD_ID`` and initialize ``tie_result=1``.
      6) Compute homogenized columns.
      7) **Persist + repartition** to stabilize memory and improve shuffle throughput.
      8) Apply tie-breaking (if enabled), collecting duplicate edges.
      9) Strict RA/DEC validation and DP1 field flagging.
      10) Select final columns and write prepared Parquet (coalesced partitions).
      11) Optionally build HATS + margin cache artifacts.

    Performance notes:
      * Persisting after homogenization pins the cleaned dataframe in worker
        memory and avoids re-reading from storage during the upcoming shuffle.
      * Repartitioning to medium-large partitions (~256MB) balances throughput
        and memory pressure for your workers (25 cores, 50GB).

    Args:
        entry: YAML node for the product. Required keys:
            - ``internal_name`` (str)
            - ``path`` (str or structured path for ProductHandle)
            - ``columns`` (mapping of standard->source; nulls allowed)
        translation_config: Configuration with translation rules and tie-breaking
            settings (e.g., ``tiebreaking_priority``, ``instrument_type_priority``,
            ``delta_z_threshold``, optional ``save_expr_columns`` and
            ``expr_column_schema``).
        logs_dir: Directory for log files.
        temp_dir: Directory for temporary and output artifacts.
        combine_mode: One of
            ``"concatenate"``,
            ``"concatenate_and_mark_duplicates"``,
            ``"concatenate_and_remove_duplicates"``.

    Returns:
        (hats_path, ra_col, dec_col, product_name, margin_cache_path, compared_to_path)

    Raises:
        ValueError: On malformed YAML mapping, missing required columns, invalid
            tie-breaking configuration, or invalid coordinates.
    """
    from dask.distributed import get_client as _get_client, wait  # local import

    client = _get_client()

    product_name = entry["internal_name"]
    logger = _build_logger(logs_dir, LOGGER_NAME, "prepare_all.log")
    logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: prepare_catalog id={product_name}")

    # 1) Load product into a Dask DataFrame.
    ph = ProductHandle(entry["path"])
    df = ph.to_ddf()

    # 2) Validate & rename according to YAML mapping; enforce base schema.
    df = _validate_and_rename(df, entry, logger)

    # 3) Honor already-homogenized columns declared via YAML (if any).
    df = _honor_user_homogenized_mapping(df, entry, product_name, logger)

    # 4) Normalize dtypes/values (strings, floats, optional `type`, and z_flag mapping).
    df, type_cast_ok = _normalize_types(df, product_name, logger)

    # 5) Assign CRD_IDs and init tie bookkeeping.
    df, compared_to_path = _generate_crd_ids(df, product_name, temp_dir)
    df["tie_result"] = 1
    compared_to_dict_solo: dict[str, set[str]] = defaultdict(set)

    # 6) Compute homogenized fields (vectorized translations / fast-paths).
    df, used_type_fastpath, tiebreaking_priority, instrument_type_priority, translation_rules_uc = _homogenize(
        df, translation_config, product_name, logger, type_cast_ok=type_cast_ok
    )

    # 7) Persist + repartition BEFORE the tie-breaking shuffle.
    part_size = "256MB"  # hardcoded: good default for 25 cores / 50GB workers
    try:
        df = df.persist(); wait(df)
        with dask.config.set({"dataframe.shuffle.method": "tasks"}):
            df = df.repartition(partition_size=part_size)
        logger.info(f"{product_name} persisted and repartitioned with partition_size={part_size}. "
                    f"npartitions={df.npartitions}")
    except Exception as e:
        logger.warning(f"⚠️ {product_name} persist/repartition step failed or was skipped: {e}")

    # 8) Resolve duplicates and collect pair links (if requested).
    if combine_mode in ["concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"]:
        df = _apply_tiebreaking_and_collect(
            df,
            translation_config,
            tiebreaking_priority,
            instrument_type_priority,
            compared_to_dict_solo,
            product_name,
            logger,
        )

    # Persist the local `compared_to` graph even if empty (diagnostics/repro).
    with open(compared_to_path, "w") as f:
        json.dump({k: sorted(v) for k, v in compared_to_dict_solo.items()}, f)

    # 9) Validate geometry early to fail fast on malformed inputs.
    _validate_ra_dec_or_fail(df, product_name)

    # 10) Flag DP1 fields.
    df = _flag_dp1(df)

    # 11) Assemble final columns.
    df = _select_output_columns(
        df,
        translation_rules_uc,
        tiebreaking_priority,
        used_type_fastpath,
        save_expr_columns=translation_config.get("save_expr_columns", False),
        schema_hints=_normalize_schema_hints(translation_config.get("expr_column_schema")),
    )

    # Coalesce partitions before writing to reduce small-file counts and speed up HATS import.
    try:
        nworkers = len(client.scheduler_info()["workers"]) or 1
    except Exception:
        nworkers = 1
    target_out_parts = max(2 * nworkers, 8)
    try:
        with dask.config.set({"dataframe.shuffle.method": "tasks"}):
            df_to_write = df.repartition(npartitions=target_out_parts)
        logger.info(f"{product_name} output repartitioned for write: npartitions={df_to_write.npartitions}")
    except Exception as e:
        logger.warning(f"⚠️ {product_name} output repartition failed; writing with current npartitions. Reason: {e}")
        df_to_write = df

    # 12) Write prepared parquet.
    out_path = _save_parquet(df_to_write, temp_dir, product_name)

    # 13) Optionally produce HATS + margin artifacts.
    hats_path, margin_cache_path = _maybe_hats_and_margin(
        out_path, product_name, logs_dir, temp_dir, logger, client, combine_mode
    )

    logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: prepare_catalog id={product_name}")
    return hats_path, "ra", "dec", product_name, margin_cache_path, compared_to_path
