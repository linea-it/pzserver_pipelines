# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Combine Redshift Catalogs – catalog preparation.

Loads a raw spectroscopic catalog, validates/normalizes schema, derives
standardized fields (IDs, homogenized flags, DP1 flags), optionally imports
HATS and generates margin cache, and writes a prepared Parquet artifact.

Public API:
    - prepare_catalog(entry, translation_config, logs_dir, temp_dir, combine_mode)
"""

# -----------------------
# Standard library
# -----------------------
import ast as _ast
import builtins
import difflib
import glob
import logging
import math
import os
import pathlib
import re
import shutil
from datetime import datetime
from typing import TYPE_CHECKING, Any

# -----------------------
# Third-party
# -----------------------
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import get_client as _get_client, wait

dask.config.set({"dataframe.shuffle.method": "tasks"})
os.environ.setdefault("DASK_DISTRIBUTED__SHUFFLE__METHOD", "tasks")

# -----------------------
# Project (lsdb/hats/CRC)
# -----------------------
import hats  # noqa: F401
import lsdb
from combine_redshift_dedup.packages.product_handle import ProductHandle
from combine_redshift_dedup.packages.utils import ensure_crc_logger
from combine_redshift_dedup.packages.specz_homogenization import (
    JADES_LETTER_TO_SCORE,
    VIMOS_FLAG_TO_SCORE,
    _honor_user_homogenized_mapping,
    _homogenize,
)
from hats_import import CollectionArguments
from hats_import.collection.run_import import run as run_collection_import

if TYPE_CHECKING:
    from dask.distributed import Client  # noqa: F401

# -----------------------
# Arrow-backed dtypes (pandas >= 2.x)
# -----------------------
import pyarrow as pa
import pyarrow.parquet as pq

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

LOGGER_NAME = "crc.specz"

DP1_REGIONS = [
    (6.02, -72.08, 2.5),   # 47 Tuc
    (37.86, 6.98, 2.5),    # Rubin SV 38 7
    (40.00, -34.45, 2.5),  # Fornax dSph
    (53.13, -28.10, 2.5),  # ECDFS
    (59.10, -48.73, 2.5),  # EDFS
    (95.00, -25.00, 2.5),  # Rubin SV 95 -25
    (106.23, -10.51, 2.5), # Seagull
]

# -----------------------
# Centralized logging
# -----------------------
def _get_logger() -> logging.Logger:
    """Return a child logger that propagates to the root 'crc' logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    return logger


def _phase_logger(base_logger: logging.Logger, phase: str, product: str | None = None) -> logging.LoggerAdapter:
    """Return a LoggerAdapter injecting phase (and product if provided)."""
    extra = {"phase": phase}
    if product:
        extra["product"] = product
    return logging.LoggerAdapter(base_logger, extra)


# -----------------------
# YAML mapping validation
# -----------------------
def _validate_and_rename(df: dd.DataFrame, entry: dict, logger: logging.Logger) -> dd.DataFrame:
    """Validate YAML column mapping and apply conflict-safe renames.

    Verifies existence of non-null source columns, suggests close matches,
    and safely parks conflicting targets as ``<target>__origN`` before renaming.

    Args:
        df: Input Dask DataFrame.
        entry: YAML entry with ``internal_name`` and ``columns`` mapping.
        logger: Logger.

    Returns:
        dd.DataFrame: DataFrame with validated/renamed columns and base schema.

    Raises:
        ValueError: If required source columns are missing.
    """
    product_name = entry["internal_name"]
    columns_cfg = (entry.get("columns") or {})
    non_null_map = {std: src for std, src in columns_cfg.items() if src not in (None, "", "null")}
    input_cols = list(map(str, df.columns))

    missing_sources = [src for src in non_null_map.values() if src not in input_cols]
    if missing_sources:
        suggestions = {src: difflib.get_close_matches(src, input_cols, n=3, cutoff=0.6) for src in missing_sources}
        raise ValueError(
            f"[{product_name}] Missing mapped source columns in input parquet: {missing_sources}\n"
            f"Configured (non-null) mapping: {non_null_map}\n"
            f"Closest matches: {suggestions}\n"
            f"Available columns (sample): {sorted(input_cols)[:30]} ..."
        )

    # Resolve collisions: if target exists and differs from its source, park it aside.
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
                f"{product_name} Resolve rename collision: '{tgt}' already exists; "
                f"'{tgt}' → '{parked}' before mapping '{src}'→'{tgt}'"
            )
            df = df.rename(columns={tgt: parked})

    col_map = {src: std for std, src in non_null_map.items()}
    if col_map:
        logger.info(f"{product_name} Rename map (sample up to 6): {list(col_map.items())[:6]}")
        df = df.rename(columns=col_map)
    else:
        logger.info(f"{product_name} No non-null column mappings; skipping rename.")

    # Tag source and ensure minimal base schema
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


def _rename_duplicate_columns_dd(df: dd.DataFrame, logger: logging.Logger) -> dd.DataFrame:
    """Make column names unique across partitions by appending __dupN.

    Args:
        df: Input Dask DataFrame.
        logger: Logger.

    Returns:
        dd.DataFrame: DataFrame with unique column names.
    """
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
            logger.warning(
                f"Renamed duplicate columns in partition: {sample}"
                f"{' ...' if len(renamed_pairs) > 6 else ''}"
            )

        out = pdf.copy()
        out.columns = new_cols
        return out

    meta_fixed = _renamer(df._meta)
    return df.map_partitions(_renamer, meta=meta_fixed)


def _rename_duplicate_columns_pd(pdf: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Make pandas columns unique by appending __dupN.

    Args:
        pdf: Input pandas DataFrame.
        logger: Logger.

    Returns:
        pd.DataFrame: DataFrame with unique column names.
    """
    cols = pd.Index(map(str, pdf.columns))
    if cols.has_duplicates:
        seen: dict[str, int] = {}
        new_cols: list[str] = []
        renamed: list[tuple[str, str]] = []

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
            sample = ", ".join([f"{a}->{b}" for a, b in renamed[:6]])
            logger.warning(
                f"Renamed duplicate columns (pandas): {sample}"
                f"{' ...' if len(renamed) > 6 else ''}"
            )

        pdf = pdf.copy()
        pdf.columns = new_cols

    return pdf


def _next_available_prev_name(existing: list[str], base: str) -> str:
    """Return a unique previous-result column name based on base.

    Args:
        existing: Current columns.
        base: Desired base name.

    Returns:
        Unique column name.
    """
    if base not in existing:
        return base
    i = 2
    while f"{base}{i}" in existing:
        i += 1
    return f"{base}{i}"


def _stash_previous_results(df: dd.DataFrame, entry: dict, logger: logging.Logger) -> dd.DataFrame:
    """Stash previous-run results into CRD_ID_prev*/compared_to_prev*/group_id_prev*.

    Args:
        df: Frame after rename.
        entry: YAML node for product.
        logger: Logger.

    Returns:
        dd.DataFrame: Frame with stashed previous columns when applicable.
    """
    cols = list(map(str, df.columns))
    columns_cfg = (entry.get("columns") or {})
    non_null_map = {str(std): str(src) for std, src in columns_cfg.items()
                    if src not in (None, "", "null")}

    mapped_id_from_crd = str(non_null_map.get("id", "")).strip().lower() == "crd_id"

    # Handle YAML-mapped id → CRD_ID
    if mapped_id_from_crd and "id" in cols:
        new_name = _next_available_prev_name(cols, "CRD_ID_prev")
        logger.info(f"Stash CRD_ID from YAML-mapped 'id' → {new_name} (keep 'id')")
        df[new_name] = df["id"]
        cols.append(new_name)

    # Handle CRD_ID
    if "CRD_ID" in cols:
        new_name = _next_available_prev_name(cols, "CRD_ID_prev")
        if new_name != "CRD_ID":
            logger.info(f"Stash previous CRD_ID → {new_name}")
            df = df.rename(columns={"CRD_ID": new_name})
            cols.append(new_name)

    # Handle compared_to
    if "compared_to" in cols:
        new_name = _next_available_prev_name(cols, "compared_to_prev")
        if new_name != "compared_to":
            logger.info(f"Stash previous compared_to → {new_name}")
            df = df.rename(columns={"compared_to": new_name})
            cols.append(new_name)

    # Handle group_id
    if "group_id" in cols:
        new_name = _next_available_prev_name(cols, "group_id_prev")
        if new_name != "group_id":
            logger.info(f"Stash previous group_id → {new_name}")
            df = df.rename(columns={"group_id": new_name})
            cols.append(new_name)

    return df


# -----------------------
# Type helpers
# -----------------------
def _add_missing_with_dtype(_df: dd.DataFrame, col: str, pd_dtype: Any) -> dd.DataFrame:
    """Add missing column with a specific dtype and valid Dask meta.

    Args:
        _df: Input DataFrame.
        col: Column to add.
        pd_dtype: Target pandas/Arrow dtype.

    Returns:
        dd.DataFrame: Frame with column added (if missing).
    """
    if col in _df.columns:
        return _df

    meta_added = _df._meta.assign(**{col: pd.Series(pd.array([], dtype=pd_dtype))})

    def _adder(part: pd.DataFrame) -> pd.DataFrame:
        p = part.copy()
        p[col] = pd.Series(pd.NA, index=p.index, dtype=pd_dtype)
        return p

    return _df.map_partitions(_adder, meta=meta_added)


def _normalize_string_series_to_na(s: pd.Series) -> pd.Series:
    """Normalize to string dtype and coerce placeholders to <NA>.

    Args:
        s: Input series.

    Returns:
        pd.Series: Normalized string series.
    """
    s = s.astype(DTYPE_STR).str.strip()
    low = s.fillna("").str.lower()
    mask_empty = (low == "") | low.isin(["none", "null", "nan"])
    return s.mask(mask_empty, pd.NA).astype(DTYPE_STR)


def _to_nullable_boolean_strict(s: pd.Series) -> pd.Series:
    """Convert to nullable boolean strictly (non-bool → <NA>).

    Args:
        s: Input series.

    Returns:
        Nullable boolean series.
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
        hints: Mapping column -> dtype.

    Returns:
        dict: Normalized hints.
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

    Args:
        df: Frame after YAML rename.
        product_name: Catalog identifier.
        logger: Logger.

    Returns:
        Tuple[dd.DataFrame, bool]: (normalized frame, whether 'type' normalized).
    """
    # 1) Normalize string-like
    string_like = ["id", "instrument_type", "survey", "instrument_type_homogenized", "source"]
    for col in string_like:
        if col in df.columns:
            df[col] = df[col].map_partitions(
                _normalize_string_series_to_na,
                meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
            )
            if col == "survey":
                df[col] = df[col].str.upper()

    # 2) Optional 'type'
    type_cast_ok = False
    if "type" in df.columns:
        try:
            df["type"] = df["type"].map_partitions(
                _normalize_string_series_to_na,
                meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
            ).str.lower()
            type_cast_ok = True
        except Exception as e:
            logger.warning(f"{product_name} Failed to normalize 'type' to lower-case: {e}")
            type_cast_ok = False

    # 3) JADES/VIMOS z_flag mapping if needed
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

    # 4) Float-like coercion
    float_like = ["ra", "dec", "z", "z_err", "z_flag", "z_flag_homogenized"]
    for col in float_like:
        if col in df.columns:
            coerced = dd.to_numeric(df[col], errors="coerce")
            invalid_mask = dd.isna(coerced) & ~dd.isna(df[col])
            invalid_count = dask.compute(invalid_mask.sum())[0]
            if invalid_count > 0:
                sample_vals = df[col].loc[invalid_mask].head(5, compute=True).tolist()
                raise ValueError(
                    f"[{product_name}] Failed to convert '{col}' to numeric: "
                    f"{invalid_count} non-numeric value(s). Examples: {sample_vals}."
                )
            df[col] = coerced.map_partitions(
                lambda s: s.astype(DTYPE_FLOAT),
                meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
            )

    return df, type_cast_ok


# -----------------------
# CRD_ID generation
# -----------------------
def _generate_crd_ids(df: dd.DataFrame, product_name: str, temp_dir: str) -> dd.DataFrame:
    """Assign stable, catalog-scoped CRD_IDs.

    Args:
        df: Input frame after schema normalization.
        product_name: Internal name (expects numeric prefix before underscore).
        temp_dir: Unused, kept for signature stability.

    Returns:
        dd.DataFrame: Frame with CRD_ID column (Arrow string dtype).

    Raises:
        ValueError: If numeric prefix cannot be extracted from product_name.
    """
    m = re.match(r"(\d+)_", product_name)
    if not m:
        raise ValueError(f"Could not extract numeric prefix from internal_name '{product_name}'")
    catalog_prefix = m.group(1)

    sizes = df.map_partitions(len).compute().tolist()
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)

    def _add_crd(part: pd.DataFrame, start: int) -> pd.DataFrame:
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
    return df


# -----------------------
# RA/DEC strict validation (fail fast)
# -----------------------
def _validate_ra_dec_or_fail(df: dd.DataFrame, product_name: str) -> None:
    """Validate RA/DEC finiteness and ranges; raise on invalid rows.

    Args:
        df: Frame with 'ra' and 'dec'.
        product_name: Catalog identifier.

    Raises:
        ValueError: If invalid RA/DEC entries are detected.
    """
    def _isfinite_series(s: pd.Series) -> pd.Series:
        arr = s.astype("float64")
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
            f"[{product_name}] Invalid RA/DEC rows: {invalid_total}\n"
            f"  RA NaN={na_ra}, DEC NaN={na_dec}\n"
            f"  RA non-finite={nonfinite_ra}, DEC non-finite={nonfinite_dec}\n"
            f"  RA <0={oor_ra_low}, RA >=360={oor_ra_high}\n"
            f"  DEC <-90={oor_dec_low}, DEC >90={oor_dec_high}\n"
            f"  Sample (up to 5): {sample_records}"
        )


# -----------------------
# DP1 flagging
# -----------------------
def _flag_dp1(df: dd.DataFrame) -> dd.DataFrame:
    """Flag rows within predefined DP1 circular fields.

    Args:
        df: Frame with 'ra' and 'dec' (degrees).

    Returns:
        dd.DataFrame: Adds 'is_in_DP1_fields' (nullable int).
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
        p["is_in_DP1_fields"] = pd.Series(in_any, index=p.index, dtype=DTYPE_INT)
        return p

    meta = df._meta.assign(is_in_DP1_fields=pd.Series(pd.array([], dtype=DTYPE_INT)))
    return df.map_partitions(_compute, meta=meta)


# -----------------------
# Column selection
# -----------------------
def _extract_variables_from_expr(expr: str) -> set[str]:
    """Extract variable names referenced in an expression.

    Args:
        expr: Expression string.

    Returns:
        Set of variable names.
    """
    try:
        tree = _ast.parse(expr, mode="eval")
    except Exception:
        return set()

    class _Visitor(_ast.NodeVisitor):
        def __init__(self):
            self.vars = set()
        def visit_Name(self, node: _ast.Name) -> None:  # type: ignore[override]
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
    """Assemble final output schema and coerce optional expression columns.

    Args:
      df: Frame after tie-breaking and DP1 flagging.
      translation_rules_uc: Upper-cased translation rules.
      tiebreaking_priority: Priority columns to append if present.
      used_type_fastpath: Whether `type` was reused for instrument_type.
      save_expr_columns: Keep variables used in YAML expressions.
      schema_hints: Normalized hints {'int','float','str','bool'}.

    Returns:
      dd.DataFrame: Subset with deterministic column order.
    """
    # Base schema (only columns that exist will be kept at the end).
    final_cols = [
        "CRD_ID", "id", "ra", "dec", "z", "z_flag", "z_err",
        "instrument_type", "survey", "source", "tie_result",
        "is_in_DP1_fields", "compared_to",
        # New optional current component label (will only be kept if present)
        "group_id",
    ]

    # Optional homogenized fields.
    if "z_flag_homogenized" in df.columns:
        final_cols.append("z_flag_homogenized")
    if "instrument_type_homogenized" in df.columns:
        final_cols.append("instrument_type_homogenized")

    # Normalize compared_to to nullable string if present.
    if "compared_to" in df.columns:
        df["compared_to"] = df["compared_to"].map_partitions(
            _normalize_string_series_to_na,
            meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
        )

    # Fast path: instrument_type <- type
    if used_type_fastpath and "type" in df.columns:
        df["instrument_type"] = df["type"].astype(DTYPE_STR)

    # Add tiebreaking columns if present and not already included.
    extra = [c for c in tiebreaking_priority if c not in final_cols and c in df.columns]
    final_cols += extra

    # Collect variables referenced in YAML expressions (if requested).
    extra_expr_cols = set()
    if save_expr_columns:
        for ruleset in translation_rules_uc.values():
            for key in ["z_flag_translation", "instrument_type_translation"]:
                rule = ruleset.get(key, {})
                for cond in rule.get("conditions", []):
                    expr = cond.get("expr", "")
                    vars_in_expr = _extract_variables_from_expr(expr)
                    extra_expr_cols.update({v for v in vars_in_expr if v in df.columns})

    # Keep expression vars that are not already standard/final.
    standard = {"id", "ra", "dec", "z", "z_flag", "z_err", "instrument_type", "survey"}
    already = set(final_cols)
    needed = [c for c in extra_expr_cols if c not in standard and c not in already]
    if save_expr_columns:
        final_cols += needed

    # --- Include *_prev columns if present (now also for group_id). ---
    prev_like = [c for c in df.columns if str(c).startswith("CRD_ID_prev")]
    prev_like += [c for c in df.columns if str(c).startswith("compared_to_prev")]
    prev_like += [c for c in df.columns if str(c).startswith("group_id_prev")]  # NEW
    for c in prev_like:
        if c not in final_cols:
            final_cols.append(c)

    # Optional dtype coercions for expression vars (guided by schema_hints).
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
                # Create missing columns with requested dtype.
                if kind == "str":
                    df = _add_missing_with_dtype(df, col, DTYPE_STR)
                elif kind == "float":
                    df = _add_missing_with_dtype(df, col, DTYPE_FLOAT)
                elif kind == "int":
                    df = _add_missing_with_dtype(df, col, DTYPE_INT)
                elif kind == "bool":
                    df = _add_missing_with_dtype(df, col, DTYPE_BOOL)

    # De-duplicate while preserving order, then filter to existing columns.
    final_cols = list(dict.fromkeys(final_cols))
    df = df[[c for c in final_cols if c in df.columns]]
    return df


# -----------------------
# Save parquet
# -----------------------
def _save_parquet(df: dd.DataFrame, temp_dir: str, product_name: str) -> str:
    """Write a partitioned Parquet artifact for the prepared catalog.

    Args:
        df: Prepared dataframe.
        temp_dir: Base temp directory.
        product_name: Internal name.

    Returns:
        str: Output directory path with Parquet files.
    """
    out_path = os.path.join(temp_dir, f"prepared_{product_name}")
    logger = logging.getLogger(LOGGER_NAME)

    df = _rename_duplicate_columns_dd(df, logger)
    with dask.config.set({"dataframe.shuffle.method": "tasks"}):
        df.to_parquet(out_path, write_index=False, engine="pyarrow")
    return out_path


# ---- Arrow schema builder ----
def _build_arrow_schema_for_catalog(df: pd.DataFrame, schema_hints: dict | None = None) -> pa.Schema:
    """Build a robust Arrow schema for lsdb.from_dataframe.

    Args:
        df: Sample pandas frame to infer types.
        schema_hints: Optional hints for expression columns.

    Returns:
        pa.Schema: Arrow schema.
    """
    canonical = {
        "CRD_ID": pa.string(),
        "id": pa.string(),
        "source": pa.string(),
        "survey": pa.string(),
        "instrument_type": pa.string(),
        "instrument_type_homogenized": pa.string(),
        "ra": pa.float64(),
        "dec": pa.float64(),
        "z": pa.float64(),
        "z_err": pa.float64(),
        "z_flag": pa.float64(),
        "z_flag_homogenized": pa.float64(),
        "tie_result": pa.int8(),
        "is_in_DP1_fields": pa.int64(),
        "compared_to": pa.string(),
    }

    def _hint_to_pa(dt: str) -> pa.DataType:
        k = str(dt).lower()
        if k == "str":
            return pa.string()
        if k == "float":
            return pa.float64()
        if k == "int":
            return pa.int64()
        if k == "bool":
            return pa.bool_()
        return pa.string()

    schema_hints = schema_hints or {}
    fields: list[pa.Field] = []

    for col in df.columns:
        name = str(col)

        if name in canonical:
            fields.append(pa.field(name, canonical[name], nullable=True))
            continue

        if name.startswith("CRD_ID_prev") or name.startswith("compared_to_prev"):
            fields.append(pa.field(name, pa.string(), nullable=True))
            continue

        if name in schema_hints:
            fields.append(pa.field(name, _hint_to_pa(schema_hints[name]), nullable=True))
            continue

        dt = df[name].dtype
        if isinstance(dt, pd.ArrowDtype):
            fields.append(pa.field(name, dt.pyarrow_dtype, nullable=True))
            continue

        s = str(dt)
        if s.startswith("Float") or s == "float64":
            fields.append(pa.field(name, pa.float64(), nullable=True))
        elif s.startswith("Int") or s in {"int64", "int32", "int16", "int8"}:
            fields.append(pa.field(name, pa.int64() if "8" not in s else pa.int8(), nullable=True))
        elif s in {"boolean", "bool"}:
            fields.append(pa.field(name, pa.bool_(), nullable=True))
        else:
            fields.append(pa.field(name, pa.string(), nullable=True))

    seen = set()
    uniq_fields = []
    for f in fields:
        if f.name not in seen:
            uniq_fields.append(f)
            seen.add(f.name)

    return pa.schema(uniq_fields)


# =======================
# Collections
# =======================
def _write_schema_file_for_collection(
    parquet_path: str,
    schema_out_path: str,
    logger: logging.Logger,
    schema_hints: dict | None = None,
) -> str:
    """Create a minimal Parquet file containing only the desired Arrow schema.

    Args:
        parquet_path: Directory with prepared Parquet files.
        schema_out_path: Destination file for schema-only Parquet.
        logger: Logger.
        schema_hints: Optional expr hints.

    Returns:
        str: Path to the written schema file.

    Raises:
        ValueError: If no Parquet files are found.
    """
    base_path = os.path.join(parquet_path, "base")
    pattern = os.path.join(base_path, "*.parquet") if os.path.exists(base_path) \
        else os.path.join(parquet_path, "*.parquet")
    files = glob.glob(pattern)
    files.sort()
    if not files:
        raise ValueError(f"No Parquet files found at '{parquet_path}' to infer schema")

    sample_df = pd.read_parquet(files[0])
    sample_df = _rename_duplicate_columns_pd(sample_df, logger)

    schema = _build_arrow_schema_for_catalog(sample_df, schema_hints=schema_hints or {})

    os.makedirs(os.path.dirname(schema_out_path), exist_ok=True)
    empty_tbl = pa.Table.from_arrays(
        [pa.array([], type=f.type) for f in schema],
        names=[f.name for f in schema],
    )
    pq.write_table(empty_tbl, schema_out_path)

    logger.info(f"Wrote schema file for collection: {schema_out_path}")
    return schema_out_path


def _build_collection_with_retry(
    parquet_path: str,
    logs_dir: str,
    logger: logging.Logger,
    client,
    try_margin: bool = True,
    *,
    schema_hints: dict | None = None,
    size_threshold_mb: int = 200,
) -> str:
    """Build a HATS Collection from a Parquet folder.

    Args:
        parquet_path: Prepared parquet path (e.g., .../merged_stepX or prepared_*).
        logs_dir: Logs directory (unused here, kept for parity).
        logger: Logger.
        client: Dask client or None.
        try_margin: Try margin import first.
        schema_hints: Expr hints for importer schema.
        size_threshold_mb: Fast-path threshold.

    Returns:
        str: Collection path written as "<parquet_path>_hats".

    Raises:
        RuntimeError: If both fast-path and importer fallback fail.
    """
    parquet_path = os.path.normpath(parquet_path)
    parent_dir = os.path.dirname(parquet_path) or "."
    base_name = os.path.basename(parquet_path)
    output_artifact_name = f"{base_name}_hats"
    collection_path = os.path.join(parent_dir, output_artifact_name)

    base_path = os.path.join(parquet_path, "base")
    pattern = os.path.join(base_path, "*.parquet") if os.path.exists(base_path) \
        else os.path.join(parquet_path, "*.parquet")
    in_file_paths = glob.glob(pattern)
    in_file_paths.sort()
    if not in_file_paths:
        raise ValueError(f"No Parquet files found at '{parquet_path}'")

    try:
        total_size_mb = sum(os.path.getsize(f) for f in in_file_paths) / 1024**2
    except Exception:
        total_size_mb = float("inf")

    # FAST PATH
    if total_size_mb <= size_threshold_mb:
        try:
            logger.info(f"Small catalog ({total_size_mb:.1f} MB). Building collection via fast path → {collection_path}")
            pdf_list = [pd.read_parquet(p) for p in in_file_paths]
            dfp = pd.concat(pdf_list, ignore_index=True) if len(pdf_list) > 1 else pdf_list[0]
            dfp = _rename_duplicate_columns_pd(dfp, logger)

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
                ("compared_to", DTYPE_STR),
            ]:
                if col in dfp.columns:
                    try:
                        dfp[col] = dfp[col].astype(pd_dtype)
                    except Exception:
                        pass

            for c in map(str, dfp.columns):
                if c.startswith("CRD_ID_prev") or c.startswith("compared_to_prev"):
                    try:
                        dfp[c] = dfp[c].astype(DTYPE_STR)
                    except Exception:
                        pass

            schema = _build_arrow_schema_for_catalog(dfp, schema_hints=_normalize_schema_hints(schema_hints or {}))

            catalog = lsdb.from_dataframe(
                dfp,
                catalog_name=output_artifact_name,
                ra_column="ra",
                dec_column="dec",
                use_pyarrow_types=True,
                schema=schema,
            )
            catalog.to_hats(collection_path, as_collection=True, overwrite=True)

            logger.info(f"Finished collection fast-path: {collection_path}")
            return collection_path

        except Exception as e:
            logger.warning(f"Collection fast-path failed for {output_artifact_name} ({type(e).__name__}: {e}). Falling back to hats_import.")

    # FALLBACK
    def _clean_partial():
        try:
            if os.path.isdir(collection_path):
                shutil.rmtree(collection_path)
        except Exception as e:
            logger.warning(f"Failed to remove partial '{collection_path}': {e}")

    schema_file = os.path.join(parent_dir, f"{output_artifact_name}_schema.parquet")
    try:
        _write_schema_file_for_collection(
            parquet_path=parquet_path,
            schema_out_path=schema_file,
            logger=logger,
            schema_hints=_normalize_schema_hints(schema_hints or {}),
        )
    except Exception as e:
        logger.warning(f"Could not build schema file for fallback import ({type(e).__name__}: {e}). Proceeding without it.")
        schema_file = None

    def _make_args(with_margin: bool):
        kw = dict(
            input_file_list=in_file_paths,
            file_reader="parquet",
            ra_column="ra",
            dec_column="dec",
        )
        if schema_file:
            kw["use_schema_file"] = schema_file
        args = (CollectionArguments(
            output_artifact_name=output_artifact_name,
            output_path=parent_dir,
            resume=False,
        ).catalog(**kw))
        if with_margin:
            args = args.add_margin(margin_threshold=5.0, is_default=True)
        return args

    if try_margin:
        try:
            logger.info(f"Building collection WITH margin (import pipeline): {output_artifact_name}")
            run_collection_import(_make_args(with_margin=True), client)
            return collection_path
        except Exception as e:
            logger.warning(f"WITH margin failed: {e}. Retrying WITHOUT margin...")
            _clean_partial()

    try:
        logger.info(f"Building collection WITHOUT margin (import pipeline): {output_artifact_name}")
        run_collection_import(_make_args(with_margin=False), client)
        return collection_path
    except Exception as e:
        _clean_partial()
        raise RuntimeError(f"Failed to build collection '{output_artifact_name}': {e}")


def _maybe_collection(
    out_path: str,
    logs_dir: str,
    logger: logging.Logger,
    client,
    combine_mode: str,
    *,
    schema_hints: dict | None = None,
    size_threshold_mb: int = 200,
) -> str:
    """Optionally build a Collection from prepared Parquet.

    Args:
        out_path: Prepared parquet path.
        logs_dir: Logs directory.
        logger: Logger.
        client: Dask client.
        combine_mode: Combine mode.
        schema_hints: Expr hints.
        size_threshold_mb: Fast-path threshold.

    Returns:
        str: Collection path or empty string for concatenate mode.
    """
    if combine_mode == "concatenate":
        return ""
    return _build_collection_with_retry(
        parquet_path=out_path,
        logs_dir=logs_dir,
        logger=logger,
        client=client,
        try_margin=True,
        schema_hints=schema_hints,
        size_threshold_mb=size_threshold_mb,
    )


# =======================
# Main orchestrator
# =======================
def prepare_catalog(
    entry: dict,
    translation_config: dict,
    param_config: dict,
    logs_dir: str,
    temp_dir: str,
    combine_mode: str = "concatenate_and_mark_duplicates",
) -> tuple[str, str, str, str, str]:
    """Prepare a spectroscopic catalog for the CRC pipeline.

    Args:
        entry: Product descriptor with keys like {"path", "internal_name", ...}.
        translation_config: YAML-derived rules.
        logs_dir: Logs directory.
        temp_dir: Temp workspace for artifacts.
        combine_mode: Combine strategy (kept for compatibility).

    Returns:
        Tuple[str, str, str, str, str]: (collection_path, "ra", "dec", internal_name, "").
    """
    # Ensure central logger in this process (worker-safe)
    try:
        ensure_crc_logger(logs_dir)
    except Exception:
        pass

    try:
        client = _get_client()
    except Exception:
        client = None

    product_name = entry["internal_name"]
    base_logger = _get_logger()
    lg = _phase_logger(base_logger, phase="preparation", product=product_name)

    # ===== START PHASE (per catalog) =====
    lg.info(f"START prepare_catalog product={product_name}")

    # 1) Load product
    ph = ProductHandle(entry["path"])
    df = ph.to_ddf()

    # 2) Validate & rename, base schema
    df = _validate_and_rename(df, entry, lg)
    df = _stash_previous_results(df, entry, lg)

    # 3) Honor user-provided homogenized columns
    df = _honor_user_homogenized_mapping(df, entry, product_name, lg)

    # 4) Normalize types and values
    df, type_cast_ok = _normalize_types(df, product_name, lg)

    # 5) Assign CRD_IDs
    df = _generate_crd_ids(df, product_name, temp_dir)

    # 6) Homogenized fields
    (
        df,
        used_type_fastpath,
        tiebreaking_priority,
        instrument_type_priority,   # noqa: F841
        translation_rules_uc,
    ) = _homogenize(df, translation_config, product_name, lg, type_cast_ok=type_cast_ok)

    # 7) Apply cut based on z_flag_homogenized if requested
    z_flag_homogenized_value_to_cut = param_config.get("z_flag_homogenized_value_to_cut", None)
    if z_flag_homogenized_value_to_cut is not None and "z_flag_homogenized" in df.columns:
        try:
            cut_val = float(z_flag_homogenized_value_to_cut)
            if 0.0 < cut_val <= 4.0:
                initial_count = df.shape[0].compute()
                df = df[df["z_flag_homogenized"] >= cut_val]
                final_count = df.shape[0].compute()
                lg.info(f"Applied z_flag_homogenized cut >= {cut_val}: {initial_count} → {final_count} rows")
            else:
                lg.warning(f"Invalid z_flag_homogenized_value_to_cut={z_flag_homogenized_value_to_cut}; skipping cut.")
        except Exception as e:
            lg.warning(f"Could not apply z_flag_homogenized cut ({e}); skipping cut.")

    # 8) Persist + repartition
    part_size = "256MB"
    try:
        if client is not None:
            df = df.persist()
            wait(df)
        with dask.config.set({"dataframe.shuffle.method": "tasks"}):
            df = df.repartition(partition_size=part_size)
        lg.info(f"Persisted and repartitioned: partition_size={part_size} npartitions={df.npartitions}")
    except Exception as e:
        lg.warning(f"Persist/repartition skipped or failed: {e}")

    # 9) Init compared_to/tie_result
    df = _add_missing_with_dtype(df, "compared_to", DTYPE_STR)
    df = _add_missing_with_dtype(df, "tie_result", DTYPE_INT8)
    df["tie_result"] = df["tie_result"].map_partitions(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(1).astype(DTYPE_INT8),
        meta=pd.Series(pd.array([], dtype=DTYPE_INT8)),
    )

    # 10) Geometry validation
    _validate_ra_dec_or_fail(df, product_name)

    # 11) DP1 flag
    df = _flag_dp1(df)

    # 12) Assemble final columns
    df = _select_output_columns(
        df,
        translation_rules_uc,
        tiebreaking_priority,
        used_type_fastpath,
        save_expr_columns=translation_config.get("save_expr_columns", False),
        schema_hints=_normalize_schema_hints(translation_config.get("expr_column_schema")),
    )

    # Coalesce partitions for write
    try:
        if client is not None:
            nworkers = max(len(client.scheduler_info().get("workers", {})), 1)
        else:
            nworkers = 1
    except Exception:
        nworkers = 1
    target_out_parts = max(2 * nworkers, 8)
    try:
        with dask.config.set({"dataframe.shuffle.method": "tasks"}):
            df_to_write = df.repartition(npartitions=target_out_parts)
        lg.info(f"Output repartitioned for write: npartitions={df_to_write.npartitions}")
    except Exception as e:
        lg.warning(f"Output repartition failed; writing current npartitions. Reason: {e}")
        df_to_write = df

    # 13) Write prepared parquet
    out_path = _save_parquet(df_to_write, temp_dir, product_name)

    # 14) Build collection (always)
    schema_hints_raw = translation_config.get("expr_column_schema")
    collection_path = _build_collection_with_retry(
        parquet_path=out_path,
        logs_dir=logs_dir,
        logger=lg,
        client=client,
        try_margin=True,
        schema_hints=schema_hints_raw,
        size_threshold_mb=200,
    )

    # ===== END PHASE (per catalog) =====
    lg.info(f"END prepare_catalog product={product_name} path={collection_path}")

    return collection_path, "ra", "dec", product_name, ""
