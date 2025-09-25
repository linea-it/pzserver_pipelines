from __future__ import annotations
"""
Crossmatch and `compared_to` updater for CRC.

This module supports two export backends, controlled by the hardcoded flag
`USE_LSDB_CONCAT`:

- If USE_LSDB_CONCAT = True (default):
    Performs the crossmatch; updates `compared_to`; normalizes dtypes on each
    input catalog independently; concatenates at the LSDB catalog level;
    writes a HATS collection via `to_hats`; returns the collection path.
    (No intermediate Parquet.)

- If USE_LSDB_CONCAT = False:
    Performs the crossmatch; updates `compared_to`; extracts the underlying
    Dask DataFrames (`._ddf`) from the two catalogs; concatenates them with
    Dask; normalizes dtypes on the merged Dask DataFrame (expected, prev, expr);
    writes a Parquet dataset; imports it as an LSDB collection (margin-first
    retry); returns the collection path.

Public API:
    - crossmatch_tiebreak(...)
    - crossmatch_tiebreak_safe(...)
"""

# -----------------------
# Backend switch
# -----------------------
USE_LSDB_CONCAT: bool = False  # Toggle behavior as described above.

# -----------------------
# Standard library
# -----------------------
import os
import time
import logging
from typing import Dict, Iterable, List, Set

# -----------------------
# Third-party
# -----------------------
import numpy as np
import pandas as pd
import dask.dataframe as dd
import lsdb

# -----------------------
# Project
# -----------------------
from combine_redshift_dedup.packages.utils import get_phase_logger
from combine_redshift_dedup.packages.specz import (
    _build_collection_with_retry,
    _normalize_string_series_to_na,
    _add_missing_with_dtype,
    DTYPE_STR, DTYPE_FLOAT, DTYPE_INT, DTYPE_BOOL, DTYPE_INT8,
)

__all__ = ["crossmatch_tiebreak", "crossmatch_tiebreak_safe"]

LOGGER_NAME = "crc.crossmatch"  # child of the pipeline root logger ("crc")


# -----------------------
# Centralized logging
# -----------------------
def _get_logger() -> logging.LoggerAdapter:
    """Return a phase-aware logger ('crc.crossmatch' with phase='crossmatch')."""
    base = logging.getLogger(LOGGER_NAME)
    base.propagate = True
    return get_phase_logger("crossmatch", base)


# -----------------------
# Utilities
# -----------------------
def _adjacency_from_pairs(left_ids: pd.Series, right_ids: pd.Series) -> Dict[str, Set[str]]:
    """Build an undirected adjacency from left-right crossmatch pairs."""
    adj: Dict[str, Set[str]] = {}
    L = left_ids.astype(str).to_numpy(dtype=object, copy=False)
    R = right_ids.astype(str).to_numpy(dtype=object, copy=False)
    get = adj.get
    for a, b in zip(L, R):
        s = get(a)
        if s is None:
            adj[a] = {b}
        else:
            s.add(b)
        s = get(b)
        if s is None:
            adj[b] = {a}
        else:
            s.add(a)
    return adj


def _merge_compared_to_partition(
    part: pd.DataFrame,
    pairs_adj: Dict[str, Iterable[str]],
) -> pd.DataFrame:
    """Partition-wise update of `compared_to` by unioning existing entries with new pairs."""
    p = part.copy()

    if "compared_to" not in p.columns:
        p["compared_to"] = pd.Series(pd.array([pd.NA] * len(p), dtype=DTYPE_STR), index=p.index)

    crd_list: List[str] = p["CRD_ID"].astype(str).tolist()

    def _norm_token(x) -> str | None:
        if pd.isna(x):
            return None
        if isinstance(x, (bool, np.bool_)):
            return None
        s = str(x).strip()
        if not s or s == "<NA>":
            return None
        return s

    def _to_str_set(it: Iterable) -> Set[str]:
        out: Set[str] = set()
        if it is None:
            return out
        for x in it:
            s = _norm_token(x)
            if s is not None:
                out.add(s)
        return out

    def _parse_existing(val) -> Set[str]:
        if pd.isna(val):
            return set()
        if isinstance(val, str):
            return _to_str_set(t.strip() for t in val.split(","))
        if isinstance(val, (list, set, tuple)):
            return _to_str_set(val)
        return _to_str_set([val])

    # Build NEW neighbor sets, parse OLD cells, then union per row.
    new_sets: List[Set[str]] = [_to_str_set(pairs_adj.get(k, ())) for k in crd_list]
    old_sets: List[Set[str]] = [_parse_existing(v) for v in p["compared_to"].tolist()]

    merged_vals: List[object] = []
    for k, old_set, new_set in zip(crd_list, old_sets, new_sets):
        nxt = set().union(old_set, new_set)
        nxt.discard(k)
        merged_vals.append(", ".join(sorted(nxt)) if nxt else pd.NA)

    p["compared_to"] = pd.Series(pd.array(merged_vals, dtype=DTYPE_STR), index=p.index)
    return p


# -----------------------
# Dtype normalization helpers (catalog-level, via map_partitions)
# -----------------------
_EXPECTED_TYPES = {
    "CRD_ID": DTYPE_STR,
    "id": DTYPE_STR,
    "ra": DTYPE_FLOAT,
    "dec": DTYPE_FLOAT,
    "z": DTYPE_FLOAT,
    "z_flag": DTYPE_FLOAT,
    "z_err": DTYPE_FLOAT,
    "instrument_type": DTYPE_STR,
    "survey": DTYPE_STR,
    "source": DTYPE_STR,
    "tie_result": DTYPE_INT8,
    "z_flag_homogenized": DTYPE_FLOAT,
    "instrument_type_homogenized": DTYPE_STR,
    "compared_to": DTYPE_STR,
}


def _cast_partition_expected(
    part: pd.DataFrame,
    expected_types: dict,
    schema_hints: dict | None,
) -> pd.DataFrame:
    """Cast/add core columns and optional expr columns on a pandas partition."""
    df = part.copy()

    # 1) Core expected types
    for col, dtype in expected_types.items():
        if col not in df.columns:
            # Create missing column with target dtype
            if dtype is DTYPE_STR:
                df[col] = pd.Series(pd.array([pd.NA] * len(df), dtype=DTYPE_STR), index=df.index)
            elif dtype is DTYPE_FLOAT:
                df[col] = pd.Series(pd.array([np.nan] * len(df), dtype=DTYPE_FLOAT), index=df.index)
            elif dtype is DTYPE_INT8:
                df[col] = pd.Series(pd.array([pd.NA] * len(df), dtype=DTYPE_INT8), index=df.index)
            elif dtype is DTYPE_INT:
                df[col] = pd.Series(pd.array([pd.NA] * len(df), dtype=DTYPE_INT), index=df.index)
            elif dtype is DTYPE_BOOL:
                df[col] = pd.Series(pd.array([pd.NA] * len(df), dtype=DTYPE_BOOL), index=df.index)
            continue

        # Cast existing column
        try:
            if dtype is DTYPE_STR:
                df[col] = _normalize_string_series_to_na(df[col])
            elif dtype is DTYPE_FLOAT:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPE_FLOAT)
            elif dtype is DTYPE_INT8:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPE_INT8)
            elif dtype is DTYPE_INT:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPE_INT)
            elif dtype is DTYPE_BOOL:
                df[col] = df[col].astype(DTYPE_BOOL)
        except Exception:
            # Be lenient: if cast fails, leave column as-is.
            pass

    # 2) Prev-like columns
    prev_like_str = [c for c in df.columns if str(c).startswith("CRD_ID_prev")]
    prev_like_str += [c for c in df.columns if str(c).startswith("compared_to_prev")]
    for c in prev_like_str:
        try:
            df[c] = _normalize_string_series_to_na(df[c])
        except Exception:
            pass

    prev_like_int = [c for c in df.columns if str(c).startswith("group_id_prev")]
    for c in prev_like_int:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(DTYPE_INT)
        except Exception:
            pass

    # 3) Optional expr columns guided by hints (if provided)
    hints = dict(schema_hints or {})
    for col, kind in hints.items():
        k = str(kind).lower()
        if col not in df.columns:
            # Create missing column with target dtype
            try:
                if k == "str":
                    df[col] = pd.Series(pd.array([pd.NA] * len(df), dtype=DTYPE_STR), index=df.index)
                elif k == "float":
                    df[col] = pd.Series(pd.array([np.nan] * len(df), dtype=DTYPE_FLOAT), index=df.index)
                elif k == "int":
                    df[col] = pd.Series(pd.array([pd.NA] * len(df), dtype=DTYPE_INT), index=df.index)
                elif k == "bool":
                    df[col] = pd.Series(pd.array([pd.NA] * len(df), dtype=DTYPE_BOOL), index=df.index)
            except Exception:
                pass
            continue

        # Cast existing
        try:
            if k == "str":
                df[col] = _normalize_string_series_to_na(df[col])
            elif k == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPE_FLOAT)
            elif k == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPE_INT)
            elif k == "bool":
                df[col] = df[col].astype(DTYPE_BOOL)
        except Exception:
            pass

    return df


def _normalize_catalog_dtypes(cat, translation_config: dict | None):
    """Return a new catalog with normalized dtypes (lazy, via map_partitions)."""
    cfg = translation_config or {}
    schema_hints_local = {} if (cfg.get("save_expr_columns") is False) else (cfg.get("expr_column_schema", {}) or {})

    # Build meta by running the caster on the meta dataframe
    meta_in = cat._ddf._meta
    meta_out = _cast_partition_expected(meta_in, _EXPECTED_TYPES, schema_hints_local)
    return cat.map_partitions(_cast_partition_expected, _EXPECTED_TYPES, schema_hints_local, meta=meta_out)


# -----------------------
# Dask/DDFrame normalization helpers (for the legacy path)
# -----------------------
def _coerce_optional_columns_for_import(
    df: dd.DataFrame,
    schema_hints: dict | None = None,
) -> dd.DataFrame:
    """Coerce prev/expr columns to consistent Arrow dtypes across partitions."""
    # 1) Prev columns → normalize
    prev_like_str = [c for c in df.columns if str(c).startswith("CRD_ID_prev")]
    prev_like_str += [c for c in df.columns if str(c).startswith("compared_to_prev")]
    for c in prev_like_str:
        df[c] = df[c].map_partitions(
            _normalize_string_series_to_na,
            meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
        )

    prev_like_int = [c for c in df.columns if str(c).startswith("group_id_prev")]
    for c in prev_like_int:
        coerced = dd.to_numeric(df[c], errors="coerce")
        df[c] = coerced.map_partitions(
            lambda s: s.astype(DTYPE_INT),
            meta=pd.Series(pd.array([], dtype=DTYPE_INT)),
        )

    # 2) Expr columns guided by hints
    hints = dict(schema_hints or {})
    if not hints:
        return df

    for col, kind in hints.items():
        k = str(kind).lower()
        if col not in df.columns:
            # Create missing with target dtype (avoid null[pyarrow] metas)
            if k == "str":
                df = _add_missing_with_dtype(df, col, DTYPE_STR)
            elif k == "float":
                df = _add_missing_with_dtype(df, col, DTYPE_FLOAT)
            elif k == "int":
                df = _add_missing_with_dtype(df, col, DTYPE_INT)
            elif k == "bool":
                df = _add_missing_with_dtype(df, col, DTYPE_BOOL)
            continue

        # Cast existing
        if k == "str":
            df[col] = df[col].map_partitions(
                _normalize_string_series_to_na,
                meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
            )
        elif k == "float":
            coerced = dd.to_numeric(df[col], errors="coerce")
            df[col] = coerced.map_partitions(
                lambda s: s.astype(DTYPE_FLOAT),
                meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
            )
        elif k == "int":
            coerced = dd.to_numeric(df[col], errors="coerce")
            df[col] = coerced.map_partitions(
                lambda s: s.astype(DTYPE_INT),
                meta=pd.Series(pd.array([], dtype=DTYPE_INT)),
            )
        elif k == "bool":
            df[col] = df[col].map_partitions(
                lambda s: s.astype(DTYPE_BOOL),
                meta=pd.Series(pd.array([], dtype=DTYPE_BOOL)),
            )

    return df


def _normalize_ddf_expected_types(merged: dd.DataFrame, translation_config: dict | None) -> dd.DataFrame:
    """Normalize expected/core columns on a merged Dask DataFrame prior to import."""
    expected_types = dict(_EXPECTED_TYPES)

    # 1) Core expected types
    for col, dtype in expected_types.items():
        if col not in merged.columns:
            continue
        try:
            if dtype == DTYPE_STR:
                merged[col] = merged[col].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
                )
            elif dtype is DTYPE_FLOAT:
                merged[col] = dd.to_numeric(merged[col], errors="coerce").map_partitions(
                    lambda s: s.astype(DTYPE_FLOAT),
                    meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
                )
            elif dtype is DTYPE_INT8:
                merged[col] = dd.to_numeric(merged[col], errors="coerce").map_partitions(
                    lambda s: s.astype(DTYPE_INT8),
                    meta=pd.Series(pd.array([], dtype=DTYPE_INT8)),
                )
            elif dtype is DTYPE_INT:
                merged[col] = dd.to_numeric(merged[col], errors="coerce").map_partitions(
                    lambda s: s.astype(DTYPE_INT),
                    meta=pd.Series(pd.array([], dtype=DTYPE_INT)),
                )
            elif dtype is DTYPE_BOOL:
                merged[col] = merged[col].map_partitions(
                    lambda s: s.astype(DTYPE_BOOL),
                    meta=pd.Series(pd.array([], dtype=DTYPE_BOOL)),
                )
        except Exception:
            # Best-effort only
            pass

    # 2) Prev/expr columns
    cfg = translation_config or {}
    schema_hints_local = {} if (cfg.get("save_expr_columns") is False) else (cfg.get("expr_column_schema", {}) or {})
    merged = _coerce_optional_columns_for_import(merged, schema_hints_local)

    return merged


# -----------------------
# Parquet writer helper (legacy path)
# -----------------------
def _safe_to_parquet(ddf, path, **kwargs) -> None:
    """Write Parquet robustly for both plain Dask and nested_dask frames.

    Tries engine="pyarrow"; if backend already sets an engine, retries without it.
    """
    try:
        ddf.to_parquet(path, engine="pyarrow", **kwargs)
    except TypeError as e:
        if "multiple values for keyword argument 'engine'" in str(e):
            ddf.to_parquet(path, **kwargs)
        else:
            raise


# =======================
# Main logic
# =======================
def crossmatch_tiebreak(
    left_cat,
    right_cat,
    logs_dir: str,
    temp_dir: str,
    step,
    client,  # kept for signature compatibility (used in legacy import path)
    translation_config: dict | None = None,
    do_import: bool = True,  # kept for signature compatibility; ignored in both paths (we always return a collection)
) -> str:
    """Crossmatch two catalogs, update `compared_to`, then export to a collection.

    Behavior depends on USE_LSDB_CONCAT (see module docstring).
    """
    logger = _get_logger()
    t0_all = time.time()

    # Parameters with defaults
    radius = float((translation_config or {}).get("crossmatch_radius_arcsec", 0.75))
    k = int((translation_config or {}).get("crossmatch_n_neighbors", 10))

    logger.info(
        "START crossmatch_update_compared_to: step=%s radius=%.3f\" n_neighbors=%d backend=%s",
        step,
        radius,
        k,
        ("LSDB+to_hats" if USE_LSDB_CONCAT else "Dask+Parquet+import"),
    )

    # 1) Spatial crossmatch
    t0 = time.time()
    xmatched = left_cat.crossmatch(
        right_cat,
        radius_arcsec=radius,
        n_neighbors=k,
        suffixes=("left", "right"),
    )
    logger.info("Crossmatch done (%.2fs)", time.time() - t0)

    # 2) Build adjacency from CRD_ID pairs
    t0 = time.time()
    pair_cols = ["CRD_IDleft", "CRD_IDright"]
    pairs_df = xmatched._ddf[pair_cols].compute()
    if len(pairs_df) == 0:
        pairs_adj: Dict[str, Set[str]] = {}
        logger.info("No pairs found; `compared_to` remains unchanged.")
    else:
        pairs_df = pairs_df.astype({"CRD_IDleft": "string", "CRD_IDright": "string"})
        pairs_df = pairs_df[pairs_df["CRD_IDleft"] != pairs_df["CRD_IDright"]].drop_duplicates()
        pairs_adj = _adjacency_from_pairs(pairs_df["CRD_IDleft"], pairs_df["CRD_IDright"])
    total_links = sum(len(v) for v in pairs_adj.values())
    logger.info(
        "Adjacency built: links=%d nodes=%d (%.2fs)",
        total_links,
        len(pairs_adj),
        time.time() - t0,
    )

    # 3) Update `compared_to` on both catalogs, partition-wise
    t0 = time.time()

    def _ensure_meta(meta_df: pd.DataFrame) -> pd.DataFrame:
        m = meta_df.copy()
        if "compared_to" not in m.columns:
            m["compared_to"] = pd.Series(pd.array([], dtype=DTYPE_STR))
        else:
            m["compared_to"] = pd.Series(pd.array([], dtype=DTYPE_STR))
        return m

    left_meta = _ensure_meta(left_cat._ddf._meta)
    right_meta = _ensure_meta(right_cat._ddf._meta)

    left_updated = left_cat.map_partitions(_merge_compared_to_partition, pairs_adj, meta=left_meta)
    right_updated = right_cat.map_partitions(_merge_compared_to_partition, pairs_adj, meta=right_meta)
    logger.info("Compared_to updated on partitions (%.2fs)", time.time() - t0)

    # 4) Export path A: LSDB concat + to_hats
    if USE_LSDB_CONCAT:
        # Normalize dtypes on each catalog independently *before* concat
        t0 = time.time()
        left_fixed = _normalize_catalog_dtypes(left_updated, translation_config)
        right_fixed = _normalize_catalog_dtypes(right_updated, translation_config)
        logger.info("Per-catalog type normalization attached (lazy) (%.2fs)", time.time() - t0)

        # Concatenate at the LSDB catalog level
        t0 = time.time()
        merged_cat = left_fixed.concat(right_fixed)
        logger.info("LSDB concat done (%.2fs)", time.time() - t0)

        # Persist as HATS collection and return path
        t0 = time.time()
        collection_path = os.path.join(temp_dir, f"merged_step{step}_hats")
        merged_cat.to_hats(
            collection_path,
            as_collection=True,
            overwrite=True,
        )
        logger.info(
            "Write complete (to_hats): step=%s path=%s (%.2fs)",
            step,
            collection_path,
            time.time() - t0,
        )
        logger.info(
            "END crossmatch_update_compared_to: step=%s links=%d nodes=%d output=%s (%.2fs)",
            step,
            total_links,
            len(pairs_adj),
            collection_path,
            time.time() - t0_all,
        )
        return collection_path

    # 4b) Export path B (legacy): Dask concat + Parquet + import
    t0 = time.time()
    lddf = left_updated._ddf
    rddf = right_updated._ddf
    merged = dd.concat([lddf, rddf])
    merged = _normalize_ddf_expected_types(merged, translation_config)
    logger.info("Dask concat + type normalization (lazy) (%.2fs)", time.time() - t0)

    t0 = time.time()
    merged_path = os.path.join(temp_dir, f"merged_step{step}")
    _safe_to_parquet(merged, merged_path, write_index=False)
    logger.info("Parquet written: path=%s (%.2fs)", merged_path, time.time() - t0)

    # Import as Collection (margin-first, with optional schema hints)
    t0 = time.time()
    cfg = translation_config or {}
    schema_hints_local = {} if (cfg.get("save_expr_columns") is False) else (cfg.get("expr_column_schema", {}) or {})
    schema_hints = schema_hints_local if schema_hints_local else None

    logger.info("START import_collection: step=%s parquet=%s", step, merged_path)
    collection_path = _build_collection_with_retry(
        parquet_path=merged_path,
        logs_dir=logs_dir,
        logger=logger,
        client=client,
        try_margin=True,
        schema_hints=schema_hints,
    )
    logger.info("END import_collection: step=%s path=%s (%.2fs)", step, collection_path, time.time() - t0)
    logger.info(
        "END crossmatch_update_compared_to: step=%s links=%d nodes=%d output=%s (%.2fs)",
        step,
        total_links,
        len(pairs_adj),
        collection_path,
        time.time() - t0_all,
    )
    return collection_path


def crossmatch_tiebreak_safe(
    left_cat,
    right_cat,
    logs_dir: str,
    temp_dir: str,
    step,
    client,  # used in legacy import path
    translation_config: dict | None = None,
    do_import: bool = True,  # ignored; always returns a collection
) -> str:
    """Wrapper around `crossmatch_tiebreak` with graceful empty-overlap fallback.

    If the crossmatch yields a known empty-overlap condition:
      - LSDB path: ensure `compared_to`, normalize each catalog, LSDB concat,
        `to_hats`, return collection.
      - Legacy path: ensure `compared_to`, Dask concat, Parquet, import, return collection.
    """
    logger = _get_logger()
    t0_safe = time.time()
    logger.info(
        "START xmatch_update_compared_to_safe: step=%s backend=%s",
        step,
        ("LSDB+to_hats" if USE_LSDB_CONCAT else "Dask+Parquet+import"),
    )

    try:
        out = crossmatch_tiebreak(
            left_cat=left_cat,
            right_cat=right_cat,
            logs_dir=logs_dir,
            temp_dir=temp_dir,
            step=step,
            client=client,
            translation_config=translation_config,
            do_import=do_import,  # ignored internally
        )
        logger.info("END xmatch_update_compared_to_safe: step=%s output=%s (%.2fs)", step, out, time.time() - t0_safe)
        return out

    except RuntimeError as e:
        msg = str(e)
        if ("The output catalog is empty" in msg) or ("Catalogs do not overlap" in msg):
            logger.info("Empty-overlap condition detected: %s", msg)

            # Ensure `compared_to` exists on both sides
            def _ensure_compared_to(cat):
                def _ensure_col(part: pd.DataFrame) -> pd.DataFrame:
                    if "compared_to" not in part.columns:
                        part = part.copy()
                        part["compared_to"] = pd.Series(pd.array([pd.NA] * len(part), dtype=DTYPE_STR), index=part.index)
                    return part

                meta = cat._ddf._meta.copy()
                if "compared_to" not in meta.columns:
                    meta["compared_to"] = pd.Series(pd.array([], dtype=DTYPE_STR))
                return cat.map_partitions(_ensure_col, meta=meta)

            left_ready = _ensure_compared_to(left_cat)
            right_ready = _ensure_compared_to(right_cat)

            if USE_LSDB_CONCAT:
                # Normalize dtypes per catalog, then LSDB concat and write HATS
                left_fixed = _normalize_catalog_dtypes(left_ready, translation_config)
                right_fixed = _normalize_catalog_dtypes(right_ready, translation_config)

                merged_cat = left_fixed.concat(right_fixed)

                collection_path = os.path.join(temp_dir, f"merged_step{step}_hats")
                merged_cat.to_hats(
                    collection_path,
                    as_collection=True,
                    overwrite=True,
                )
                logger.info(
                    "END xmatch_update_compared_to_safe: step=%s output=%s (%.2fs)",
                    step,
                    collection_path,
                    time.time() - t0_safe,
                )
                return collection_path

            # Legacy path: Dask concat + Parquet + import
            lddf = left_ready._ddf
            rddf = right_ready._ddf
            merged = dd.concat([lddf, rddf])
            merged = _normalize_ddf_expected_types(merged, translation_config)

            merged_path = os.path.join(temp_dir, f"merged_step{step}")
            _safe_to_parquet(merged, merged_path, write_index=False)

            cfg = translation_config or {}
            schema_hints_local = {} if (cfg.get("save_expr_columns") is False) else (cfg.get("expr_column_schema", {}) or {})
            schema_hints = schema_hints_local if schema_hints_local else None

            collection_path = _build_collection_with_retry(
                parquet_path=merged_path,
                logs_dir=logs_dir,
                logger=logger,
                client=client,
                try_margin=True,
                schema_hints=schema_hints,
            )
            logger.info(
                "END xmatch_update_compared_to_safe: step=%s output=%s (%.2fs)",
                step,
                collection_path,
                time.time() - t0_safe,
            )
            return collection_path

        # Unexpected exceptions are re-raised
        raise