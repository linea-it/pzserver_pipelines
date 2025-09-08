from __future__ import annotations
"""
Crossmatch and `compared_to` updater for CRC.

Performs a spatial crossmatch between two catalogs (no tie-breaking / Δz),
updates `compared_to` symmetrically on both inputs, writes a merged Parquet,
and optionally imports it as an LSDB Collection (margin-first retry).

Public API:
    - crossmatch_tiebreak(...)
    - crossmatch_tiebreak_safe(...)
"""

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
import dask.dataframe as dd
import numpy as np
import pandas as pd
import lsdb

# -----------------------
# Project
# -----------------------
from combine_redshift_dedup.packages.specz import (
    _build_collection_with_retry,
    _normalize_string_series_to_na,
    _add_missing_with_dtype,
    DTYPE_STR, DTYPE_FLOAT, DTYPE_INT, DTYPE_BOOL, DTYPE_INT8,
)
from combine_redshift_dedup.packages.utils import get_phase_logger

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
# Type utilities
# -----------------------
def _coerce_optional_columns_for_import(
    df: dd.DataFrame,
    schema_hints: dict | None = None,
) -> dd.DataFrame:
    """Coerce prev/expr columns to consistent Arrow dtypes across partitions.

    Args:
        df: Merged Dask dataframe.
        schema_hints: Mapping {col_name: 'str'|'float'|'int'|'bool'}.

    Returns:
        dd.DataFrame with coerced/added columns.
    """
    # 1) Prev columns → string
    prev_like = [c for c in df.columns if str(c).startswith("CRD_ID_prev")]
    prev_like += [c for c in df.columns if str(c).startswith("compared_to_prev")]
    for c in prev_like:
        df[c] = df[c].map_partitions(
            _normalize_string_series_to_na,
            meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
        )

    # 2) Expr columns guided by hints
    hints = dict(schema_hints or {})
    if not hints:
        return df

    for col, kind in hints.items():
        k = str(kind).lower()
        if col not in df.columns:
            # create missing with target dtype (so partições vazias não viram null[pyarrow])
            if k == "str":
                df = _add_missing_with_dtype(df, col, DTYPE_STR)
            elif k == "float":
                df = _add_missing_with_dtype(df, col, DTYPE_FLOAT)
            elif k == "int":
                df = _add_missing_with_dtype(df, col, DTYPE_INT)
            elif k == "bool":
                df = _add_missing_with_dtype(df, col, DTYPE_BOOL)
            continue

        # cast existing
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
            # conservador: apenas dtype target; upstream já deve ter saneado valores
            df[col] = df[col].map_partitions(
                lambda s: s.astype(DTYPE_BOOL),
                meta=pd.Series(pd.array([], dtype=DTYPE_BOOL)),
            )

    return df

# -----------------------
# Saving utilities
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


# -----------------------
# Internal helpers
# -----------------------
def _adjacency_from_pairs(left_ids: pd.Series, right_ids: pd.Series) -> Dict[str, Set[str]]:
    """Build an undirected adjacency from left-right crossmatch pairs.

    Args:
        left_ids: Series of left CRD_IDs.
        right_ids: Series of right CRD_IDs.

    Returns:
        Mapping CRD_ID -> set of neighbor CRD_IDs.
    """
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
    """Partition-wise update of `compared_to` by unioning existing entries with new pairs.

    Args:
        part: Partition dataframe.
        pairs_adj: Mapping CRD_ID -> iterable of neighbors.

    Returns:
        Partition with updated `compared_to` (Arrow string dtype), NA if empty.
    """
    p = part.copy()

    if "compared_to" not in p.columns:
        p["compared_to"] = pd.Series(pd.NA, index=p.index)

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


# =======================
# Main logic
# =======================
def crossmatch_tiebreak(
    left_cat,
    right_cat,
    logs_dir: str,
    temp_dir: str,
    step,
    client,  # required if do_import=True
    translation_config: dict | None = None,
    do_import: bool = True,
):
    """Crossmatch two catalogs, update `compared_to`, and save/import the merged result.

    Steps:
      1) Crossmatch `left_cat` vs `right_cat` (default radius 0.75 arcsec).
      2) Build undirected adjacency (CRD_ID ↔ CRD_ID) without self-pairs.
      3) Update `compared_to` on both sides by unioning neighbors.
      4) Concatenate, harmonize key dtypes, and write Parquet.
      5) Optionally import as a Collection (margin-first fallback).

    Returns:
      Collection path (if imported) or merged Parquet folder path if `do_import=False`.
    """
    logger = _get_logger()
    t0_all = time.time()
    radius = float((translation_config or {}).get("crossmatch_radius_arcsec", 0.75))

    logger.info(
        "START crossmatch_update_compared_to: step=%s radius=%.3f\" import=%s",
        step,
        radius,
        bool(do_import),
    )

    # 1) Spatial crossmatch
    t0 = time.time()
    xmatched = left_cat.crossmatch(
        right_cat,
        radius_arcsec=radius,
        n_neighbors=10,
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

    # 3) Ensure `compared_to` meta and update both catalogs partition-wise
    def _meta_with_compared_to(meta_df: pd.DataFrame) -> pd.DataFrame:
        m = meta_df.copy()
        if "compared_to" not in m.columns:
            m["compared_to"] = pd.Series(pd.array([], dtype=DTYPE_STR))
        else:
            m["compared_to"] = pd.Series(pd.array([], dtype=DTYPE_STR))
        return m

    t0 = time.time()
    left_meta = _meta_with_compared_to(left_cat._ddf._meta)
    right_meta = _meta_with_compared_to(right_cat._ddf._meta)

    left_updated = left_cat.map_partitions(_merge_compared_to_partition, pairs_adj, meta=left_meta)
    right_updated = right_cat.map_partitions(_merge_compared_to_partition, pairs_adj, meta=right_meta)
    logger.info("Compared_to updated on partitions (%.2fs)", time.time() - t0)

    # 4) Concatenate the updated frames
    t0 = time.time()
    merged = dd.concat([left_updated._ddf, right_updated._ddf])

    # 5) Normalize expected dtypes (focus on compared_to; keep others if present)
    expected_types = {
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
        except Exception as e:
            logger.warning("Failed to cast column '%s' to %s: %s", col, dtype, e)

    # 5b) Coerce prev/expr columns to stable Arrow dtypes across partitions
    schema_hints_local = (translation_config or {}).get("expr_column_schema", {}) or {}
    merged = _coerce_optional_columns_for_import(merged, schema_hints_local)

    logger.info("Type normalization complete (%.2fs)", time.time() - t0)

    # 6) Save merged parquet
    t0 = time.time()
    merged_path = os.path.join(temp_dir, f"merged_step{step}")
    _safe_to_parquet(merged, merged_path, write_index=False)
    logger.info("Parquet written: path=%s (%.2fs)", merged_path, time.time() - t0)

    # 7) Optional import (Collection)
    if do_import:
        t0 = time.time()
        logger.info("START import_collection: step=%s parquet=%s", step, merged_path)
        schema_hints = (translation_config or {}).get("expr_column_schema")

        collection_path = _build_collection_with_retry(
            parquet_path=merged_path,
            logs_dir=logs_dir,         # interface symmetry; central logger is used
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

    logger.info(
        "END crossmatch_update_compared_to: step=%s links=%d nodes=%d output=%s (%.2fs)",
        step,
        total_links,
        len(pairs_adj),
        merged_path,
        time.time() - t0_all,
    )
    return merged_path


def crossmatch_tiebreak_safe(
    left_cat,
    right_cat,
    logs_dir: str,
    temp_dir: str,
    step,
    client,  # required if do_import=True
    translation_config: dict | None = None,
    do_import: bool = True,
):
    """Wrapper around `crossmatch_tiebreak` with graceful empty-overlap fallback.

    If the crossmatch yields a known empty-overlap condition, concatenates inputs
    (no new `compared_to` links), writes the merged Parquet, optionally imports
    as a Collection, and returns the path.
    """
    logger = _get_logger()
    t0_safe = time.time()
    logger.info("START xmatch_update_compared_to_safe: step=%s import=%s", step, bool(do_import))

    try:
        out = crossmatch_tiebreak(
            left_cat=left_cat,
            right_cat=right_cat,
            logs_dir=logs_dir,
            temp_dir=temp_dir,
            step=step,
            client=client,
            translation_config=translation_config,
            do_import=do_import,
        )
        logger.info("END xmatch_update_compared_to_safe: step=%s output=%s (%.2fs)", step, out, time.time() - t0_safe)
        return out

    except RuntimeError as e:
        msg = str(e)
        if ("The output catalog is empty" in msg) or ("Catalogs do not overlap" in msg):
            logger.info("Empty-overlap condition detected: %s", msg)

            # Ensure `compared_to` exists with Arrow string dtype on both sides
            lddf = left_cat._ddf
            rddf = right_cat._ddf
            if "compared_to" not in lddf.columns:
                lddf = _add_missing_with_dtype(lddf, "compared_to", DTYPE_STR)
            if "compared_to" not in rddf.columns:
                rddf = _add_missing_with_dtype(rddf, "compared_to", DTYPE_STR)

            # Save merged result
            t0 = time.time()
            merged = dd.concat([lddf, rddf])
            merged_path = os.path.join(temp_dir, f"merged_step{step}")
            _safe_to_parquet(merged, merged_path, write_index=False)
            logger.info("Parquet written (safe path): %s (%.2fs)", merged_path, time.time() - t0)

            if do_import:
                t0 = time.time()
                logger.info("START import_collection_safe: step=%s parquet=%s", step, merged_path)
                schema_hints = (translation_config or {}).get("expr_column_schema")

                collection_path = _build_collection_with_retry(
                    parquet_path=merged_path,
                    logs_dir=logs_dir,
                    logger=logger,
                    client=client,
                    try_margin=True,
                    schema_hints=schema_hints,
                )
                logger.info(
                    "END import_collection_safe: step=%s path=%s (%.2fs)",
                    step,
                    collection_path,
                    time.time() - t0,
                )
                logger.info(
                    "END xmatch_update_compared_to_safe: step=%s output=%s (%.2fs)",
                    step,
                    collection_path,
                    time.time() - t0_safe,
                )
                return collection_path

            logger.info(
                "END xmatch_update_compared_to_safe: step=%s output=%s (%.2fs)",
                step,
                merged_path,
                time.time() - t0_safe,
            )
            return merged_path

        # Unexpected exceptions are re-raised
        raise
