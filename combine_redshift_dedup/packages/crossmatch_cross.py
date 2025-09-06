from __future__ import annotations
"""
Crossmatch and `compared_to` updater for Combine Redshift Catalogs (CRC).

This module performs a spatial crossmatch between two catalogs **without any
tie-breaking or Δz logic**. It updates the `compared_to` column on both inputs
by adding newly crossmatched CRD_IDs (comma-separated, symmetric neighbors),
writes a merged Parquet, and (optionally) imports it as an LSDB **Collection**
(using `_build_collection_with_retry`, which tries a fast path and falls back
to the hats-import pipeline with margin-first retry).

Public API:
    - crossmatch_tiebreak(...)       # name kept for compatibility; no tie-breaking is done
    - crossmatch_tiebreak_safe(...)  # safe wrapper that concatenates on known empty-overlap errors
"""

# -----------------------
# Standard library
# -----------------------
import logging
import os
import pathlib
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Iterable, List, Set, Optional

# -----------------------
# Third-party
# -----------------------
import dask
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

if TYPE_CHECKING:
    # from dask.distributed import Client
    pass

# -----------------------
# Module exports & constants
# -----------------------
__all__ = ["crossmatch_tiebreak", "crossmatch_tiebreak_safe"]

LOGGER_NAME = "crossmatch_and_merge_logger"
LOG_FILE = "crossmatch_and_merge_all.log"


# -----------------------
# Logging utilities
# -----------------------
def _build_logger(logs_dir: str, name: str, file_name: str) -> logging.Logger:
    """Create and configure a file-based logger (single FileHandler, INFO)."""
    log_dir = pathlib.Path(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_dir / file_name, encoding="utf-8", delay=True)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    return logger


# -----------------------
# Saving utilities
# -----------------------
def _safe_to_parquet(ddf, path, **kwargs) -> None:
    """Write parquet robustly across dd.DataFrame and nested_dask frames.

    Attempts `engine="pyarrow"`. If the backend already injects `engine`
    (nested_dask), retry without explicitly setting `engine`.
    """
    try:
        ddf.to_parquet(path, engine="pyarrow", **kwargs)
    except TypeError as e:
        if "multiple values for keyword argument 'engine'" in str(e):
            ddf.to_parquet(path, **kwargs)  # nested_dask path
        else:
            raise


# -----------------------
# Internal helpers
# -----------------------
def _adjacency_from_pairs(left_ids: pd.Series, right_ids: pd.Series) -> Dict[str, Set[str]]:
    """Build an undirected adjacency dict from left-right crossmatch pairs."""
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

    Defensive to odd dtypes (Arrow/Pandas NA, lists/tuples/sets, booleans).
    NA semantics are preserved: empty neighbor set -> NA; otherwise a sorted,
    comma-separated string. Output dtype is Arrow-backed `DTYPE_STR`.
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

    # Build NEW neighbor sets (pure Python), parse OLD cells, then union per row.
    new_sets: List[Set[str]] = [_to_str_set(pairs_adj.get(k, ())) for k in crd_list]
    old_sets: List[Set[str]] = [_parse_existing(v) for v in p["compared_to"].tolist()]

    merged_vals: List[object] = []
    for k, old_set, new_set in zip(crd_list, old_sets, new_sets):
        nxt = set().union(old_set, new_set)
        if k in nxt:
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
    client,
    translation_config: dict | None = None,
    do_import: bool = True,
):
    """Crossmatch two catalogs, update `compared_to`, and save/import the merged result.

    Steps:
      1) Crossmatch `left_cat` vs. `right_cat` within `crossmatch_radius_arcsec` (default 0.75).
      2) Build an undirected adjacency (CRD_ID ↔ CRD_ID) from pairs (no self-pairs).
      3) Partition-wise update of `compared_to` on **both** sides by unioning neighbors.
      4) Concatenate left/right, harmonize key dtypes, and write a merged Parquet.
      5) If `do_import=True`:
           - USE_COLLECTION=True: import as a **Collection** (margin-first fallback) and
             return the collection path.
           - USE_COLLECTION=False: import as a **HATS** catalog and return its artifact path.
         Else, return the merged Parquet folder path.

    Notes:
      - There is no tie-breaking or Δz logic here.
      - `tie_result` is not interpreted; it is just carried through.

    Returns:
      str: Collection path (new mode), HATS path (legacy mode), or merged Parquet path if `do_import=False`.
    """
    logger = _build_logger(logs_dir, LOGGER_NAME, LOG_FILE)
    radius = float((translation_config or {}).get("crossmatch_radius_arcsec", 0.75))

    logger.info(
        "%s: Starting: XMATCH_AND_UPDATE_COMPARED_TO id=merged_step%s",
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
        step,
    )

    # 1) Spatial crossmatch
    xmatched = left_cat.crossmatch(
        right_cat,
        radius_arcsec=radius,
        n_neighbors=10,
        suffixes=("left", "right"),
    )

    # 2) Build adjacency from CRD_ID pairs
    pair_cols = ["CRD_IDleft", "CRD_IDright"]
    pairs_df = xmatched._ddf[pair_cols].compute()
    if len(pairs_df) == 0:
        pairs_adj: Dict[str, Set[str]] = {}
        logger.info("No pairs found in crossmatch; `compared_to` remains unchanged.")
    else:
        pairs_df = pairs_df.astype({"CRD_IDleft": "string", "CRD_IDright": "string"})
        pairs_df = pairs_df[pairs_df["CRD_IDleft"] != pairs_df["CRD_IDright"]].drop_duplicates()
        pairs_adj = _adjacency_from_pairs(pairs_df["CRD_IDleft"], pairs_df["CRD_IDright"])

    logger.info(
        "Compared_to update: %d total links across %d nodes",
        sum(len(v) for v in pairs_adj.values()),
        len(pairs_adj),
    )

    # 3) Ensure `compared_to` meta and update both catalogs partition-wise
    def _meta_with_compared_to(meta_df: pd.DataFrame) -> pd.DataFrame:
        m = meta_df.copy()
        if "compared_to" not in m.columns:
            m["compared_to"] = pd.Series(pd.array([], dtype=DTYPE_STR))
        else:
            m["compared_to"] = pd.Series(pd.array([], dtype=DTYPE_STR))
        return m

    left_meta = _meta_with_compared_to(left_cat._ddf._meta)
    right_meta = _meta_with_compared_to(right_cat._ddf._meta)

    left_updated = left_cat.map_partitions(_merge_compared_to_partition, pairs_adj, meta=left_meta)
    right_updated = right_cat.map_partitions(_merge_compared_to_partition, pairs_adj, meta=right_meta)

    # 4) Concatenate the updated frames
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
            elif dtype in (DTYPE_FLOAT,):
                merged[col] = dd.to_numeric(merged[col], errors="coerce").map_partitions(
                    lambda s: s.astype(DTYPE_FLOAT),
                    meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
                )
            elif dtype in (DTYPE_INT8,):
                merged[col] = dd.to_numeric(merged[col], errors="coerce").map_partitions(
                    lambda s: s.astype(DTYPE_INT8),
                    meta=pd.Series(pd.array([], dtype=DTYPE_INT8)),
                )
            elif dtype in (DTYPE_INT,):
                merged[col] = dd.to_numeric(merged[col], errors="coerce").map_partitions(
                    lambda s: s.astype(DTYPE_INT),
                    meta=pd.Series(pd.array([], dtype=DTYPE_INT)),
                )
            elif dtype == DTYPE_BOOL:
                merged[col] = merged[col].map_partitions(
                    lambda s: s.astype(DTYPE_BOOL),
                    meta=pd.Series(pd.array([], dtype=DTYPE_BOOL)),
                )
        except Exception as e:
            logger.warning("Failed to cast column '%s' to %s: %s", col, dtype, e)

    # 6) Save merged parquet
    merged_path = os.path.join(temp_dir, f"merged_step{step}")
    _safe_to_parquet(merged, merged_path, write_index=False)

    # 7) Optional import (Collection only)
    if do_import:
        logger.info(
            "%s: Starting: COLLECTION_IMPORT id=merged_step%s",
            datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
            step,
        )
        schema_hints = (translation_config or {}).get("expr_column_schema")

        # New builder: derives "<merged_path>_hats" and writes there
        collection_path = _build_collection_with_retry(
            parquet_path=merged_path,
            logs_dir=logs_dir,
            logger=logger,
            client=client,
            try_margin=True,
            schema_hints=schema_hints,
        )
        logger.info(
            "%s: Finished: COLLECTION_IMPORT id=merged_step%s path=%s",
            datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
            step,
            collection_path,
        )
        logger.info(
            "%s: Finished: xmatch_update_compared_to id=merged_step%s",
            datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
            step,
        )
        return collection_path

    return merged_path


def crossmatch_tiebreak_safe(
    left_cat,
    right_cat,
    logs_dir: str,
    temp_dir: str,
    step,
    client,                        # required if do_import=True
    translation_config: dict | None = None,
    do_import: bool = True,
):
    """Wrapper around `crossmatch_tiebreak` with graceful fallback.

    If crossmatch raises a known "no overlap / empty result" error, this wrapper:
      - concatenates inputs (no new `compared_to` links),
      - writes merged Parquet,
      - optionally imports as Collection (new mode) or HATS (legacy mode),
      - and returns the corresponding path.
    """
    logger = _build_logger(logs_dir, LOGGER_NAME, LOG_FILE)
    logger.info(
        "%s: Starting: xmatch_update_compared_to (safe) id=merged_step%s",
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
        step,
    )

    try:
        return crossmatch_tiebreak(
            left_cat=left_cat,
            right_cat=right_cat,
            logs_dir=logs_dir,
            temp_dir=temp_dir,
            step=step,
            client=client,
            translation_config=translation_config,
            do_import=do_import,
        )

    except RuntimeError as e:
        # Known conditions: non-overlapping catalogs / empty crossmatch
        msg = str(e)
        if ("The output catalog is empty" in msg) or ("Catalogs do not overlap" in msg):
            logger.info("%s Proceeding by merging left and right without crossmatching.", msg)

            # Ensure `compared_to` exists with Arrow string dtype on both sides
            lddf = left_cat._ddf
            rddf = right_cat._ddf
            if "compared_to" not in lddf.columns:
                lddf = _add_missing_with_dtype(lddf, "compared_to", DTYPE_STR)
            if "compared_to" not in rddf.columns:
                rddf = _add_missing_with_dtype(rddf, "compared_to", DTYPE_STR)

            merged = dd.concat([lddf, rddf])

            # Save merged result
            merged_path = os.path.join(temp_dir, f"merged_step{step}")
            _safe_to_parquet(merged, merged_path, write_index=False)

            if do_import:
                logger.info(
                    "%s: Starting: COLLECTION_IMPORT (safe) id=merged_step%s",
                    datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
                    step,
                )
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
                    "%s: Finished: COLLECTION_IMPORT (safe) id=merged_step%s path=%s",
                    datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
                    step,
                    collection_path,
                )
                logger.info(
                    "%s: Finished: xmatch_update_compared_to (safe) id=merged_step%s",
                    datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
                    step,
                )
                return collection_path

            # If not importing, return the parquet path
            logger.info(
                "%s: Finished: xmatch_update_compared_to (safe) id=merged_step%s",
                datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
                step,
            )
            return merged_path

        # Unexpected exceptions are re-raised
        raise
