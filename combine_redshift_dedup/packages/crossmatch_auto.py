from __future__ import annotations
"""
Self-crossmatch and `compared_to` updater for Combine Redshift Catalogs (CRC).

This module performs a *self* spatial crossmatch of a single catalog and updates
its `compared_to` column (adding neighbor CRD_IDs and removing self-links).

Behavior: Collection-only
- Works with an LSDB *Collection* already imported on disk.
- Overwrites the *collection* in place via `to_hats(collection_path, overwrite=True)`.
- Returns the `collection_path`.

In both the previous and current design the core logic is the same: self-crossmatch,
build an adjacency (CRD_ID ↔ CRD_ID) without self-pairs, merge the neighbors into
`compared_to`, and persist the result.

Public API:
    - crossmatch_auto(...)
"""

# =======================
# Standard library
# =======================
import os
import pathlib
import logging
from datetime import datetime
from typing import Dict, Iterable, List, Set, Tuple, Optional, Union

# =======================
# Third-party
# =======================
import numpy as np
import pandas as pd
import lsdb  # LSDB Catalog type / open_catalog

# =======================
# Project
# =======================
from combine_redshift_dedup.packages.specz import (
    DTYPE_STR,                  # Arrow-backed string dtype to enforce on `compared_to`
)

__all__ = ["crossmatch_auto"]

LOGGER_NAME = "crossmatch_auto_logger"
LOG_FILE = "crossmatch_auto_all.log"


# =======================
# Logging utilities
# =======================
def _build_logger(logs_dir: str, name: str, file_name: str) -> logging.Logger:
    """Create and configure a file-based logger."""
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


# =======================
# Internal helpers
# =======================
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
    """Update `compared_to` on one partition by unioning existing entries with new pairs.

    Defensive against odd dtypes (Arrow scalars, lists/tuples/sets, booleans).
    Preserves NA semantics: empty neighbor set -> NA; otherwise a sorted,
    comma-separated string. Output dtype is `specz.DTYPE_STR`.
    """
    p = part.copy()

    # Ensure column exists; dtype is enforced at write time.
    if "compared_to" not in p.columns:
        p["compared_to"] = pd.Series(pd.NA, index=p.index)

    # Work with Python lists to avoid dtype surprises.
    crd_list: List[str] = p["CRD_ID"].astype(str).tolist()

    def _norm_token(x) -> str | None:
        if pd.isna(x):
            return None
        if isinstance(x, (bool, np.bool_)):  # drop accidental bools
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

    # Build NEW neighbor sets (pure Python).
    new_sets: List[Set[str]] = [_to_str_set(pairs_adj.get(k, ())) for k in crd_list]

    # Parse OLD compared_to values into sets.
    old_vals = p["compared_to"].tolist()
    old_sets: List[Set[str]] = [_parse_existing(v) for v in old_vals]

    # Union per row; drop self; stringify or keep NA if empty.
    merged_vals: List[object] = []
    for k, old_set, new_set in zip(crd_list, old_sets, new_sets):
        nxt = set().union(old_set, new_set)
        if k in nxt:
            nxt.discard(k)
        merged_vals.append(", ".join(sorted(nxt)) if nxt else pd.NA)

    # Enforce Arrow-backed string dtype.
    p["compared_to"] = pd.Series(pd.array(merged_vals, dtype=DTYPE_STR), index=p.index)
    return p


def _self_xmatch_pairs(
    catalog: "lsdb.catalog.Catalog",
    radius_arcsec: float,
    n_neighbors: int,
    logger: logging.Logger,
) -> Dict[str, Set[str]]:
    """Run a self-crossmatch and return an adjacency (CRD_ID -> neighbors)."""
    xmatched = catalog.crossmatch(
        catalog,
        radius_arcsec=radius_arcsec,
        n_neighbors=n_neighbors,
        suffixes=("left", "right"),
    )
    pair_cols = ["CRD_IDleft", "CRD_IDright"]
    pairs_df = xmatched[pair_cols].compute()
    if len(pairs_df) == 0:
        logger.info("No pairs found in self-crossmatch; `compared_to` remains unchanged.")
        return {}
    pairs_df = pairs_df.astype({"CRD_IDleft": "string", "CRD_IDright": "string"})
    pairs_df = pairs_df[pairs_df["CRD_IDleft"] != pairs_df["CRD_IDright"]].drop_duplicates()
    return _adjacency_from_pairs(pairs_df["CRD_IDleft"], pairs_df["CRD_IDright"])


def _update_compared_to(
    catalog: "lsdb.catalog.Catalog",
    pairs_adj: Dict[str, Set[str]],
) -> "lsdb.catalog.Catalog":
    """Return a new Catalog whose partitions have an updated `compared_to` column."""
    meta_pdf = catalog._ddf._meta.copy()
    if "compared_to" not in meta_pdf.columns:
        meta_pdf = meta_pdf.assign(compared_to=pd.Series(pd.array([], dtype=DTYPE_STR)))
    else:
        meta_pdf["compared_to"] = pd.Series(pd.array([], dtype=DTYPE_STR))
    return catalog.map_partitions(_merge_compared_to_partition, pairs_adj, meta=meta_pdf)


# =======================
# Main (dispatch by mode)
# =======================
def crossmatch_auto(
    catalog: "lsdb.catalog.Catalog",
    collection_path: str,
    logs_dir: str,
    translation_config: dict | None = None,
) -> str:
    """Self-crossmatch a catalog, update `compared_to`, and persist the result.

    Collection-only mode:
      * `collection_path` must be provided and must point to an existing Collection folder,
        e.g. ".../temp/cat001_hats".
      * The function writes a NEW collection alongside it, named "<artifact>_auto"
        (e.g., ".../temp/cat001_hats_auto"), and returns that new path.
    """
    logger = _build_logger(logs_dir, LOGGER_NAME, LOG_FILE)
    if not collection_path:
        raise ValueError("`collection_path` must be provided (collection-only mode).")

    # Parameters with defaults
    radius = float((translation_config or {}).get("crossmatch_radius_arcsec", 0.75))
    k = int((translation_config or {}).get("crossmatch_n_neighbors", 10))

    # Derive parent dir and artifact names
    parent_dir, artifact = os.path.split(os.path.normpath(collection_path))
    if not parent_dir:
        parent_dir = "."
    artifact_auto = f"{artifact}_auto"
    collection_path_auto = os.path.join(parent_dir, artifact_auto)

    logger.info(
        "%s: Starting: SELF_XMATCH_AND_UPDATE (Collection) src=%s dst=%s",
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
        collection_path,
        collection_path_auto,
    )

    # 1) Self-crossmatch → adjacency (CRD_ID -> neighbors)
    pairs_adj = _self_xmatch_pairs(catalog, radius, k, logger)
    logger.info(
        "Compared_to update (self-xmatch): %d total links across %d nodes",
        sum(len(v) for v in pairs_adj.values()), len(pairs_adj),
    )

    # 2) Update `compared_to`
    updated = _update_compared_to(catalog, pairs_adj)

    # 3) Persist as a NEW collection: write to the parent directory,
    #    specifying the new catalog_name (<artifact>_auto).
    logger.info(
        "%s: Starting: WRITE_COLLECTION base_dir=%s catalog_name=%s",
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
        parent_dir,
        artifact_auto,
    )
    updated.to_hats(
        collection_path_auto,
        #catalog_name=artifact_auto,
        as_collection=True,
        overwrite=True,
    )
    logger.info(
        "%s: Finished: WRITE_COLLECTION path=%s",
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
        collection_path_auto,
    )

    logger.info(
        "%s: Finished: SELF_XMATCH_AND_UPDATE (Collection) dst=%s",
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
        collection_path_auto,
    )
    return collection_path_auto
