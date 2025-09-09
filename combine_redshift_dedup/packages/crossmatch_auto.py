from __future__ import annotations
"""
Self-crossmatch and `compared_to` updater for CRC.

Runs a self spatial crossmatch on a single catalog, updates its `compared_to`
column (adds neighbor CRD_IDs, removes self-links), and writes a new on-disk
collection `<artifact>_auto`.

Public API:
    - crossmatch_auto(...)
"""

# =======================
# Standard library
# =======================
import os
import time
import logging
from typing import Dict, Iterable, List, Set, Optional

# =======================
# Third-party
# =======================
import numpy as np
import pandas as pd
import lsdb  # LSDB Catalog type / open_catalog

# =======================
# Project
# =======================
from combine_redshift_dedup.packages.specz import DTYPE_STR  # Arrow-backed string dtype
from combine_redshift_dedup.packages.utils import get_phase_logger

__all__ = ["crossmatch_auto"]

LOGGER_NAME = "crc.crossmatch_auto"  # child of the central pipeline logger


# =======================
# Logging helper
# =======================
def _get_logger() -> logging.LoggerAdapter:
    """Return a phase-aware logger ('crc.crossmatch_auto' with phase='automatch')."""
    base = logging.getLogger(LOGGER_NAME)
    base.propagate = True
    return get_phase_logger("automatch", base)


# =======================
# Internal helpers
# =======================
def _adjacency_from_pairs(left_ids: pd.Series, right_ids: pd.Series) -> Dict[str, Set[str]]:
    """Build an undirected adjacency from left-right crossmatch pairs.

    Args:
        left_ids: Series with left CRD_IDs.
        right_ids: Series with right CRD_IDs.

    Returns:
        Dict mapping CRD_ID -> set of neighbor CRD_IDs.
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
    """Update one partition by unioning `compared_to` with new adjacency.

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

    def _norm_token(x) -> Optional[str]:
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

    # New neighbor sets (Python-side).
    new_sets: List[Set[str]] = [_to_str_set(pairs_adj.get(k, ())) for k in crd_list]

    # Old compared_to as sets.
    old_sets: List[Set[str]] = [_parse_existing(v) for v in p["compared_to"].tolist()]

    # Union; drop self; stringify or keep NA.
    merged_vals: List[object] = []
    for k, old_set, new_set in zip(crd_list, old_sets, new_sets):
        nxt = set().union(old_set, new_set)
        nxt.discard(k)
        merged_vals.append(", ".join(sorted(nxt)) if nxt else pd.NA)

    p["compared_to"] = pd.Series(pd.array(merged_vals, dtype=DTYPE_STR), index=p.index)
    return p


def _self_xmatch_pairs(
    catalog: "lsdb.catalog.Catalog",
    radius_arcsec: float,
    n_neighbors: int,
    logger: logging.LoggerAdapter,
) -> Dict[str, Set[str]]:
    """Run a self-crossmatch and return an adjacency (CRD_ID -> neighbors).

    Args:
        catalog: LSDB catalog to crossmatch with itself.
        radius_arcsec: Matching radius in arcseconds.
        n_neighbors: Maximum neighbors per source.
        logger: Logger.

    Returns:
        Mapping CRD_ID -> set of neighbor CRD_IDs.
    """
    logger.info(
        "Running self-crossmatch: radius=%.3f\" n_neighbors=%d",
        radius_arcsec,
        n_neighbors,
    )
    xmatched = catalog.crossmatch(
        catalog,
        radius_arcsec=radius_arcsec,
        n_neighbors=n_neighbors,
        suffixes=("left", "right"),
    )
    pair_cols = ["CRD_IDleft", "CRD_IDright"]
    pairs_df = xmatched[pair_cols].compute()
    if len(pairs_df) == 0:
        logger.info("Self-crossmatch: no pairs found; `compared_to` remains unchanged.")
        return {}
    pairs_df = pairs_df.astype({"CRD_IDleft": "string", "CRD_IDright": "string"})
    pairs_df = pairs_df[pairs_df["CRD_IDleft"] != pairs_df["CRD_IDright"]].drop_duplicates()

    adj = _adjacency_from_pairs(pairs_df["CRD_IDleft"], pairs_df["CRD_IDright"])
    total_links = sum(len(v) for v in adj.values())
    logger.info("Self-crossmatch: %d unique pairs across %d nodes", total_links, len(adj))
    return adj


def _update_compared_to(
    catalog: "lsdb.catalog.Catalog",
    pairs_adj: Dict[str, Set[str]],
) -> "lsdb.catalog.Catalog":
    """Return a new Catalog with updated `compared_to` in each partition.

    Args:
        catalog: LSDB catalog.
        pairs_adj: Mapping CRD_ID -> neighbors.

    Returns:
        New catalog object with updated `compared_to`.
    """
    meta_pdf = catalog._ddf._meta.copy()
    if "compared_to" not in meta_pdf.columns:
        meta_pdf = meta_pdf.assign(compared_to=pd.Series(pd.array([], dtype=DTYPE_STR)))
    else:
        meta_pdf["compared_to"] = pd.Series(pd.array([], dtype=DTYPE_STR))
    return catalog.map_partitions(_merge_compared_to_partition, pairs_adj, meta=meta_pdf)


# =======================
# Public API
# =======================
def crossmatch_auto(
    catalog: "lsdb.catalog.Catalog",
    collection_path: str,
    logs_dir: str,  # kept for API compatibility; logging is centralized
    translation_config: dict | None = None,
) -> str:
    """Run self-crossmatch, update `compared_to`, and persist as `<artifact>_auto`.

    Args:
        catalog: LSDB catalog already imported as a collection.
        collection_path: Existing collection path (e.g., ".../temp/003_xxx_hats").
        logs_dir: Unused (logging is centralized via the root pipeline logger).
        translation_config: Optional dict with `crossmatch_radius_arcsec` and
            `crossmatch_n_neighbors`.

    Returns:
        Path to the new collection `<artifact>_auto`.
    """
    logger = _get_logger()
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

    # START (per-catalog)
    t0 = time.time()
    logger.info(
        "START automatch: artifact=%s radius=%.3f\" n_neighbors=%d src=%s dst=%s",
        artifact,
        radius,
        k,
        collection_path,
        collection_path_auto,
    )

    # 1) Self-crossmatch → adjacency (CRD_ID -> neighbors)
    pairs_adj = _self_xmatch_pairs(catalog, radius, k, logger)

    # 2) Update `compared_to`
    updated = _update_compared_to(catalog, pairs_adj)
    total_links = sum(len(v) for v in pairs_adj.values())
    n_nodes = len(pairs_adj)

    # 3) Persist as a NEW collection in the parent directory
    logger.info("Writing collection: base_dir=%s catalog_name=%s", parent_dir, artifact_auto)
    updated.to_hats(
        collection_path_auto,
        as_collection=True,
        overwrite=True,
    )
    logger.info("Write complete: path=%s", collection_path_auto)

    # END (per-catalog)
    dt = time.time() - t0
    logger.info(
        "END automatch: artifact=%s links=%d nodes=%d output=%s (%.2fs)",
        artifact,
        total_links,
        n_nodes,
        collection_path_auto,
        dt,
    )

    return collection_path_auto
