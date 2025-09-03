# crossmatch.py
from __future__ import annotations
"""
Crossmatch and merge logic for Combine Redshift Catalogs (CRC).

This module performs a spatial crossmatch between two catalogs, applies a
priority-based tie-breaker (with optional Δz disambiguation), accumulates
pairwise links for diagnostics, and outputs a merged artifact. Optionally,
it imports the merged catalog into HATS.

Public API:
    - crossmatch_tiebreak(...)
    - crossmatch_tiebreak_safe(...)
"""

# -----------------------
# Standard library
# -----------------------
import json
import logging
import os
import pathlib
from collections import defaultdict, deque
from datetime import datetime
from typing import TYPE_CHECKING
from functools import reduce
from collections import deque

# -----------------------
# Third-party
# -----------------------
import dask
import dask.dataframe as dd
import lsdb
import numpy as np
import pandas as pd

# -----------------------
# Project
# -----------------------
from combine_redshift_dedup.packages.specz import (
    import_catalog,
    _normalize_string_series_to_na,
    _add_missing_with_dtype,
    _to_nullable_boolean_strict,
    _normalize_schema_hints,
    DTYPE_STR, DTYPE_FLOAT, DTYPE_INT, DTYPE_BOOL, DTYPE_INT8,
)

if TYPE_CHECKING:
    # from dask.distributed import Client  # Not strictly needed for hints here
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
    """Create and configure a file-based logger.

    Args:
        logs_dir (str): Directory where the log file will be written.
        name (str): Logger name (e.g., ``LOGGER_NAME``).
        file_name (str): Log file name (e.g., ``"crossmatch_and_merge_all.log"``).

    Returns:
        logging.Logger: Configured logger (single FileHandler, INFO level).
    """
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
def _safe_to_parquet(ddf, path, **kwargs):
    """Write parquet robustly across dd.DataFrame and nested_dask frames.

    Tries with engine='pyarrow'. If the backend already injects 'engine'
    (nested_dask), fall back to calling without 'engine'.
    """
    try:
        ddf.to_parquet(path, engine="pyarrow", **kwargs)
    except TypeError as e:
        if "multiple values for keyword argument 'engine'" in str(e):
            ddf.to_parquet(path, **kwargs)  # nested_dask path
        else:
            raise

# =======================
# Main logic
# =======================
def crossmatch_tiebreak(
    left_cat,
    right_cat,
    tiebreaking_priority,
    logs_dir,
    temp_dir,
    step,
    client,
    compared_to_left,
    compared_to_right,
    instrument_type_priority,
    translation_config,
    do_import: bool = True,
):
    """Crossmatch two catalogs and resolve duplicates using prioritized rules.

    Workflow:
      1) Filter out eliminated rows (``tie_result == 0``) from both sides.
      2) Spatial crossmatch within ``crossmatch_radius_arcsec`` (default 0.75).
      3) Row-wise tie decision using ``tiebreaking_priority``; stars (flag 6)
         are eliminated early when both sides provide homogenized flags.
      4) Collect pair links (for diagnostics) and persist as JSON.
      5) Optional Δz disambiguation for hard ties (both sides remain ``2``).
      6) Collapse left/right results into a single ``tie_result`` per ``CRD_ID``
         per connected component: one survivor → 1; ≥2 survivors → 2.
      7) Apply final decisions back to left/right catalogs and merge.
      8) Normalize dtypes (nullable), save Parquet, and optionally import HATS.

    Args:
        left_cat: Left catalog (lsdb catalog-like; must support ``query``,
            ``crossmatch``, and expose a Dask DataFrame via ``._ddf``).
        right_cat: Right catalog (same requirements as left).
        tiebreaking_priority (list[str]): Ordered criteria; may include
            ``"z_flag_homogenized"`` and/or ``"instrument_type_homogenized"``.
        logs_dir (str): Directory for log files.
        temp_dir (str): Directory for temporary/merged outputs.
        step (int | str): Step identifier to name output artifacts.
        client: Dask client (used only when importing to HATS via ``import_catalog``).
        compared_to_left (dict): Pre-existing compared_to links for the left set.
        compared_to_right (dict): Pre-existing compared_to links for the right set.
        instrument_type_priority (dict[str, float]): Priority map for instrument types.
        translation_config (dict): Config with keys like
            ``crossmatch_radius_arcsec`` and ``delta_z_threshold``.
        do_import (bool, optional): If True, import merged catalog to HATS and
            return an lsdb handle; otherwise return the merged Parquet path.

    Returns:
        tuple[object, str]: ``(merged_handle_or_path, compared_to_path)``, where
        the first element is an lsdb catalog if ``do_import`` is True, else a str.

    Raises:
        ValueError: If both ``tiebreaking_priority`` is empty and
            ``delta_z_threshold`` is 0/None (no way to deduplicate).
    """
    # --- Logger ---
    logger = _build_logger(logs_dir, LOGGER_NAME, LOG_FILE)

    # --- Config ---
    crossmatch_radius = translation_config.get("crossmatch_radius_arcsec", 0.75)
    delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)
    if not tiebreaking_priority and (delta_z_threshold is None or float(delta_z_threshold) == 0.0):
        raise ValueError(
            "Cannot deduplicate: tiebreaking_priority is empty and delta_z_threshold is not set or is zero. "
            "Please define at least one deduplication criterion."
        )

    # --- Pre-filter eliminated rows ---
    left_alive = left_cat.query("tie_result == 1 or tie_result == 2")
    right_alive = right_cat.query("tie_result == 1 or tie_result == 2")

    # --- Spatial crossmatch (within radius) ---
    xmatched = left_alive.crossmatch(
        right_alive,
        radius_arcsec=crossmatch_radius,
        n_neighbors=2,
        suffixes=("left", "right"),
    )

    # --- Hard tie accumulator (persisted across steps) ---
    hard_tie_path = os.path.join(temp_dir, "hard_tie_ids.json")
    if os.path.exists(hard_tie_path):
        with open(hard_tie_path, "r") as f:
            hard_tie_cumulative = set(json.load(f))
    else:
        hard_tie_cumulative = set()

    # ------------------------------------------------------------------
    # Row-wise tie decision on crossmatched pairs
    # ------------------------------------------------------------------
    def decide_tie(row: pd.Series) -> tuple[int, int]:
        """Return (tie_left, tie_right) for a pair, using priority criteria.

        Semantics:
          - If both sides are stars (z_flag_homogenized==6) → drop both (0,0).
          - Else if exactly one is star → keep the other (2/1 vs 0).
          - Else iterate over priority columns:
              * If both sides have values: pick the larger.
              * Track the first column where only one side has value; if all
                comparisons tie, prefer that side.
          - If no information: (2,2) -> hard tie.
        """
        left_was_tie = row.get("tie_resultleft", 1) == 2
        right_was_tie = row.get("tie_resultright", 1) == 2

        zflag_left = row.get("z_flag_homogenizedleft")
        zflag_right = row.get("z_flag_homogenizedright")

        # Early star rule
        #if pd.notna(zflag_left) and zflag_left == 6 and pd.notna(zflag_right) and zflag_right == 6:
        #    return (0, 0)
        #elif pd.notna(zflag_left) and zflag_left == 6:
        #    return (0, 2 if right_was_tie else 1)
        #elif pd.notna(zflag_right) and zflag_right == 6:
        #    return (2 if left_was_tie else 1, 0)

        if not tiebreaking_priority:
            return (2, 2)  # defer to Δz

        first_non_nan_side = None

        for col in tiebreaking_priority:
            v1 = row.get(f"{col}left")
            v2 = row.get(f"{col}right")

            if col == "instrument_type_homogenized":
                v1 = instrument_type_priority.get(v1, 0)
                v2 = instrument_type_priority.get(v2, 0)

            if pd.notna(v1) and pd.notna(v2):
                if v1 > v2:
                    return (2 if left_was_tie else 1, 0)
                elif v2 > v1:
                    return (0, 2 if right_was_tie else 1)
                else:
                    continue  # equal → next criterion

            if pd.notna(v1) and pd.isna(v2) and first_non_nan_side is None:
                first_non_nan_side = "left"
            elif pd.isna(v1) and pd.notna(v2) and first_non_nan_side is None:
                first_non_nan_side = "right"

        if first_non_nan_side == "left":
            return (2 if left_was_tie else 1, 0)
        elif first_non_nan_side == "right":
            return (0, 2 if right_was_tie else 1)
        else:
            return (2, 2)  # no info

    tie_results = xmatched.map_partitions(
        lambda p: p.apply(decide_tie, axis=1, result_type="expand"),
        meta={0: "i8", 1: "i8"},
    )
    xmatched = xmatched.assign(tie_left=tie_results[0], tie_right=tie_results[1])

    # ------------------------------------------------------------------
    # Collect links for compared_to (distributed) and prep Δz resolution
    # ------------------------------------------------------------------
    # Work on the underlying Dask DataFrame (xmatched._ddf) to avoid early .compute().
    xmatched_ddf = xmatched._ddf
    pairs_dd = xmatched_ddf[["CRD_IDleft", "CRD_IDright", "tie_left", "tie_right", "zleft", "zright"]].persist()

    # === Build compared_to_new WITHOUT pulling all edges to the driver ===
    def _pairs_partition_to_dict(pdf: pd.DataFrame) -> dict[str, set[str]]:
        """Return an undirected adjacency dict from a single pandas partition."""
        out: dict[str, set[str]] = {}
        if pdf.empty:
            return out
        L = pdf["CRD_IDleft"].astype(str).to_numpy()
        R = pdf["CRD_IDright"].astype(str).to_numpy()
        for a, b in zip(L, R):
            s = out.get(a)
            if s is None:
                out[a] = {b}
            else:
                s.add(b)
            s = out.get(b)
            if s is None:
                out[b] = {a}
            else:
                s.add(a)
        return out

    def _merge_two(a: dict[str, set[str]], b: dict[str, set[str]]) -> dict[str, set[str]]:
        """In-place union of adjacency dicts."""
        if not a:
            return b
        if not b:
            return a
        for k, v in b.items():
            s = a.get(k)
            if s is None:
                a[k] = set(v)
            else:
                s.update(v)
        return a

    # Partition-wise adjacencies (delayed) from Dask partitions
    parts = [dask.delayed(_pairs_partition_to_dict)(p) for p in pairs_dd.to_delayed()]
    compared_to_new = reduce(lambda A, B: dask.delayed(_merge_two)(A, B),
                             parts, dask.delayed(dict)()).compute()

    logger.info(
        "New pairs (distributed agg): %d total links in %d objects",
        sum(len(v) for v in compared_to_new.values()),
        len(compared_to_new),
    )

    # === Merge with previous compared_to dicts and persist JSON ===
    compared_to_dict = defaultdict(set)
    for d in [compared_to_left, compared_to_right]:
        for k, v in d.items():
            compared_to_dict[str(k)].update(map(str, v))
    for k, new_vals in compared_to_new.items():
        compared_to_dict[str(k)].update(map(str, new_vals))

    total_links = sum(len(v) for v in compared_to_dict.values())
    logger.info("Compared_to (merged): %d total links across %d objects", total_links, len(compared_to_dict))
    logger.info("Step %s: Compared_to merge completed", step)

    compared_to_serializable = {k: sorted(v) for k, v in compared_to_dict.items()}
    compared_to_path = os.path.join(temp_dir, f"compared_to_xmatch_step{step}.json")
    with open(compared_to_path, "w") as f:
        json.dump(compared_to_serializable, f)

    # === Identify losers early (distributed) ===
    # Use boolean indexing (not .loc) to avoid Dask loc pitfalls.
    losersL = (
        pairs_dd[pairs_dd["tie_left"] == 0]["CRD_IDleft"]
        .astype(str).drop_duplicates().compute()
    )
    losersR = (
        pairs_dd[pairs_dd["tie_right"] == 0]["CRD_IDright"]
        .astype(str).drop_duplicates().compute()
    )
    losers_set = set(pd.Index(losersL).union(pd.Index(losersR)))

    # Only surviving edges proceed to Δz / component collapse (much smaller set).
    survivors_dd = pairs_dd[(pairs_dd["tie_left"] != 0) & (pairs_dd["tie_right"] != 0)]

    # ------------------------------------------------------------------
    # Δz tie resolution (hard ties only) on the survivors set
    # ------------------------------------------------------------------
    # Materialize to pandas now that the table is significantly smaller.
    pairs = survivors_dd[["CRD_IDleft", "CRD_IDright", "tie_left", "tie_right", "zleft", "zright"]].compute()
    pairs = pairs.sort_values(["CRD_IDleft", "CRD_IDright"], kind="mergesort").reset_index(drop=True)

    if float(delta_z_threshold) > 0.0 and not pairs.empty:
        thr = float(delta_z_threshold)

        def apply_delta_z_fix(pairs_df: pd.DataFrame) -> pd.DataFrame:
            """
            Δz disambiguation within the subgraph |Δz| <= thr:

            Phase 1) Isolated 1–to–1 hard ties:
              - Consider pairs with tie_left==2 & tie_right==2 only.
              - If both nodes have degree 1 within Δz-bounded subgraph, drop the side
                with smaller compared_to degree (tie → lexicographic).
              - If exactly one node is in hard_tie_cumulative, keep it.

            Phase 2) Collapse remaining multi–multi components:
              - In each connected component inside |Δz| <= thr, pick a winner:
                  (a) single node in hard_tie_cumulative → winner
                  (b) else highest compared_to degree
                  (c) else lexicographic by CRD_ID
                Mark others as losers by setting one of their ties to 0.
            """
            pairs_df = pairs_df.copy()

            both_tie = (pairs_df["tie_left"] == 2) & (pairs_df["tie_right"] == 2)
            z_ok = pairs_df["zleft"].notna() & pairs_df["zright"].notna()
            within = both_tie & z_ok & (
                (pairs_df["zleft"].astype(float) - pairs_df["zright"].astype(float)).abs() <= thr
            )
            if not within.any():
                return pairs_df

            # Phase 1: isolated 1–to–1s
            W1 = pairs_df.loc[within, ["CRD_IDleft", "CRD_IDright"]]
            degL = W1["CRD_IDleft"].value_counts()
            degR = W1["CRD_IDright"].value_counts()

            for idx, row in pairs_df.loc[within].iterrows():
                L = str(row["CRD_IDleft"]); R = str(row["CRD_IDright"])
                if degL.get(L, 0) == 1 and degR.get(R, 0) == 1:
                    left_in_hard  = L in hard_tie_cumulative
                    right_in_hard = R in hard_tie_cumulative
                    n_left  = len(compared_to_dict.get(L, []))
                    n_right = len(compared_to_dict.get(R, []))

                    drop = "right"  # default: keep left
                    if right_in_hard and not left_in_hard:
                        drop = "left"
                    elif (not left_in_hard) and (not right_in_hard):
                        if n_left < n_right:
                            drop = "left"
                        elif n_left == n_right and R < L:
                            drop = "left"

                    if drop == "left":
                        pairs_df.at[idx, "tie_left"] = 0
                    else:
                        pairs_df.at[idx, "tie_right"] = 0

            # Phase 2: collapse multi–multi components
            W = pairs_df.loc[within, ["CRD_IDleft", "CRD_IDright"]].astype(str)
            if not W.empty:
                adj: dict[str, set[str]] = defaultdict(set)
                nodes = set()
                for L, R in W.itertuples(index=False):
                    adj[L].add(R); adj[R].add(L)
                    nodes.add(L); nodes.add(R)

                left_is_loser  = (pairs_df["tie_left"]  == 0)
                right_is_loser = (pairs_df["tie_right"] == 0)

                def node_is_loser(n: str) -> bool:
                    n = str(n)
                    return (
                        ((pairs_df["CRD_IDleft"].astype(str) == n)  & left_is_loser).any()
                        or
                        ((pairs_df["CRD_IDright"].astype(str) == n) & right_is_loser).any()
                    )

                seen: set[str] = set()
                for start in nodes:
                    if start in seen:
                        continue
                    comp: list[str] = []
                    dq = deque([start]); seen.add(start)
                    while dq:
                        u = dq.popleft()
                        comp.append(u)
                        for v in adj[u]:
                            if v not in seen:
                                seen.add(v); dq.append(v)

                    survivors = [n for n in comp if not node_is_loser(n)]
                    if len(survivors) <= 1:
                        continue

                    def degree(n: str) -> int:
                        return len(compared_to_dict.get(str(n), []))

                    hard_in_comp = [n for n in survivors if n in hard_tie_cumulative]
                    if len(hard_in_comp) == 1:
                        winner = hard_in_comp[0]
                    else:
                        winner = sorted(survivors, key=lambda n: (-degree(n), str(n)))[0]

                    for n in survivors:
                        if n == winner:
                            continue
                        maskL = within & (pairs_df["CRD_IDleft"].astype(str)  == n)
                        maskR = within & (pairs_df["CRD_IDright"].astype(str) == n)
                        if maskL.any():
                            first_idx = pairs_df.index[maskL][0]
                            pairs_df.at[first_idx, "tie_left"] = 0
                        elif maskR.any():
                            first_idx = pairs_df.index[maskR][0]
                            pairs_df.at[first_idx, "tie_right"] = 0

            return pairs_df

        pairs = apply_delta_z_fix(pairs)

    # Update losers with any Δz-induced losses (materialized in 'pairs')
    if not pairs.empty:
        more_losers = (
            pd.Index(pairs.loc[pairs["tie_left"] == 0,  "CRD_IDleft"].astype(str))
            .union(pairs.loc[pairs["tie_right"] == 0, "CRD_IDright"].astype(str))
        )
        losers_set.update(list(more_losers))

    # ==============================================================
    # Build per-node priority table to allow "unique-winner" override
    # Lexicographic order follows 'tiebreaking_priority'
    # ==============================================================

    def _priority_series_from_side(ddf: dd.DataFrame, side: str, col: str) -> dd.Series:
        """Return a numeric series for a tiebreaking column on given side."""
        s = ddf[f"{col}{side}"]
        if col == "instrument_type_homogenized":
            # Map instrument type to numeric priority (unknown -> 0)
            return s.map_partitions(
                lambda p: p.map(instrument_type_priority).astype("float64"),
                meta=("prio", "f8"),
            )
        else:
            return dd.to_numeric(s, errors="coerce")

    # Build a (CRD_ID, [priority columns...]) table from left + right sides and aggregate by max
    frames = []
    for side in ("left", "right"):
        series_list = [xmatched_ddf[f"CRD_ID{side}"].astype(str).rename("CRD_ID")]
        for col in tiebreaking_priority:
            series_list.append(_priority_series_from_side(xmatched_ddf, side, col).rename(col))
        frames.append(dd.concat(series_list, axis=1))

    node_prio_dd = dd.concat(frames)
    # Aggregate per-node (max over observed values)
    node_prio_df = node_prio_dd.groupby("CRD_ID")[tiebreaking_priority].max().compute()
    # Use -inf for missing to enforce deterministic lexicographic comparison
    node_prio_df = node_prio_df.astype("float64").replace({np.nan: -np.inf})

    def _pick_unique_winner(ids: list[str]) -> str | None:
        """
        Return a single CRD_ID if there is a unique lexicographic maximum across
        tiebreaking_priority columns. Otherwise, return None.
        """
        if not ids or not tiebreaking_priority:
            return None
        sub = node_prio_df.reindex(ids).fillna(-np.inf)
        cand = sub.index
        for col in tiebreaking_priority:
            col_vals = sub.loc[cand, col]
            m = col_vals.max()
            cand = col_vals.index[col_vals == m]
            if len(cand) == 1:
                return cand[0]
        return None

    # ------------------------------------------------------------------
    # Collapse per-node decisions using the FULL crossmatch graph
    #   - losers (in losers_set) -> 0 (always)
    #   - if a connected component (FULL graph) has ≥2 survivors:
    #         try unique-winner override by priority:
    #             if unique lexicographic max exists -> winner=1, others=0
    #             else -> all survivors=2 (multi survivor)
    #   - if exactly 1 survivor -> 1
    # ------------------------------------------------------------------

    # Initialize final_map with ALL losers so they are applied unconditionally
    final_map: dict[str, int] = {str(n): 0 for n in losers_set}

    # Build full undirected adjacency from compared_to_new (already symmetric)
    adj_full: dict[str, set[str]] = {str(k): set(map(str, v)) for k, v in compared_to_new.items()}

    # Universe of nodes present in the full graph
    all_nodes_full: set[str] = set(adj_full.keys())
    for v in adj_full.values():
        all_nodes_full.update(v)

    if all_nodes_full:
        seen: set[str] = set()
        losers_all = set(map(str, losers_set))

        for start in list(all_nodes_full):
            if start in seen:
                continue

            # BFS over FULL graph (including losers as transit nodes)
            comp: list[str] = []
            dq = deque([start]); seen.add(start)
            while dq:
                u = dq.popleft()
                comp.append(u)
                for w in adj_full.get(u, ()):
                    if w not in seen:
                        seen.add(w); dq.append(w)

            # Survivors in this component (exclude explicit losers)
            comp_survivors = [n for n in comp if n not in losers_all]

            if len(comp_survivors) >= 2:
                # Try unique-winner override by priority
                winner = _pick_unique_winner(comp_survivors)
                if winner is not None:
                    final_map[winner] = 1
                    for n in comp_survivors:
                        if n != winner:
                            final_map[n] = 0
                else:
                    # No unique max → multi-survivor semantics
                    for n in comp_survivors:
                        final_map[n] = 2

            elif len(comp_survivors) == 1:
                final_map[comp_survivors[0]] = 1
            # else: component has only losers → nothing to add (all 0 already)

    # ------------------------------------------------------------------
    # Apply final decisions back to left/right and merge (unchanged)
    # ------------------------------------------------------------------
    def apply_final(df_part: pd.DataFrame) -> pd.DataFrame:
        out = df_part.copy()
        if final_map:
            mapped = out["CRD_ID"].astype(str).map(final_map)
            out["tie_result"] = mapped.fillna(out["tie_result"])
        return out

    left_cat  = left_cat.map_partitions(apply_final,  meta=left_cat._ddf._meta)
    right_cat = right_cat.map_partitions(apply_final, meta=right_cat._ddf._meta)

    left_ddf, right_ddf = left_cat._ddf, right_cat._ddf
    merged = dd.concat([left_ddf, right_ddf])

    # ------------------------------------------------------------------
    # Stabilize dtypes (nullable/Arrow) before saving
    # ------------------------------------------------------------------
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
        "tie_result": DTYPE_INT8,  # align with specz.py
        "z_flag_homogenized": DTYPE_FLOAT,
        "instrument_type_homogenized": DTYPE_STR,
    }
    for col in tiebreaking_priority:
        expected_types[col] = DTYPE_STR if col == "instrument_type_homogenized" else DTYPE_FLOAT
    
    # Add dynamic prev-columns as strings
    for c in map(str, merged.columns):
        if c.startswith("CRD_ID_prev") or c.startswith("compared_to_prev"):
            expected_types[c] = DTYPE_STR
    
    # Apply casts
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

    # ------------------------------------------------------------------
    # (Optional) Sanitize expr columns types, if requested in YAML
    # ------------------------------------------------------------------
    if translation_config.get("save_expr_columns", False):
        schema_hints_raw = translation_config.get("expr_column_schema", {}) or {}
        schema_hints = _normalize_schema_hints(schema_hints_raw)

        standard = {"id", "ra", "dec", "z", "z_flag", "z_err", "instrument_type", "survey"}

        for col, kind in schema_hints.items():
            if col in standard:
                continue

            if col in merged.columns:
                if kind == "str":
                    merged[col] = merged[col].map_partitions(
                        _normalize_string_series_to_na,
                        meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
                    )
                elif kind == "float":
                    coerced = dd.to_numeric(merged[col], errors="coerce")
                    merged[col] = coerced.map_partitions(
                        lambda s: s.astype(DTYPE_FLOAT),
                        meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
                    )
                elif kind == "int":
                    coerced = dd.to_numeric(merged[col], errors="coerce")
                    merged[col] = coerced.map_partitions(
                        lambda s: s.astype(DTYPE_INT),
                        meta=pd.Series(pd.array([], dtype=DTYPE_INT)),
                    )
                elif kind == "bool":
                    merged[col] = merged[col].map_partitions(
                        _to_nullable_boolean_strict,
                        meta=pd.Series(pd.array([], dtype=DTYPE_BOOL)),
                    )
            else:
                if kind == "str":
                    merged = _add_missing_with_dtype(merged, col, DTYPE_STR)
                elif kind == "float":
                    merged = _add_missing_with_dtype(merged, col, DTYPE_FLOAT)
                elif kind == "int":
                    merged = _add_missing_with_dtype(merged, col, DTYPE_INT)
                elif kind == "bool":
                    merged = _add_missing_with_dtype(merged, col, DTYPE_BOOL)

    # --- Save merged parquet ---
    merged_path = os.path.join(temp_dir, f"merged_step{step}")
    _safe_to_parquet(merged, merged_path, write_index=False)

    # --- Optional HATS import ---
    if do_import:
        import_catalog(merged_path, "ra", "dec", f"merged_step{step}_hats", temp_dir, logs_dir, logger, client)
        logger.info("%s: Finished: crossmatch_and_merge id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
        return lsdb.read_hats(os.path.join(temp_dir, f"merged_step{step}_hats")), compared_to_path
    else:
        logger.info("%s: Finished: crossmatch_and_merge id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
        return merged_path, compared_to_path


def crossmatch_tiebreak_safe(
    left_cat,
    right_cat,
    tiebreaking_priority,
    logs_dir,
    temp_dir,
    step,
    client,
    compared_to_left,
    compared_to_right,
    instrument_type_priority,
    translation_config,
    do_import: bool = True,
):
    """Wrapper around :func:`crossmatch_tiebreak` with graceful fallback.

    If the crossmatch raises a known runtime error indicating no overlap or an
    empty output, this function concatenates the inputs, merges the compared_to
    dicts, and proceeds with saving and optional HATS import.

    Args:
        (same as :func:`crossmatch_tiebreak`)

    Returns:
        tuple[object, str]: ``(merged_handle_or_path, compared_to_path)``.
    """
    logger = _build_logger(logs_dir, LOGGER_NAME, LOG_FILE)
    logger.info(
        "%s: Starting: crossmatch_and_merge id=merged_step%s",
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
        step,
    )

    try:
        return crossmatch_tiebreak(
            left_cat=left_cat,
            right_cat=right_cat,
            tiebreaking_priority=tiebreaking_priority,
            logs_dir=logs_dir,
            temp_dir=temp_dir,
            step=step,
            client=client,
            compared_to_left=compared_to_left,
            compared_to_right=compared_to_right,
            instrument_type_priority=instrument_type_priority,
            translation_config=translation_config,
            do_import=do_import,
        )

    except RuntimeError as e:
        # Known conditions: non-overlapping catalogs / empty crossmatch result
        msg = str(e)
        if ("The output catalog is empty" in msg) or ("Catalogs do not overlap" in msg):
            logger.info("%s Proceeding by merging left and right without crossmatching.", msg)

            # Merge input compared_to dicts cumulatively
            compared_to_dict = defaultdict(set)
            for d in [compared_to_left, compared_to_right]:
                for k, v in d.items():
                    compared_to_dict[str(k)].update(map(str, v))

            # Persist compared_to
            compared_to_serializable = {k: sorted(v) for k, v in compared_to_dict.items()}
            compared_to_path = os.path.join(temp_dir, f"compared_to_xmatch_step{step}.json")
            with open(compared_to_path, "w") as f:
                json.dump(compared_to_serializable, f)

            # Concatenate left and right Dask DataFrames
            left_df = left_cat._ddf
            right_df = right_cat._ddf
            merged = dd.concat([left_df, right_df])

            # Save merged result
            merged_path = os.path.join(temp_dir, f"merged_step{step}")
            _safe_to_parquet(merged, merged_path, write_index=False)

            # Optional HATS import
            if do_import:
                import_catalog(
                    merged_path,
                    "ra",
                    "dec",
                    f"merged_step{step}_hats",
                    temp_dir,
                    logs_dir,
                    logger,
                    client,
                )
                logger.info(
                    "%s: Finished: crossmatch_and_merge id=merged_step%s",
                    datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
                    step,
                )
                return lsdb.read_hats(os.path.join(temp_dir, f"merged_step{step}_hats")), compared_to_path
            else:
                logger.info(
                    "%s: Finished: crossmatch_and_merge id=merged_step%s",
                    datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
                    step,
                )
                return merged_path, compared_to_path

        # Unexpected exceptions are re-raised
        raise
