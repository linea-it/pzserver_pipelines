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

# -----------------------
# Third-party
# -----------------------
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
    left_alive = left_cat.query("tie_result != 0")
    right_alive = right_cat.query("tie_result != 0")

    # --- Spatial crossmatch (within radius) ---
    xmatched = left_alive.crossmatch(
        right_alive,
        radius_arcsec=crossmatch_radius,
        n_neighbors=10,
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
        if pd.notna(zflag_left) and zflag_left == 6 and pd.notna(zflag_right) and zflag_right == 6:
            return (0, 0)
        elif pd.notna(zflag_left) and zflag_left == 6:
            return (0, 2 if right_was_tie else 1)
        elif pd.notna(zflag_right) and zflag_right == 6:
            return (2 if left_was_tie else 1, 0)

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
    # Collect pairs for Δz processing and compared_to accumulation
    # ------------------------------------------------------------------
    pairs = xmatched[["CRD_IDleft", "CRD_IDright", "tie_left", "tie_right", "zleft", "zright"]].compute()

    # Build new compared_to links from this step
    compared_to_new = defaultdict(set)
    for _, row in pairs.iterrows():
        left_id = str(row["CRD_IDleft"])
        right_id = str(row["CRD_IDright"])
        compared_to_new[left_id].add(right_id)
        compared_to_new[right_id].add(left_id)

    logger.info(
        "New pairs this step: %d total links in %d objects",
        sum(len(v) for v in compared_to_new.values()),
        len(compared_to_new),
    )

    # Merge with previous compared_to dicts
    compared_to_dict = defaultdict(set)
    for d in [compared_to_left, compared_to_right]:
        for k, v in d.items():
            compared_to_dict[str(k)].update(map(str, v))
    for k, new_vals in compared_to_new.items():
        compared_to_dict[str(k)].update(map(str, new_vals))

    total_links = sum(len(v) for v in compared_to_dict.values())
    logger.info("Compared_to (merged): %d total links across %d objects", total_links, len(compared_to_dict))
    logger.info("Step %s: Compared_to merge completed", step)

    # Persist compared_to
    compared_to_serializable = {k: sorted(v) for k, v in compared_to_dict.items()}
    compared_to_path = os.path.join(temp_dir, f"compared_to_xmatch_step{step}.json")
    with open(compared_to_path, "w") as f:
        json.dump(compared_to_serializable, f)

    # ------------------------------------------------------------------
    # Δz tie resolution (hard ties only)
    # ------------------------------------------------------------------
    # Stable order helps debugging
    pairs = pairs.sort_values(["CRD_IDleft", "CRD_IDright"], kind="mergesort").reset_index(drop=True)
    
    if float(delta_z_threshold) > 0.0:
        thr = float(delta_z_threshold)
    
        def apply_delta_z_fix(pairs_df: pd.DataFrame) -> pd.DataFrame:
            """
            Δz disambiguation in two phases driven by the Δz threshold:
    
            Phase 1) One-to-one resolution inside the subgraph |Δz| <= thr:
              - Consider only pairs that are still hard ties (2,2).
              - If a hard-tie pair is isolated (degree==1 on both sides within the
                thresholded subgraph), drop the side with fewer `compared_to` links
                (break ties lexicographically by CRD_ID).
              - If exactly one node of the pair belongs to the cumulative hard-tie
                set, keep that node and drop the other.
    
            Phase 2) Component collapse inside the same thresholded subgraph:
              - After phase (1), for any connected component (within |Δz| <= thr)
                that still has >= 2 survivors, deterministically pick a single
                winner and mark all other nodes in that component as losers by
                setting at least one of their ties to 0.
              - Winner selection priority:
                  a) If exactly one survivor is in `hard_tie_cumulative` → pick it.
                  b) Otherwise, pick the node with the largest `compared_to` degree.
                  c) Remaining ties → pick by CRD_ID (lexicographic ascending).
            """
            pairs_df = pairs_df.copy()
    
            # Build the |Δz| <= thr mask on the current rows
            both_tie = (pairs_df["tie_left"] == 2) & (pairs_df["tie_right"] == 2)
            z_ok = pairs_df["zleft"].notna() & pairs_df["zright"].notna()
            within = both_tie & z_ok & (
                (pairs_df["zleft"].astype(float) - pairs_df["zright"].astype(float)).abs() <= thr
            )
            if not within.any():
                return pairs_df
    
            # ---------------------------
            # Phase 1: isolated 1–to–1s
            # ---------------------------
            W1 = pairs_df.loc[within, ["CRD_IDleft", "CRD_IDright"]]
            degL = W1["CRD_IDleft"].value_counts()
            degR = W1["CRD_IDright"].value_counts()
    
            for idx, row in pairs_df.loc[within].iterrows():
                L = str(row["CRD_IDleft"])
                R = str(row["CRD_IDright"])
                # Only handle isolated 1–to–1 hard ties inside the Δz-threshold subgraph
                if degL.get(L, 0) == 1 and degR.get(R, 0) == 1:
                    left_in_hard  = L in hard_tie_cumulative
                    right_in_hard = R in hard_tie_cumulative
                    n_left  = len(compared_to_dict.get(L, []))
                    n_right = len(compared_to_dict.get(R, []))
    
                    drop = "right"  # default: keep left
                    if right_in_hard and not left_in_hard:
                        drop = "left"
                    elif (not left_in_hard) and (not right_in_hard):
                        # Drop the node with fewer links; tie → lexicographic
                        if n_left < n_right:
                            drop = "left"
                        elif n_left == n_right and R < L:
                            drop = "left"
    
                    if drop == "left":
                        pairs_df.at[idx, "tie_left"] = 0
                    else:
                        pairs_df.at[idx, "tie_right"] = 0
    
            # ---------------------------------------------------------
            # Phase 2: collapse multi–multi components (|Δz| <= thr)
            # ---------------------------------------------------------
            W = pairs_df.loc[within, ["CRD_IDleft", "CRD_IDright"]].astype(str)
            if not W.empty:
                # Adjacency only from edges within the threshold
                adj: dict[str, set[str]] = defaultdict(set)
                nodes = set()
                for L, R in W.itertuples(index=False):
                    adj[L].add(R); adj[R].add(L)
                    nodes.add(L); nodes.add(R)
    
                # Helpers to check if a node has already been marked as a loser in any pair
                left_is_loser  = (pairs_df["tie_left"]  == 0)
                right_is_loser = (pairs_df["tie_right"] == 0)
    
                def node_is_loser(n: str) -> bool:
                    n = str(n)
                    return (
                        ((pairs_df["CRD_IDleft"].astype(str) == n)  & left_is_loser).any()
                        or
                        ((pairs_df["CRD_IDright"].astype(str) == n) & right_is_loser).any()
                    )
    
                # BFS over connected components inside the Δz subgraph
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
    
                    # Nodes that haven't been eliminated by phase (1)
                    survivors = [n for n in comp if not node_is_loser(n)]
                    if len(survivors) <= 1:
                        continue  # nothing left to resolve in this component
    
                    # Determine a single winner by the stated priority
                    def degree(n: str) -> int:
                        return len(compared_to_dict.get(str(n), []))
    
                    hard_in_comp = [n for n in survivors if n in hard_tie_cumulative]
                    if len(hard_in_comp) == 1:
                        winner = hard_in_comp[0]
                    else:
                        # Highest compared_to degree; tie-break by CRD_ID (ascending)
                        winner = sorted(survivors, key=lambda n: (-degree(n), str(n)))[0]
    
                    # Mark all other survivors as losers by setting at least one tie to 0
                    for n in survivors:
                        if n == winner:
                            continue
                        maskL = within & (pairs_df["CRD_IDleft"].astype(str)  == n)
                        maskR = within & (pairs_df["CRD_IDright"].astype(str) == n)
    
                        # A single 0 is enough to classify the node as a loser
                        if maskL.any():
                            first_idx = pairs_df.index[maskL][0]
                            pairs_df.at[first_idx, "tie_left"] = 0
                        elif maskR.any():
                            first_idx = pairs_df.index[maskR][0]
                            pairs_df.at[first_idx, "tie_right"] = 0
    
            return pairs_df
    
        # Apply the Δz disambiguation
        pairs = apply_delta_z_fix(pairs)
    
    # ------------------------------------------------------------------
    # Collapse per-pair decisions to a single tie_result per CRD_ID
    # Component rule: ≥2 survivors → 2; exactly 1 → 1; losers → 0
    # ------------------------------------------------------------------
    pairs = pairs.sort_values(["CRD_IDleft", "CRD_IDright"], kind="mergesort").reset_index(drop=True)
    
    if pairs.empty:
        final_map: dict[str, int] = {}
    else:
        # Bipartite edges (L–R)
        W = pairs[["CRD_IDleft", "CRD_IDright"]].astype(str)
    
        # Universe of nodes
        all_nodes = pd.Index(pd.unique(pd.concat([W["CRD_IDleft"], W["CRD_IDright"]], ignore_index=True)))
    
        # Losers (0) — if a node lost in any pair, it's a loser
        losers = (
            pd.Index(pairs.loc[pairs["tie_left"] == 0, "CRD_IDleft"].astype(str))
            .union(pairs.loc[pairs["tie_right"] == 0, "CRD_IDright"].astype(str))
        )
    
        # Build adjacency for the full graph (all crossmatched edges, not only within thr)
        adj: dict[str, set[str]] = defaultdict(set)
        for L, R in W.itertuples(index=False):
            adj[L].add(R)
            adj[R].add(L)
    
        # Traverse connected components and assign final decisions
        final_map = {}
        seen: set[str] = set()
    
        for start in all_nodes:
            if start in seen:
                continue
    
            # BFS to gather component nodes
            comp: list[str] = []
            dq = deque([start]); seen.add(start)
            while dq:
                u = dq.popleft()
                comp.append(u)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v); dq.append(v)
    
            comp_set = set(comp)
            comp_losers = comp_set & set(losers)
            comp_survivors = [n for n in comp if n not in comp_losers]
    
            # Mark losers
            for n in comp_losers:
                final_map[n] = 0
    
            # Survivors: ≥2 -> 2, exactly 1 -> 1
            if len(comp_survivors) >= 2:
                for n in comp_survivors:
                    final_map[n] = 2
            elif len(comp_survivors) == 1:
                final_map[comp_survivors[0]] = 1
            # If no survivors, all were marked 0 above
    
    # Persist/extend hard ties (those kept as 2)
    hard_tie_ids = [k for k, v in final_map.items() if v == 2]
    hard_tie_cumulative.update(hard_tie_ids)
    with open(hard_tie_path, "w") as f:
        json.dump(list(hard_tie_cumulative), f)

    # ------------------------------------------------------------------
    # Apply final decisions back to left/right and merge
    # ------------------------------------------------------------------
    def apply_final(df_part: pd.DataFrame) -> pd.DataFrame:
        out = df_part.copy()
        if final_map:
            mapped = out["CRD_ID"].astype(str).map(final_map)
            # Only overwrite where a final decision exists
            out["tie_result"] = mapped.fillna(out["tie_result"])
        return out

    left_cat = left_cat.map_partitions(apply_final, meta=left_cat._ddf._meta)
    right_cat = right_cat.map_partitions(apply_final, meta=right_cat._ddf._meta)

    left_ddf = left_cat._ddf
    right_ddf = right_cat._ddf
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
