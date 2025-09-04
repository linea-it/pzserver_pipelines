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

    Returns:
        tuple[object, str]: ``(merged_handle_or_path, compared_to_path)``.
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

    logger.info("%s: Starting: XMATCH_IN_MEMORY_computation id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
    # --- Pre-filter eliminated rows ---
    left_alive = left_cat.query("tie_result == 1 or tie_result == 2")
    right_alive = right_cat.query("tie_result == 1 or tie_result == 2")

    # --- Spatial crossmatch (within radius) ---
    xmatched = left_alive.crossmatch(
        right_alive,
        radius_arcsec=crossmatch_radius,
        n_neighbors=10,
        suffixes=("left", "right"),
    )

    # ------------------------------------------------------------------
    # Eager compute of xmatched for faster connected-component analysis
    #   - Pull only the columns needed for downstream logic to reduce memory.
    # ------------------------------------------------------------------
    needed_cols = {
        "CRD_IDleft", "CRD_IDright", "zleft", "zright",
        "tie_resultleft", "tie_resultright",
    }
    for col in tiebreaking_priority:
        needed_cols.add(f"{col}left")
        needed_cols.add(f"{col}right")

    xmatched_ddf = xmatched._ddf
    xmatched_df = xmatched_ddf[list(needed_cols)].compute()

    logger.info("%s: Finished: XMATCH_IN_MEMORY_computation id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
    # Normalize IDs to string once (stable keys/comparisons)
    xmatched_df["CRD_IDleft"] = xmatched_df["CRD_IDleft"].astype(str)
    xmatched_df["CRD_IDright"] = xmatched_df["CRD_IDright"].astype(str)

    # Optional: Categorical IDs to reduce memory and speed counts/groupbys
    try:
        xmatched_df["CRD_IDleft"] = pd.Categorical(xmatched_df["CRD_IDleft"], ordered=False)
        xmatched_df["CRD_IDright"] = pd.Categorical(xmatched_df["CRD_IDright"], ordered=False)
    except Exception:
        # Categories are an optimization; ignore if any corner-case arises
        pass

    # --- Hard tie accumulator (persisted across steps) ---
    hard_tie_path = os.path.join(temp_dir, "hard_tie_ids.json")
    if os.path.exists(hard_tie_path):
        with open(hard_tie_path, "r") as f:
            hard_tie_cumulative = set(json.load(f))
    else:
        hard_tie_cumulative = set()

    # ------------------------------------------------------------------
    # Vectorized tie decision on the eager in-memory pairs
    #   - Lexicographic priority across columns in tiebreaking_priority.
    #   - Missing values rank as -inf; "first non-NaN side" rule preserved.
    #   - If no information: (2, 2) hard tie.
    #   - Previous tie (==2) maintains 2 for the chosen side; else 1.
    # ------------------------------------------------------------------
    logger.info("%s: Starting: TIE_BREAKING_computation id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
    n = len(xmatched_df)
    tie_left = np.full(n, 2, dtype=np.int8)
    tie_right = np.full(n, 2, dtype=np.int8)

    left_was_tie = (
        (pd.to_numeric(xmatched_df.get("tie_resultleft"), errors="coerce") == 2).to_numpy()
        if "tie_resultleft" in xmatched_df
        else np.zeros(n, dtype=bool)
    )
    right_was_tie = (
        (pd.to_numeric(xmatched_df.get("tie_resultright"), errors="coerce") == 2).to_numpy()
        if "tie_resultright" in xmatched_df
        else np.zeros(n, dtype=bool)
    )

    def _to_numeric_priority(colname: str, side: str) -> np.ndarray:
        s = xmatched_df[f"{colname}{side}"]
        if colname == "instrument_type_homogenized":
            return s.map(instrument_type_priority).astype("float64").to_numpy()
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")

    if tiebreaking_priority:
        left_mat = []
        right_mat = []
        left_has_any = np.zeros(n, dtype=bool)
        right_has_any = np.zeros(n, dtype=bool)

        for col in tiebreaking_priority:
            lv = _to_numeric_priority(col, "left")
            rv = _to_numeric_priority(col, "right")
            left_mat.append(lv)
            right_mat.append(rv)
            left_has_any |= ~np.isnan(lv)
            right_has_any |= ~np.isnan(rv)

        L = np.stack(left_mat, axis=1)   # (n, k)
        R = np.stack(right_mat, axis=1)  # (n, k)
        L_cmp = np.where(np.isnan(L), -np.inf, L)
        R_cmp = np.where(np.isnan(R), -np.inf, R)

        unresolved = np.ones(n, dtype=bool)
        for i in range(L_cmp.shape[1]):
            li = L_cmp[:, i]
            ri = R_cmp[:, i]
            left_better = (li > ri) & unresolved
            right_better = (ri > li) & unresolved

            if left_better.any():
                tie_left[left_better] = np.where(left_was_tie[left_better], 2, 1).astype(np.int8)
                tie_right[left_better] = 0
            if right_better.any():
                tie_left[right_better] = 0
                tie_right[right_better] = np.where(right_was_tie[right_better], 2, 1).astype(np.int8)

            unresolved &= ~(left_better | right_better)
            if not unresolved.any():
                break

        if unresolved.any():
            prefer_left = unresolved & left_has_any & ~right_has_any
            prefer_right = unresolved & right_has_any & ~left_has_any

            if prefer_left.any():
                tie_left[prefer_left] = np.where(left_was_tie[prefer_left], 2, 1).astype(np.int8)
                tie_right[prefer_left] = 0
                unresolved &= ~prefer_left
            if prefer_right.any():
                tie_left[prefer_right] = 0
                tie_right[prefer_right] = np.where(right_was_tie[prefer_right], 2, 1).astype(np.int8)
                unresolved &= ~prefer_right
            # Remaining unresolved keep (2,2)
    # else: no criteria -> stay (2,2) to be handled by Δz

    xmatched_df["tie_left"] = tie_left
    xmatched_df["tie_right"] = tie_right

    # ------------------------------------------------------------------
    # Construct compared_to edges (undirected) in-memory and persist
    # ------------------------------------------------------------------
    def _build_undirected_adjacency(left_ids: pd.Series, right_ids: pd.Series) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {}
        L = left_ids.to_numpy(dtype=object)
        R = right_ids.to_numpy(dtype=object)
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

    compared_to_new = _build_undirected_adjacency(xmatched_df["CRD_IDleft"], xmatched_df["CRD_IDright"])

    logger.info(
        "New pairs (eager agg): %d total links in %d objects",
        sum(len(v) for v in compared_to_new.values()),
        len(compared_to_new),
    )

    # Merge with previous dicts
    compared_to_dict = defaultdict(set)
    for d in (compared_to_left, compared_to_right, compared_to_new):
        for k, v in d.items():
            if isinstance(v, set):
                compared_to_dict[str(k)].update(map(str, v))
            else:
                compared_to_dict[str(k)].update(map(str, v))

    compared_to_serializable = {k: sorted(v) for k, v in compared_to_dict.items()}
    compared_to_path = os.path.join(temp_dir, f"compared_to_xmatch_step{step}.json")
    with open(compared_to_path, "w") as f:
        json.dump(compared_to_serializable, f)

    total_links = sum(len(v) for v in compared_to_dict.values())
    logger.info("Compared_to (merged): %d total links across %d objects", total_links, len(compared_to_dict))
    logger.info("Step %s: Compared_to merge completed", step)

    # === Identify losers early ===
    losers_set = set(
        xmatched_df.loc[xmatched_df["tie_left"] == 0, "CRD_IDleft"].astype(str)
    ) | set(
        xmatched_df.loc[xmatched_df["tie_right"] == 0, "CRD_IDright"].astype(str)
    )

    # === Prepare survivors for Δz resolution ===
    survivors_mask = (xmatched_df["tie_left"] != 0) & (xmatched_df["tie_right"] != 0)
    pairs = xmatched_df.loc[survivors_mask, ["CRD_IDleft", "CRD_IDright", "tie_left", "tie_right", "zleft", "zright"]].copy()

    logger.info("%s: Finished: TIE_BREAKING_computation id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
    logger.info("%s: Starting: DELTAZ_computation id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
    # ------------------------------------------------------------------
    # Δz tie resolution (hard ties only) on the survivors set (in-memory)
    #   Phase 1 vetorizada; Phase 2 igual à original.
    # ------------------------------------------------------------------
    if float(delta_z_threshold) > 0.0 and not pairs.empty:
        thr = float(delta_z_threshold)

        def apply_delta_z_fix(pairs_df: pd.DataFrame) -> pd.DataFrame:
            """
            Δz disambiguation within the subgraph |Δz| <= thr:
        
            Phase 1) Isolated 1–to–1 hard ties (vetorizado, posicional).
            Phase 2) Collapse multi–multi components (union-find/DSU em inteiros).
            Regras de desempate inalteradas:
              (a) único em hard_tie_cumulative → winner
              (b) maior grau em compared_to_dict
              (c) lexicográfico por CRD_ID
            """
            t0 = datetime.now()
            df = pairs_df.copy()
        
            # ---- Subgrafo Δz (somente hard ties 2/2 com z numérico e |Δz|<=thr) ----
            both_tie = (df["tie_left"] == 2) & (df["tie_right"] == 2)
            z_ok = df["zleft"].notna() & df["zright"].notna()
            dz = (pd.to_numeric(df["zleft"], errors="coerce") - pd.to_numeric(df["zright"], errors="coerce")).abs()
            within_mask = both_tie & z_ok & (dz <= thr)
        
            if not bool(within_mask.any()):
                logger.info("Δz: within=0 → nada a fazer")
                return df
        
            # POSIÇÕES (não rótulos) das linhas dentro do subgrafo
            pos_within = np.flatnonzero(within_mask.to_numpy())
            W = df.iloc[pos_within, :][["CRD_IDleft", "CRD_IDright"]]
        
            # IDs como strings para operações de comparação hash/lex
            L_ids = W["CRD_IDleft"].astype(str).to_numpy()
            R_ids = W["CRD_IDright"].astype(str).to_numpy()
        
            logger.info("Δz: within edges=%d, unique nodes=%d",
                        len(pos_within), len(np.unique(np.concatenate([L_ids, R_ids]))))
        
            # Índices de coluna para setitem posicional rápido
            col_tie_left  = df.columns.get_loc("tie_left")
            col_tie_right = df.columns.get_loc("tie_right")
        
            # =========================
            # Phase 1: 1–para–1 vetorizado (posicional)
            # =========================
            uniqL, cntL = np.unique(L_ids, return_counts=True)
            uniqR, cntR = np.unique(R_ids, return_counts=True)
            degL_map = dict(zip(uniqL, cntL))
            degR_map = dict(zip(uniqR, cntR))
        
            degL_arr = np.fromiter((degL_map.get(x, 0) for x in L_ids), count=L_ids.size, dtype=np.int32)
            degR_arr = np.fromiter((degR_map.get(x, 0) for x in R_ids), count=R_ids.size, dtype=np.int32)
            isolated = (degL_arr == 1) & (degR_arr == 1)
        
            loser_nodes = set()
            if bool(isolated.any()):
                nodes_iso = set(np.concatenate([L_ids[isolated], R_ids[isolated]]))
                degree_cache = {n: len(compared_to_dict.get(n, [])) for n in nodes_iso}
        
                L_degcmp = np.fromiter((degree_cache.get(x, 0) for x in L_ids), count=L_ids.size, dtype=np.int32)
                R_degcmp = np.fromiter((degree_cache.get(x, 0) for x in R_ids), count=R_ids.size, dtype=np.int32)
                L_inhard = np.fromiter((x in hard_tie_cumulative for x in L_ids), count=L_ids.size, dtype=bool)
                R_inhard = np.fromiter((x in hard_tie_cumulative for x in R_ids), count=R_ids.size, dtype=bool)
        
                drop_left  = isolated & (~L_inhard) & (R_inhard)
                drop_right = isolated & ( L_inhard) & (~R_inhard)
        
                undec = isolated & ~(drop_left | drop_right)
                if bool(undec.any()):
                    less_left  = undec & (L_degcmp < R_degcmp)
                    less_right = undec & (R_degcmp < L_degcmp)
                    eq_deg     = undec & ~(less_left | less_right)
                    # Lexicográfico quando empata por grau
                    lex_drop_left  = eq_deg & (R_ids < L_ids)
                    lex_drop_right = eq_deg & ~lex_drop_left
                    drop_left  |= (less_left  | lex_drop_left)
                    drop_right |= (less_right | lex_drop_right)
        
                # Atribuição POSICIONAL
                if bool(drop_left.any()):
                    df.iloc[pos_within[drop_left],  col_tie_left]  = 0
                    loser_nodes.update(L_ids[drop_left].tolist())
                if bool(drop_right.any()):
                    df.iloc[pos_within[drop_right], col_tie_right] = 0
                    loser_nodes.update(R_ids[drop_right].tolist())
        
                logger.info("Δz: phase1 isolated=%d → drop_left=%d, drop_right=%d",
                            int(isolated.sum()), int(drop_left.sum()), int(drop_right.sum()))
            else:
                logger.info("Δz: phase1 isolated=0")
        
            # =========================
            # Phase 2: multi–multi (union-find em inteiros)
            # =========================
            # >>> Mais rápido que np.isin em object: membership por set + fromiter
            losers_set = loser_nodes  # já é um set
            maskL = np.fromiter((x in losers_set for x in L_ids), count=L_ids.size, dtype=bool)
            maskR = np.fromiter((y in losers_set for y in R_ids), count=R_ids.size, dtype=bool)
            keep_edge = ~(maskL | maskR)
        
            survivors_edges = int(keep_edge.sum())
            if survivors_edges == 0:
                logger.info("Δz: phase2 survivor edges=0 → nada a colapsar")
                logger.info("Δz: total elapsed %ss", (datetime.now() - t0).total_seconds())
                return df
        
            L_surv = L_ids[keep_edge]
            R_surv = R_ids[keep_edge]
            pos_surv = pos_within[keep_edge]
        
            # Compacta nós para inteiros (mais rápido que dict: factorize)
            combined = np.concatenate([L_surv, R_surv])
            codes, uniques = pd.factorize(combined, sort=False)
            U = codes[:L_surv.size].astype(np.int32, copy=False)
            V = codes[L_surv.size:].astype(np.int32, copy=False)
            all_nodes = np.asarray(uniques)   # mapeia índice -> CRD_ID
            m = all_nodes.size
        
            # DSU
            parent = np.arange(m, dtype=np.int32)
            rank = np.zeros(m, dtype=np.int8)
            def find(a: int) -> int:
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            def union(a: int, b: int) -> None:
                ra = find(a); rb = find(b)
                if ra == rb: return
                if rank[ra] < rank[rb]:
                    parent[ra] = rb
                elif rank[ra] > rank[rb]:
                    parent[rb] = ra
                else:
                    parent[rb] = ra
                    rank[ra] += 1
            for a, b in zip(U, V):
                union(a, b)
        
            roots = np.fromiter((find(i) for i in range(m)), count=m, dtype=np.int32)
            comp_map: dict[int, list[int]] = {}
            for i, r in enumerate(roots):
                comp_map.setdefault(int(r), []).append(i)
        
            logger.info("Δz: phase2 survivors edges=%d, survivors nodes=%d, components=%d",
                        survivors_edges, m, len(comp_map))
        
            if not comp_map:
                logger.info("Δz: total elapsed %ss", (datetime.now() - t0).total_seconds())
                return df
        
            # Para marcar losers em O(1), guarde a primeira posição (posicional) por nó
            first_left_pos: dict[str, int] = {}
            first_right_pos: dict[str, int] = {}
            for j, (lid, rid) in enumerate(zip(L_ids, R_ids)):
                pj = int(pos_within[j])
                if lid not in first_left_pos:
                    first_left_pos[lid] = pj
                if rid not in first_right_pos:
                    first_right_pos[rid] = pj
        
            # Helpers
            def degree(n: str) -> int:
                return len(compared_to_dict.get(n, []))
            hard_set = hard_tie_cumulative
        
            # Resolve cada componente
            for nodes_ix in comp_map.values():
                comp_nodes = [all_nodes[i] for i in nodes_ix]
                if len(comp_nodes) <= 1:
                    continue
        
                hard_in_comp = [n for n in comp_nodes if n in hard_set]
                if len(hard_in_comp) == 1:
                    winner = hard_in_comp[0]
                else:
                    winner = max(comp_nodes, key=lambda n: (degree(n), n))
        
                # Marque perdedores em um único par (posicional)
                for n in comp_nodes:
                    if n == winner:
                        continue
                    p = first_left_pos.get(n)
                    if p is not None:
                        df.iat[p, col_tie_left] = 0
                    else:
                        p = first_right_pos.get(n)
                        if p is not None:
                            df.iat[p, col_tie_right] = 0
        
            logger.info("Δz: total elapsed %ss", (datetime.now() - t0).total_seconds())
            return df
            

        pairs = apply_delta_z_fix(pairs)

        # update losers_set with Δz-induced losses
        if not pairs.empty:
            more_losers = set(pairs.loc[pairs["tie_left"] == 0,  "CRD_IDleft"].astype(str)) | \
                          set(pairs.loc[pairs["tie_right"] == 0, "CRD_IDright"].astype(str))
            losers_set |= more_losers

    logger.info("%s: Finished: DELTAZ_computation id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
    logger.info("%s: Starting: OVERRIDE_computation id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
    # ==============================================================
    # Build per-node priority table to allow "unique-winner" override
    #   - Only for nodes that could survive: all_nodes_full - losers_set
    #   - Lexicographic order follows 'tiebreaking_priority'
    # ==============================================================
    # Full adjacency from compared_to_new (already symmetric)
    adj_full: dict[str, set[str]] = {str(k): set(map(str, v)) for k, v in compared_to_new.items()}

    all_nodes_full: set[str] = set(adj_full.keys())
    for v in adj_full.values():
        all_nodes_full.update(v)

    node_prio_df = None
    if tiebreaking_priority and all_nodes_full:
        nodes_need_prio = {n for n in all_nodes_full if n not in losers_set}
        if nodes_need_prio:
            frames = []
            for side in ("left", "right"):
                sid = f"CRD_ID{side}"
                mask = xmatched_df[sid].astype(str).isin(nodes_need_prio)
                if not mask.any():
                    continue
                data = {"CRD_ID": xmatched_df.loc[mask, sid].astype(str)}
                for col in tiebreaking_priority:
                    s = xmatched_df.loc[mask, f"{col}{side}"]
                    if col == "instrument_type_homogenized":
                        data[col] = s.map(instrument_type_priority).astype("float64")
                    else:
                        data[col] = pd.to_numeric(s, errors="coerce").astype("float64")
                frames.append(pd.DataFrame(data))
            node_long = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(columns=["CRD_ID", *tiebreaking_priority])
            node_prio_df = node_long.groupby("CRD_ID", sort=False)[tiebreaking_priority].max()
            node_prio_df = node_prio_df.astype("float64").replace({np.nan: -np.inf})

    def _pick_unique_winner(ids: list[str]) -> str | None:
        """Return a single CRD_ID if it is a unique lexicographic maximum; else None."""
        if not ids or not tiebreaking_priority or node_prio_df is None:
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
    # ------------------------------------------------------------------
    final_map: dict[str, int] = {str(n): 0 for n in losers_set}

    if all_nodes_full:
        seen: set[str] = set()
        losers_all = set(map(str, losers_set))

        for start in list(all_nodes_full):
            if start in seen:
                continue

            comp: list[str] = []
            dq = deque([start]); seen.add(start)
            while dq:
                u = dq.popleft()
                comp.append(u)
                for w in adj_full.get(u, ()):
                    if w not in seen:
                        seen.add(w); dq.append(w)

            comp_survivors = [n for n in comp if n not in losers_all]

            if len(comp_survivors) >= 2:
                winner = _pick_unique_winner(comp_survivors)
                if winner is not None:
                    final_map[winner] = 1
                    for n in comp_survivors:
                        if n != winner:
                            final_map[n] = 0
                else:
                    for n in comp_survivors:
                        final_map[n] = 2
            elif len(comp_survivors) == 1:
                final_map[comp_survivors[0]] = 1
            # else: only losers in this component -> keep zeros

    logger.info("%s: Finished: OVERRIDE_computation id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
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
        "tie_result": DTYPE_INT8,
        "z_flag_homogenized": DTYPE_FLOAT,
        "instrument_type_homogenized": DTYPE_STR,
    }
    for col in tiebreaking_priority:
        expected_types[col] = DTYPE_STR if col == "instrument_type_homogenized" else DTYPE_FLOAT
    for c in map(str, merged.columns):
        if c.startswith("CRD_ID_prev") or c.startswith("compared_to_prev"):
            expected_types[c] = DTYPE_STR

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

    # (Optional) Sanitize expr columns types, if requested in YAML
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
        logger.info("%s: Starting: IMPORT_computation id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
        import_catalog(merged_path, "ra", "dec", f"merged_step{step}_hats", temp_dir, logs_dir, logger, client)
        logger.info("%s: Finished: IMPORT_computation id=merged_step%s", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), step)
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
