# deduplication.py
from __future__ import annotations
"""Deduplication for Combine Redshift Catalogs (CRC).

Builds a graph from ``CRD_ID <-> compared_to`` and resolves duplicates with
configurable priorities. Provides a pandas solver (`deduplicate_pandas`) and a
Dask/LSDB per-partition driver (`run_dedup_with_lsdb_map_partitions`).
"""

# ==============================
# Imports
# ==============================
from typing import (
    Iterable,
    Mapping,
    Sequence,
    Optional,
    Dict,
    List,
    Tuple,
)
import math
import logging
import time

import dask
import numpy as np
import pandas as pd
import dask.dataframe as dd

from combine_redshift_dedup.packages.utils import get_phase_logger, log_phase

# ==============================
# Logger (child of 'crc')
# ==============================
LOGGER_NAME = "crc.dedup"


def _base_logger() -> logging.Logger:
    """Return the child base logger ('crc.dedup')."""
    lg = logging.getLogger(LOGGER_NAME)
    lg.setLevel(logging.NOTSET)
    lg.propagate = True
    return lg


def _phase_logger() -> logging.LoggerAdapter:
    """Return a phase-aware logger (phase='deduplication')."""
    return get_phase_logger("deduplication", _base_logger())


# ==============================
# Public API
# ==============================
__all__ = [
    "deduplicate_pandas",
    "run_dedup_with_lsdb_map_partitions",
]

# ==============================
# Small helpers: string/parse/score
# ==============================
def _canon_id_series(s: pd.Series) -> pd.Series:
    """Return canonical CRD-like IDs (strip and remove zero-width chars)."""
    t = s.astype("string")
    t = t.str.replace(r"[\u200B-\u200D\uFEFF]", "", regex=True)
    return t.str.strip()


def _norm_str(x) -> str | None:
    """Return a normalized string or None."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    return s if s else None


def _parse_compared_to_cell(val) -> List[str]:
    """Parse a single `compared_to` cell into a list of CRD_ID strings."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    s = str(val).strip()
    if not s:
        return []
    return [t for t in (tok.strip() for tok in s.split(",")) if t]


def _score_instrument_type(series: pd.Series, priority_map: Mapping[str, int]) -> pd.Series:
    """Score instrument types using a priority map."""
    norm_map = {str(k).strip().lower(): int(v) for k, v in priority_map.items()}

    def _score_one(v):
        v = _norm_str(v)
        return norm_map.get(v, 0) if v is not None else 0

    return series.map(_score_one).astype("int64")


# ==============================
# Graph building
# ==============================
def _split_cmp_vectorized(s: pd.Series) -> pd.Series:
    """Split `compared_to` values vectorially into lists."""
    s = s.astype("string")
    return s.str.split(",")


def _build_edges_fast(
    df: pd.DataFrame,
    *,
    crd_col: str,
    compared_col: str,
    zf_series: Optional[pd.Series] = None,
    edge_log: bool = False,
):
    """Build undirected edges among NON-STAR rows (vectorized path).

    Returns:
        (nodes_index, edges_uv, diag) where:
          - nodes_index: node labels for the condensed graph
          - edges_uv: (E, 2) int32 array of undirected edges (u < v)
          - diag: dict with basic diagnostics (counts). If `edge_log` is False,
                  only cheap counts are filled; expensive ones are set to None.
    """
    # --- filter A-side (rows) to non-stars
    non_star_mask = pd.Series(True, index=df.index)
    if zf_series is not None:
        non_star_mask &= ~pd.to_numeric(zf_series, errors="coerce").eq(6)

    A_df = df.loc[non_star_mask, [crd_col, compared_col]].copy().reset_index(drop=True)
    if A_df.empty:
        diag = {
            "edge_log_enabled": bool(edge_log),
            "n_rows_nonstar": int(non_star_mask.sum()),
            "n_rows_star_excluded": int((~non_star_mask).sum()),
            "n_edges_raw": 0,
            "n_edges_kept": 0,
            "n_edges_starB_excluded": None if not edge_log else 0,
        }
        return pd.Index([], dtype="object"), np.empty((0, 2), dtype=np.int32), diag

    # Non-star IDs present on A-side
    present_nonstar = set(
        _canon_id_series(df.loc[non_star_mask, crd_col]).dropna().unique()
    )

    # If diagnostics are enabled, precompute star IDs (for B-side exclusion count)
    if edge_log and (zf_series is not None):
        star_ids = set(
            _canon_id_series(
                df.loc[pd.to_numeric(zf_series, errors="coerce").eq(6), crd_col]
            ).dropna().unique()
        )
    else:
        star_ids = None  # skip expensive star-neighbor counting

    # explode compared_to
    cmp_lists = _split_cmp_vectorized(A_df[compared_col])
    edges_raw = (
        A_df[[crd_col]]
        .assign(v=cmp_lists)
        .rename(columns={crd_col: "u"})
        .explode("v", ignore_index=True)
    )

    # canonicalize ids
    edges_raw["u"] = _canon_id_series(edges_raw["u"])
    edges_raw["v"] = _canon_id_series(edges_raw["v"])

    # drop empties
    edges_raw = edges_raw.dropna(subset=["u", "v"])
    edges_raw = edges_raw[(edges_raw["u"] != "") & (edges_raw["v"] != "")]
    n_edges_raw = int(len(edges_raw))

    # Optional: count how many B-neighbors were stars (diagnostic only)
    if edge_log and (star_ids is not None) and n_edges_raw:
        n_edges_starB_excluded = int(edges_raw["v"].isin(star_ids).sum())
    else:
        n_edges_starB_excluded = None

    # keep only neighbors that are non-stars present on A-side
    edges_raw = edges_raw[edges_raw["v"].isin(present_nonstar)]
    if edges_raw.empty:
        diag = {
            "edge_log_enabled": bool(edge_log),
            "n_rows_nonstar": int(non_star_mask.sum()),
            "n_rows_star_excluded": int((~non_star_mask).sum()),
            "n_edges_raw": n_edges_raw,
            "n_edges_kept": 0,
            "n_edges_starB_excluded": n_edges_starB_excluded,
        }
        return pd.Index([], dtype="object"), np.empty((0, 2), dtype=np.int32), diag

    # unique undirected edges (u < v)
    nodes_edge = pd.Index(pd.unique(pd.concat([edges_raw["u"], edges_raw["v"]], ignore_index=True)))

    # --- EXTRA LOG (edge_log): sanity-check that no star IDs leaked into edge nodes
    if edge_log and (zf_series is not None) and len(nodes_edge):
        star_ids_fast = set(
            _canon_id_series(
                df.loc[pd.to_numeric(zf_series, errors="coerce").eq(6), crd_col]
            ).dropna().unique()
        )
        leaked = [cid for cid in nodes_edge.astype("string") if cid in star_ids_fast]
        if leaked:
            _phase_logger().error(
                "Star IDs leaked into fast-path edge nodes (logic violation): "
                "count=%d sample=%s",
                len(leaked), leaked[:5],
            )
            
    id2ix = {cid: i for i, cid in enumerate(nodes_edge)}
    u = edges_raw["u"].map(id2ix).to_numpy(dtype=np.int32, copy=False)
    v = edges_raw["v"].map(id2ix).to_numpy(dtype=np.int32, copy=False)

    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    mask = lo != hi
    lo, hi = lo[mask], hi[mask]
    if lo.size == 0:
        diag = {
            "edge_log_enabled": bool(edge_log),
            "n_rows_nonstar": int(non_star_mask.sum()),
            "n_rows_star_excluded": int((~non_star_mask).sum()),
            "n_edges_raw": n_edges_raw,
            "n_edges_kept": 0,
            "n_edges_starB_excluded": n_edges_starB_excluded,
        }
        return nodes_edge, np.empty((0, 2), dtype=np.int32), diag

    uv = np.stack([lo, hi], axis=1)
    view = uv.view([("x", uv.dtype), ("y", uv.dtype)])
    uv = np.unique(view).view(uv.dtype).reshape(-1, 2)

    diag = {
        "edge_log_enabled": bool(edge_log),
        "n_rows_nonstar": int(non_star_mask.sum()),
        "n_rows_star_excluded": int((~non_star_mask).sum()),
        "n_edges_raw": n_edges_raw,
        "n_edges_kept": int(uv.shape[0]),
        # None when edge_log=False to indicate we skipped the costly check
        "n_edges_starB_excluded": n_edges_starB_excluded,
    }
    return nodes_edge, uv, diag


def _connected_components_scipy(n_nodes: int, edges_uv: np.ndarray) -> np.ndarray:
    """Compute connected components using SciPy."""
    if n_nodes == 0:
        return np.array([], dtype=np.int64)
    if edges_uv.size == 0:
        return np.arange(n_nodes, dtype=np.int64)

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    data = np.ones(edges_uv.shape[0], dtype=np.int8)
    A = coo_matrix((data, (edges_uv[:, 0], edges_uv[:, 1])), shape=(n_nodes, n_nodes))
    A = A + A.T
    _, labels = connected_components(A, directed=False, return_labels=True)
    return labels


def _build_edges_pdf(df: pd.DataFrame, crd_col: str, compared_col: str) -> pd.DataFrame:
    """Build undirected edges (u, v) from `compared_to` (fallback)."""
    if df.empty:
        return pd.DataFrame({"u": pd.Series([], dtype="string"),
                             "v": pd.Series([], dtype="string")})

    present_ids = set(df[crd_col].astype(str))
    a = df[crd_col].astype(str)
    b_lists = df[compared_col].apply(_parse_compared_to_cell)
    edges = (
        pd.DataFrame({"u": a, "v": b_lists})
        .explode("v", ignore_index=True)
        .dropna(subset=["v"])
    )
    if edges.empty:
        return pd.DataFrame({"u": pd.Series([], dtype="string"),
                             "v": pd.Series([], dtype="string")})

    edges["v"] = edges["v"].astype(str)
    edges = edges[edges["v"].isin(present_ids)]
    u = edges["u"].astype(str)
    v = edges["v"].astype(str)
    lo = np.minimum(u.values, v.values)
    hi = np.maximum(u.values, v.values)
    mask = lo != hi
    if not mask.any():
        return pd.DataFrame({"u": pd.Series([], dtype="string"),
                             "v": pd.Series([], dtype="string")})

    out = (
        pd.DataFrame({"u": pd.Series(lo[mask], dtype="string"),
                      "v": pd.Series(hi[mask], dtype="string")})
        .drop_duplicates(ignore_index=True)
    )
    return out


# ==============================
# Δz clustering + CC fallback (DSU)
# ==============================
def _collapse_within_dz(
    mask: pd.Series,
    gid: pd.Series,
    zvals: pd.Series,
    crd_s: pd.Series,
    threshold: float,
) -> pd.Series:
    """Collapse contiguous survivors in z within threshold; keep one per cluster."""
    thr = float(threshold or 0.0)
    if thr <= 0.0:
        return mask

    m = mask.to_numpy(dtype=bool, na_value=False)
    if not m.any():
        return mask

    n = len(mask)
    pos_all = np.arange(n, dtype=np.int64)
    pos = pos_all[m]

    gid_arr = pd.Index(gid).to_numpy() if isinstance(gid, pd.Series) else np.asarray(gid)
    z_arr = pd.to_numeric(zvals, errors="coerce").to_numpy()
    crd_arr = crd_s.astype(str).to_numpy()

    pos_def = pos[~np.isnan(z_arr[pos])]
    if pos_def.size == 0:
        return mask

    order = np.lexsort((crd_arr[pos_def], z_arr[pos_def], gid_arr[pos_def]))
    pos_sorted = pos_def[order]
    g_sorted = gid_arr[pos_sorted]
    z_sorted = z_arr[pos_sorted]
    crd_sorted = crd_arr[pos_sorted]

    k = len(pos_sorted)
    new_gid = np.empty(k, dtype=bool)
    new_gid[0] = True
    if k > 1:
        new_gid[1:] = (g_sorted[1:] != g_sorted[:-1])

    z_jump = np.empty(k, dtype=float)
    z_jump[0] = np.inf
    if k > 1:
        z_jump[1:] = np.abs(z_sorted[1:] - z_sorted[:-1])

    is_break = new_gid | (z_jump > thr)
    clus = np.cumsum(is_break)

    sub = pd.DataFrame({"pos": pos_sorted, "gid": g_sorted, "clus": clus, "crd": crd_sorted})
    min_crd = sub.groupby(["gid", "clus"], sort=False)["crd"].transform("min")
    keep_mask = (sub["crd"] == min_crd)
    winners_pos = sub.loc[keep_mask].groupby(["gid", "clus"], sort=False).head(1)["pos"].to_numpy()

    out_np = m.copy()
    out_np[pos_def] = False
    out_np[winners_pos] = True
    return pd.Series(out_np, index=mask.index)


class _DSU:
    """Disjoint Set Union for connected components."""

    __slots__ = ("p",)

    def __init__(self, n: int):
        """Initialize DSU with `n` elements."""
        self.p = list(range(n))

    def find(self, a: int) -> int:
        """Find set representative with path compression."""
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        """Union sets of a and b (attach higher index under lower)."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


def _connected_components(
    nodes: Iterable[str],
    edges: pd.DataFrame,
) -> Dict[str, int]:
    """Compute connected components over nodes using undirected edges."""
    nodes = list(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    dsu = _DSU(n)

    if not edges.empty:
        u = edges["u"].astype(str).values
        v = edges["v"].astype(str).values
        for uu, vv in zip(u, v):
            iu = idx.get(uu)
            iv = idx.get(vv)
            if iu is not None and iv is not None:
                dsu.union(iu, iv)

    roots = [dsu.find(i) for i in range(n)]
    root_to_gid: Dict[int, int] = {}
    next_gid = 0
    out: Dict[str, int] = {}
    for node, r in zip(nodes, roots):
        if r not in root_to_gid:
            root_to_gid[r] = next_gid
            next_gid += 1
        out[node] = root_to_gid[r]
    return out


# ==============================
# 1-D safe helpers (numeric)
# ==============================
def _series_1d_from(df_or_series) -> pd.Series:
    """Ensure a 1-D Series when a duplicated-name selection returns a DataFrame."""
    if isinstance(df_or_series, pd.DataFrame):
        return df_or_series.iloc[:, 0]
    return df_or_series


def _to_numeric(series_like) -> pd.Series:
    """Coerce to float (1-D safe, NaN for non-castable)."""
    s = _series_1d_from(series_like)
    return pd.to_numeric(s, errors="coerce")


# ==============================
# Guard Restore
# ==============================
def _only_star_neighbors_series(col: pd.Series, star_ids: set[str]) -> pd.Series:
    """True when compared_to is non-empty AND all neighbors are star IDs."""
    s = col.astype("string").fillna("")
    lst = s.str.split(",")
    out = []
    for tokens in lst:
        toks = [t.strip() for t in tokens if t and t.strip()]
        out.append(bool(toks) and all((tok in star_ids) for tok in toks))
    return pd.Series(out, index=col.index, dtype="boolean")


def _apply_guard_restore_local(
    df: pd.DataFrame,
    *,
    crd_col: str,
    compared_col: str,
    zf_series: Optional[pd.Series],
    tie_col: str,
    tie_col_orig: str,
) -> pd.DataFrame:
    """Restore original tie_result for non-stars with empty compared_to OR only-star neighbors."""
    if tie_col_orig not in df.columns:
        return df  # nada a restaurar

    # estrela (fixos em 3) e máscara de compared_to vazio
    is_star = pd.Series(False, index=df.index)
    if zf_series is not None:
        is_star = pd.to_numeric(zf_series, errors="coerce").eq(6.0)

    cmp_str   = df[compared_col].astype("string")
    cmp_empty = cmp_str.isna() | cmp_str.str.strip().eq("")

    # conjunto local de IDs de estrelas nesta partição/view
    star_ids = set(df.loc[is_star, crd_col].astype("string"))

    only_star_neighbors = (~cmp_empty) & _only_star_neighbors_series(df[compared_col], star_ids)

    # regra pedida:
    # - se objeto é estrela -> mantém tie=3 (não restaurar)
    # - se não-estrela e (compared_to vazio OU só vizinhos estrelas) -> restaurar
    restore_mask = (~is_star) & (cmp_empty | only_star_neighbors)

    # aplica restauração
    df.loc[restore_mask, tie_col] = df.loc[restore_mask, tie_col_orig]
    return df

# ==============================
# Per-group resolver
# ==============================
def _resolve_group(
    g: pd.DataFrame,
    *,
    crd_col: str,
    z_col: str,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Optional[Mapping[str, int]],
    delta_z_threshold: float,
) -> pd.DataFrame:
    """Resolve ties within a single connected component."""
    crd = crd_col
    out = g[[crd]].copy()
    out["tie_result_new"] = 0

    star_mask = pd.Series(False, index=g.index)
    if "z_flag_homogenized" in g.columns:
        zf_series = _to_numeric(g["z_flag_homogenized"])
        star_mask = zf_series.eq(6)
        out.loc[star_mask.index[star_mask], "tie_result_new"] = 3

    cand = g[~star_mask].copy()
    if cand.empty:
        return out[[crd, "tie_result_new"]]

    survivors_pos = np.arange(len(cand), dtype=np.int64)

    for col in tiebreaking_priority:
        if survivors_pos.size <= 1:
            break

        sub = cand.iloc[survivors_pos]
        col_vals_1d = _series_1d_from(sub[col])

        if col == "instrument_type_homogenized":
            if instrument_type_priority is None:
                raise ValueError(
                    "instrument_type_priority is required when "
                    "'instrument_type_homogenized' is used in tiebreaking_priority."
                )
            scores = _score_instrument_type(col_vals_1d, instrument_type_priority)
        else:
            scores = _to_numeric(col_vals_1d)

        scores = scores.astype("float64")
        scores = scores.where(~scores.isna(), other=-np.inf)
        mx = scores.max()
        if np.isneginf(mx):
            continue

        keep_mask_np = scores.eq(mx).to_numpy()
        survivors_pos = survivors_pos[keep_mask_np]

    if survivors_pos.size > 1 and (delta_z_threshold or 0.0) > 0.0:
        sub = cand.iloc[survivors_pos]
        zvals = _to_numeric(_series_1d_from(sub[z_col])).dropna()
        if not zvals.empty and (zvals.max() - zvals.min()) <= float(delta_z_threshold):
            crds = sub[crd].astype(str).to_numpy()
            keep_local = np.argmin(crds)
            survivors_pos = np.array([survivors_pos[keep_local]], dtype=np.int64)

    if survivors_pos.size == 1:
        winner_label = cand.index[survivors_pos[0]]
        out.loc[winner_label, "tie_result_new"] = 1
    elif survivors_pos.size > 1:
        winner_labels = cand.index.take(survivors_pos)
        out.loc[winner_labels, "tie_result_new"] = 2

    return out[[crd, "tie_result_new"]]


# ==============================
# Public API (Pandas)
# ==============================
def deduplicate_pandas(
    df: pd.DataFrame,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Optional[Mapping[str, int]] = None,
    *,
    delta_z_threshold: float | int | None = 0.0,
    crd_col: str = "CRD_ID",
    compared_col: str = "compared_to",
    z_col: str = "z",
    tie_col: str = "tie_result",
    edge_log: bool = False,
    partition_tag: str | None = None,
    logger: logging.LoggerAdapter | None = None,
    group_col: str | None = None,   # <<< NOVO
) -> pd.DataFrame:
    """Graph-based deduplication with vectorized per-group resolution and Δz collapse."""
    required = {crd_col, compared_col, z_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()

    tie_col_orig = f"{tie_col}_orig"
    if tie_col in out.columns:
        try:
            out[tie_col_orig] = out[tie_col].astype("Int8")
        except Exception:
            out[tie_col_orig] = out[tie_col]

    crd_norm = out[crd_col].astype("string").str.strip()
    priority_set = set(tiebreaking_priority)

    zf_series: Optional[pd.Series] = None
    if "z_flag_homogenized" in out.columns:
        zf_series = _to_numeric(out["z_flag_homogenized"])

    # >>> pass edge_log flag down so diagnostics are computed only when requested
    nodes_edge, edges_uv, diag = _build_edges_fast(
        out,
        crd_col=crd_col,
        compared_col=compared_col,
        zf_series=zf_series,
        edge_log=edge_log,
    )

    # --- logging behavior (quiet for partitions) ---
    if edge_log:
        lg = logger or _phase_logger()
        tag = f"[{partition_tag}]" if partition_tag else "[global]"
        starB = diag.get("n_edges_starB_excluded")
        if isinstance(starB, int) and starB > 0:
            lg.warning(
                "%s Star neighbors excluded during edge build: star_rows_excl=%d, edges_raw=%d, starB_excl=%d, edges_kept=%d",
                tag,
                diag.get("n_rows_star_excluded", 0),
                diag.get("n_edges_raw", 0),
                starB,
                diag.get("n_edges_kept", 0),
            )
        elif partition_tag is None:
            lg.info(
                "%s Edge build summary: star_rows_excl=%d, edges_raw=%d, starB_excl=%s, edges_kept=%d",
                tag,
                diag.get("n_rows_star_excluded", 0),
                diag.get("n_edges_raw", 0),
                str(starB),
                diag.get("n_edges_kept", 0),
            )

    labels_edge = _connected_components_scipy(len(nodes_edge), edges_uv)

    if labels_edge.size:
        s_map_index = pd.Index(nodes_edge).astype("string").str.strip()
        s_map = pd.Series(labels_edge.astype("int64"), index=s_map_index)
        mapped = crd_norm.map(s_map)
    else:
        mapped = pd.Series(np.nan, index=out.index, dtype="float64")

    # Boolean NumPy mask of rows without a mapped group id from the fast path
    na_mask = mapped.isna().to_numpy()

    # Preallocate group ids array
    gids = np.full(len(out), -1, dtype=np.int64)

    # Fill groups for rows that already mapped via the fast path
    if (~na_mask).any():
        gids[~na_mask] = mapped[~na_mask].to_numpy(dtype=np.int64, copy=False)

    # Try to bridge NA rows via neighbor groups from the fast path label map
    if na_mask.any() and labels_edge.size:
        pos_na = np.flatnonzero(na_mask)
        cmp_str_all = out[compared_col].astype("string").str.strip()
        cmp_lists = cmp_str_all.iloc[pos_na].str.split(",")
        sub = pd.DataFrame({"pos": pos_na, "nbr": cmp_lists}).explode("nbr", ignore_index=False)
        sub["nbr"] = sub["nbr"].astype("string").str.strip()
        sub = sub[(sub["nbr"].notna()) & (sub["nbr"] != "")]
        if not sub.empty:
            sub["nbr_gid"] = sub["nbr"].map(s_map)
            sub = sub[sub["nbr_gid"].notna()]
            if not sub.empty:
                bridged = sub.groupby("pos", sort=False)["nbr_gid"].min().astype("int64")
                gids[bridged.index.to_numpy()] = bridged.to_numpy(dtype=np.int64, copy=False)
                na_mask[bridged.index.to_numpy()] = False

    # --- Fallback for rows still without group id (avoid mixing stars in graph) ---
    if na_mask.any():
        pos_na = np.flatnonzero(na_mask)
    
        # estrelas dentro dos NA
        if zf_series is not None:
            is_star_na = np.asarray(zf_series.iloc[pos_na].eq(6).fillna(False), dtype=bool)
        else:
            is_star_na = np.zeros(pos_na.size, dtype=bool)
    
        pos_na_nonstar = pos_na[~is_star_na]
    
        # Próximo ID livre, compatível com rótulos do fast-path
        next_gid = int(labels_edge.max()) + 1 if labels_edge.size else 0
    
        # ===== NA não-estrela =====
        if pos_na_nonstar.size:
            crd_arr_all = crd_norm.to_numpy()
            cmp_arr_all = out[compared_col].astype("string").str.strip().to_numpy()
    
            crd_arr = crd_arr_all[pos_na_nonstar]
            cmp_arr = cmp_arr_all[pos_na_nonstar]
    
            sub_df = pd.DataFrame({crd_col: crd_arr, compared_col: cmp_arr})
            edges_na = _build_edges_pdf(sub_df, crd_col=crd_col, compared_col=compared_col)
            nodes_na = list(pd.Index(sub_df[crd_col]).astype("string").unique())
    
            if nodes_na:
                if edges_na.empty:
                    # Sem arestas: cada linha vira singleton, sem colidir.
                    gids[pos_na_nonstar] = np.arange(next_gid, next_gid + len(pos_na_nonstar), dtype=np.int64)
                    next_gid += len(pos_na_nonstar)
                else:
                    # Com arestas: componentes de verdade
                    comp_map = _connected_components(nodes_na, edges_na)
                    gids[pos_na_nonstar] = next_gid + np.fromiter(
                        (comp_map.get(str(cid), -1) for cid in crd_arr),
                        dtype=np.int64,
                        count=len(crd_arr),
                    )
                    next_gid += (max(comp_map.values()) + 1 if comp_map else 0)
            else:
                # Sem nós: também singleton por linha
                gids[pos_na_nonstar] = np.arange(next_gid, next_gid + len(pos_na_nonstar), dtype=np.int64)
                next_gid += len(pos_na_nonstar)
    
            # >>> IMPORTANTE: marque como mapeado para não colidir depois
            na_mask[pos_na_nonstar] = False
    
        # ===== Estrelas (cada uma é singleton) =====
        if is_star_na.any():
            n_star = int(is_star_na.sum())
            gids[pos_na[is_star_na]] = np.arange(next_gid, next_gid + n_star, dtype=np.int64)
            na_mask[pos_na[is_star_na]] = False
            next_gid += n_star

    out["__group__"] = gids
    gid = out["__group__"]
    crd_s = crd_norm

    group_sizes = pd.Series(1, index=out.index).groupby(gid).transform("sum")
    is_singleton = group_sizes.eq(1)

    if zf_series is None:
        is_star = pd.Series(False, index=out.index)
    else:
        is_star = zf_series.eq(6)

    is_singleton_np = is_singleton.to_numpy(dtype=bool, na_value=False)
    is_star_np = is_star.to_numpy(dtype=bool, na_value=False)

    tr = np.zeros(len(out), dtype=np.int8)
    tr[is_star_np] = 3
    tr[is_singleton_np & ~is_star_np] = 1


    is_multi = ~is_singleton
    non_star = ~is_star
    survivors = (is_multi & non_star).copy()

    zf_num = _to_numeric(out.get("z_flag_homogenized", pd.Series(np.nan, index=out.index))).astype("float64")

    if "instrument_type_homogenized" in priority_set:
        if instrument_type_priority is None:
            raise ValueError(
                "instrument_type_priority is required when "
                "'instrument_type_homogenized' is used in tiebreaking_priority."
            )
        it_scores = _score_instrument_type(out["instrument_type_homogenized"], instrument_type_priority).astype("float64")
    else:
        it_scores = pd.Series(0.0, index=out.index, dtype="float64")

    for col in tiebreaking_priority:
        if not survivors.to_numpy(dtype=bool, na_value=False).any():
            break

        if col == "instrument_type_homogenized":
            scores = it_scores
        elif col == "z_flag_homogenized":
            scores = zf_num
        else:
            scores = _to_numeric(out[col]).astype("float64")

        scores = scores.where(~scores.isna(), other=-np.inf)
        s_eff = scores.where(survivors, other=-np.inf)
        gmax = s_eff.groupby(gid).transform("max")
        survivors &= s_eff.eq(gmax)

    thr = float(delta_z_threshold or 0.0)
    if survivors.to_numpy(dtype=bool, na_value=False).any() and thr > 0.0:
        zvals = _to_numeric(out[z_col]).astype("float64")
        survivors = _collapse_within_dz(survivors, gid, zvals, crd_s, thr)

    n_surv_final = survivors.groupby(gid).transform("sum")
    one_winner = (survivors & n_surv_final.eq(1))
    multi_winner = (survivors & n_surv_final.ge(2))

    tr[one_winner.to_numpy(dtype=bool, na_value=False)] = 1
    tr[multi_winner.to_numpy(dtype=bool, na_value=False)] = 2

    out[tie_col] = pd.Series(tr, index=out.index).astype("Int8")

    if zf_series is not None:
        tr_num = pd.to_numeric(out[tie_col], errors="coerce")
        eq3_np = tr_num.eq(3.0).to_numpy(dtype=bool, na_value=False)
        is_star_np = is_star.to_numpy(dtype=bool, na_value=False)
        is_single_np = is_singleton.to_numpy(dtype=bool, na_value=False)

        invalid_3_np = eq3_np & ~is_star_np
        if invalid_3_np.any():
            invalid_3 = pd.Series(invalid_3_np, index=out.index)
            single = pd.Series(is_single_np, index=out.index)
            out.loc[invalid_3 & single, tie_col] = np.int8(1)
            out.loc[invalid_3 & ~single, tie_col] = np.int8(0)

    z_num = _to_numeric(out[z_col]).astype("float64")
    f_num = zf_num.fillna(-np.inf)
    t_num = (
        _score_instrument_type(out["instrument_type_homogenized"], instrument_type_priority).astype("float64")
        if ("instrument_type_homogenized" in priority_set and instrument_type_priority is not None)
        else None
    )
    thr_local = float(delta_z_threshold or 0.0)

    is_pair = group_sizes.eq(2)
    pair_one = out[tie_col].eq(1) & is_pair
    both_one = pair_one.groupby(gid).transform("sum").eq(2)
    bad_gids = pd.Index(gid[pair_one & both_one]).unique()

    if len(bad_gids) > 0:
        for gval in bad_gids:
            mask = (gid.eq(gval) & pair_one)
            pos = np.flatnonzero(mask.to_numpy(dtype=bool, na_value=False))
            if pos.size != 2:
                continue
            p1, p2 = int(pos[0]), int(pos[1])

            f1, f2 = float(f_num.iloc[p1]), float(f_num.iloc[p2])
            if f1 > f2:
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([1, 0], dtype=np.int8)
                continue
            if f2 > f1:
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([0, 1], dtype=np.int8)
                continue

            if t_num is not None:
                t1, t2 = float(t_num.iloc[p1]), float(t_num.iloc[p2])
                if t1 > t2:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([1, 0], dtype=np.int8)
                    continue
                if t2 > t1:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([0, 1], dtype=np.int8)
                    continue

            z1, z2 = z_num.iloc[p1], z_num.iloc[p2]
            if thr_local <= 0.0 or pd.isna(z1) or pd.isna(z2):
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([2, 2], dtype=np.int8)
                continue

            dz = abs(float(z1) - float(z2))
            if dz > thr_local:
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([2, 2], dtype=np.int8)
            else:
                c1, c2 = str(crd_s.iloc[p1]), str(crd_s.iloc[p2])
                if c1 <= c2:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([1, 0], dtype=np.int8)
                else:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([0, 1], dtype=np.int8)

    pair_zero = out[tie_col].eq(0) & is_pair
    both_zero = pair_zero.groupby(gid).transform("sum").eq(2)
    bad_gids0 = pd.Index(gid[pair_zero & both_zero]).unique()

    if len(bad_gids0) > 0:
        for gval in bad_gids0:
            mask = (gid.eq(gval) & pair_zero)
            pos = np.flatnonzero(mask.to_numpy(dtype=bool, na_value=False))
            if pos.size != 2:
                continue
            p1, p2 = int(pos[0]), int(pos[1])

            f1, f2 = float(f_num.iloc[p1]), float(f_num.iloc[p2])
            if f1 > f2:
                out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(1)
                out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(0)
                continue
            if f2 > f1:
                out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(0)
                out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(1)
                continue

            if t_num is not None:
                t1, t2 = float(t_num.iloc[p1]), float(t_num.iloc[p2])
                if t1 > t2:
                    out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(1)
                    out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(0)
                    continue
                if t2 > t1:
                    out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(0)
                    out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(1)
                    continue

            z1, z2 = z_num.iloc[p1], z_num.iloc[p2]
            if (thr_local > 0.0) and not (pd.isna(z1) or pd.isna(z2)):
                dz = abs(float(z1) - float(z2))
                if dz <= thr_local:
                    c1, c2 = str(crd_s.iloc[p1]), str(crd_s.iloc[p2])
                    if c1 <= c2:
                        out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(1)
                        out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(0)
                    else:
                        out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(0)
                        out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(1)
                else:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([2, 2], dtype=np.int8)
            else:
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([2, 2], dtype=np.int8)

    out = _apply_guard_restore_local(
        out,
        crd_col=crd_col,
        compared_col=compared_col,
        zf_series=zf_series,
        tie_col=tie_col,
        tie_col_orig=tie_col_orig,
    )

    drop_cols = []
    if tie_col_orig in out.columns:
        drop_cols.append(tie_col_orig)
    
    if group_col:
        out[group_col] = pd.Series(out["__group__"], index=out.index).astype("Int64")
    else:
        drop_cols.append("__group__")
    
    out.drop(columns=drop_cols, inplace=True, errors="ignore")
    
    # >>> PATCH: namespace por partição
    if group_col and partition_tag:
        try:
            import zlib
            base = np.int64(zlib.crc32(str(partition_tag).encode("utf-8"))) << np.int64(32)
            m = out[group_col].notna()
            out.loc[m, group_col] = (base + out.loc[m, group_col].astype("int64")).astype("Int64")
        except Exception:
            pass
    
    if zf_series is not None:
        out.loc[zf_series.eq(6).fillna(False), tie_col] = np.int8(3)
    
    return out



# ==============================
# LSDB per-partition dedup (with or without margin)
# ==============================

def _ensure_string_pyarrow(s: pd.Series) -> pd.Series:
    """Cast to Arrow-backed string if available, else pandas string."""
    try:
        return s.astype("string[pyarrow]", copy=False)
    except Exception:
        return s.astype("string", copy=False)


def _nullable_int8(s: pd.Series) -> pd.Series:
    """Cast to pandas nullable Int8."""
    return s.astype("Int8", copy=False)


def _to_pandas(df):
    """Return a pandas.DataFrame for NestedFrame/unknown inputs."""
    if df is None:
        return pd.DataFrame()
    if hasattr(df, "to_pandas"):
        try:
            return df.to_pandas()
        except Exception:
            pass
    return pd.DataFrame(df)


def _shrink_to_needed(df, needed, crd_col, compared_col, z_col):
    """Project to required columns and normalize essential dtypes."""
    if df is None or len(df) == 0:
        cols = list(needed)
        out = pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
        for c in (crd_col, compared_col):
            out[c] = out[c].astype("string")
        return out[list(needed)]

    keep = [c for c in df.columns if c in needed]
    out = df[keep].copy()

    if crd_col not in out:
        out[crd_col] = pd.Series(dtype="string")
    if compared_col not in out:
        out[compared_col] = pd.Series(dtype="string")
    if z_col not in out:
        out[z_col] = pd.Series(dtype="float64")

    out[crd_col] = _ensure_string_pyarrow(out[crd_col])
    out[compared_col] = _ensure_string_pyarrow(out[compared_col])
    out[z_col] = pd.to_numeric(out[z_col], errors="coerce")

    for c in needed:
        if c not in out:
            out[c] = pd.Series(dtype="float64")

    return out[[c for c in needed]]


def _dedup_local_with_margin(
    part_main,
    part_margin,
    pixel,  # diagnostics only (unused for now)
    *,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Optional[Mapping[str, int]],
    delta_z_threshold: float = 0.0,
    crd_col: str = "CRD_ID",
    compared_col: str = "compared_to",
    z_col: str = "z",
    tie_col: str = "tie_result",
    edge_log: bool = False,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Run dedup on (main + margin) and return labels for main rows only.

    Args:
      part_main: Main partition (NestedFrame/pandas-like).
      part_margin: Margin partition aligned to main.
      pixel: Partition diagnostics (unused).
      tiebreaking_priority: Ordered columns for tie-breaking.
      instrument_type_priority: Map for instrument type scoring (optional).
      delta_z_threshold: Δz threshold for final disambiguation.
      crd_col: Name of the ID column.
      compared_col: Name of the neighbors column.
      z_col: Name of the redshift column.
      tie_col: Name of the output tie-result column.
      edge_log: Enable edge diagnostics during graph build.
      group_col: If set, also emit component/group id in this column.

    Returns:
      pandas.DataFrame with [CRD_ID, tie_result, (optional) group_col] for main rows.
    """
    # Normalize inputs to pandas.
    pm = _to_pandas(part_main)
    mg = _to_pandas(part_margin)

    # Project to required columns.
    needed = {crd_col, compared_col, z_col, tie_col} | set(tiebreaking_priority or [])
    if instrument_type_priority is not None:
        needed.add("instrument_type_homogenized")
    pm = _shrink_to_needed(pm, needed, crd_col, compared_col, z_col)
    mg = _shrink_to_needed(mg, needed, crd_col, compared_col, z_col)

    # Early exit on empty view.
    if pm.empty and mg.empty:
        cols = {
            crd_col: pd.Series(dtype="string[pyarrow]"),
            tie_col: pd.Series(dtype="Int8"),
        }
        if group_col:
            cols[group_col] = pd.Series(dtype="Int64")
        return pd.DataFrame(cols)

    # Tag partition (cheap fingerprint for logs).
    try:
        i_min, i_max = pm.index.min(), pm.index.max()
        partition_tag = f"idx=[{i_min}..{i_max}]"
    except Exception:
        partition_tag = "idx=[unknown]"

    # Build view and run solver (guard-restore happens inside).
    pm["_src"] = "main"
    mg["_src"] = "margin"
    view = pd.concat([pm, mg], ignore_index=True)

    solved = deduplicate_pandas(
        view,
        tiebreaking_priority=tiebreaking_priority,
        instrument_type_priority=instrument_type_priority,
        delta_z_threshold=float(delta_z_threshold),
        crd_col=crd_col,
        compared_col=compared_col,
        z_col=z_col,
        tie_col=tie_col,
        edge_log=edge_log,
        partition_tag=partition_tag,
        logger=_phase_logger(),
        group_col=group_col,
    )

    # Keep only main rows and required columns.
    cols = [crd_col, tie_col]
    if group_col and (group_col in solved.columns):
        cols.append(group_col)

    out = solved.loc[solved["_src"] == "main", cols].copy()

    # Stable dtypes.
    out[crd_col] = _ensure_string_pyarrow(out[crd_col])
    out[tie_col] = _nullable_int8(out[tie_col])
    if group_col and (group_col in out.columns):
        out[group_col] = out[group_col].astype("Int64")

    return out


def _dedup_local_no_margin(
    part_main,
    *,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Optional[Mapping[str, int]],
    delta_z_threshold: float = 0.0,
    crd_col: str = "CRD_ID",
    compared_col: str = "compared_to",
    z_col: str = "z",
    tie_col: str = "tie_result",
    edge_log: bool = False,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Run dedup using only the main partition.

    Args:
      part_main: Main partition (NestedFrame/pandas-like).
      tiebreaking_priority: Ordered columns for tie-breaking.
      instrument_type_priority: Map for instrument type scoring (optional).
      delta_z_threshold: Δz threshold for final disambiguation.
      crd_col: Name of the ID column.
      compared_col: Name of the neighbors column.
      z_col: Name of the redshift column.
      tie_col: Name of the output tie-result column.
      edge_log: Enable edge diagnostics during graph build.
      group_col: If set, also emit component/group id in this column.

    Returns:
      pandas.DataFrame with [CRD_ID, tie_result, (optional) group_col].
    """
    # Normalize inputs to pandas.
    pm = _to_pandas(part_main)

    # Project to required columns.
    needed = {crd_col, compared_col, z_col, tie_col} | set(tiebreaking_priority or [])
    if instrument_type_priority is not None:
        needed.add("instrument_type_homogenized")
    pm = _shrink_to_needed(pm, needed, crd_col, compared_col, z_col)

    # Early exit on empty view.
    if pm.empty:
        cols = {
            crd_col: pd.Series(dtype="string[pyarrow]"),
            tie_col: pd.Series(dtype="Int8"),
        }
        if group_col:
            cols[group_col] = pd.Series(dtype="Int64")
        return pd.DataFrame(cols)

    # Tag partition (cheap fingerprint for logs).
    try:
        i_min, i_max = pm.index.min(), pm.index.max()
        partition_tag = f"idx=[{i_min}..{i_max}]"
    except Exception:
        partition_tag = "idx=[unknown]"

    # Run solver (guard-restore happens inside).
    solved = deduplicate_pandas(
        pm,
        tiebreaking_priority=tiebreaking_priority,
        instrument_type_priority=instrument_type_priority,
        delta_z_threshold=float(delta_z_threshold),
        crd_col=crd_col,
        compared_col=compared_col,
        z_col=z_col,
        tie_col=tie_col,
        edge_log=edge_log,
        partition_tag=partition_tag,
        logger=_phase_logger(),
        group_col=group_col,
    )

    # Select output columns.
    cols = [crd_col, tie_col]
    if group_col and (group_col in solved.columns):
        cols.append(group_col)

    out = solved[cols].copy()

    # Stable dtypes.
    out[crd_col] = _ensure_string_pyarrow(out[crd_col])
    out[tie_col] = _nullable_int8(out[tie_col])
    if group_col and (group_col in out.columns):
        out[group_col] = out[group_col].astype("Int64")

    return out


def _columns_set(ddf) -> set:
    """Return the set of column names from a Dask DataFrame's _meta (cheap)."""
    return set(map(str, getattr(getattr(ddf, "_meta", None), "columns", [])))

def _assert_required(ddf, required: set[str], where: str):
    """Raise KeyError if required columns are missing from ddf."""
    have = _columns_set(ddf)
    missing = sorted([c for c in required if c not in have])
    if missing:
        raise KeyError(f"Missing required columns in {where}: {missing}")

def _assert_priorities(ddf, priorities: list[str], where: str):
    """Raise KeyError if any tiebreaking_priority column is missing from ddf."""
    have = _columns_set(ddf)
    missing = sorted([c for c in priorities if c not in have])
    if missing:
        raise KeyError(f"Missing priority columns in {where}: {missing}")

def run_dedup_with_lsdb_map_partitions(
    cat,
    *,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Optional[Mapping[str, int]],
    delta_z_threshold: float = 0.0,
    crd_col: str = "CRD_ID",
    compared_col: str = "compared_to",
    z_col: str = "z",
    tie_col: str = "tie_result",
    edge_log: bool = False,
    group_col: str | None = None,   # <<< NOVO
) -> dd.DataFrame:
    """Compute dedup labels per partition via LSDB; align divisions if margin exists."""
    logger = _phase_logger()
    with log_phase("deduplication", "run_dedup_with_lsdb_map_partitions", _base_logger()) as log:
        # --- Early validations ---
        if not hasattr(cat, "_ddf"):
            raise AttributeError("Catalog does not expose _ddf.")
        if not isinstance(tiebreaking_priority, (list, tuple)) or len(tiebreaking_priority) == 0:
            raise TypeError("tiebreaking_priority must be a non-empty sequence.")
        if float(delta_z_threshold) < 0.0:
            raise ValueError("delta_z_threshold must be non-negative.")

        main_ddf = cat._ddf
        has_margin = bool(getattr(cat, "margin", None) and hasattr(cat.margin, "_ddf"))
        log.info("Inputs: has_margin=%s, npartitions(main)=%s", has_margin, main_ddf.npartitions)

        # Strict schema checks: required base + all tiebreaking priority columns
        required_base = {crd_col, compared_col, z_col}
        _assert_required(main_ddf, required_base, "main")
        _assert_priorities(main_ddf, list(tiebreaking_priority), "main")
        
        if has_margin:
            margin_ddf = cat.margin._ddf
            _assert_required(margin_ddf, required_base, "margin")
            _assert_priorities(margin_ddf, list(tiebreaking_priority), "margin")
        
        # Enforce instrument_type mapping only if requested in priorities
        if ("instrument_type_homogenized" in set(tiebreaking_priority)) and (instrument_type_priority is None):
            raise ValueError(
                "instrument_type_priority is required when "
                "'instrument_type_homogenized' is used in tiebreaking_priority."
            )

        # Output meta for map_partitions
        meta_dict = {
            crd_col: pd.Series(dtype="string[pyarrow]"),
            tie_col: pd.Series(dtype="Int8"),
        }
        if group_col:
            meta_dict[group_col] = pd.Series(dtype="Int64")
        meta = pd.DataFrame(meta_dict)

        if not has_margin:
            log.info("No margin attached; running local dedup without margin.")
            labels_dd = main_ddf.map_partitions(
                _dedup_local_no_margin,
                meta=meta,
                tiebreaking_priority=tiebreaking_priority,
                instrument_type_priority=instrument_type_priority,
                delta_z_threshold=float(delta_z_threshold),
                crd_col=crd_col,
                compared_col=compared_col,
                z_col=z_col,
                tie_col=tie_col,
                edge_log=edge_log,
                group_col=group_col
            )
        else:
            margin_ddf = cat.margin._ddf

            # Main must have known divisions.
            main_div = getattr(main_ddf, "divisions", None)
            if main_div is None:
                raise RuntimeError(
                    "Margin present but main catalog has unknown divisions; rebuild with known divisions."
                )

            # Margin must also have known divisions; if not, set a sorted index equal to main's index.
            idx_name = main_ddf._meta.index.name
            if idx_name is None:
                raise RuntimeError("Main catalog index name is None; cannot align divisions.")
            if getattr(margin_ddf, "divisions", None) is None:
                with log_phase("deduplication", "margin_set_index", _base_logger()):
                    margin_ddf = margin_ddf.set_index(idx_name, sorted=True, drop=False)

            marg_div = margin_ddf.divisions
            if marg_div is None:
                raise RuntimeError("Margin still has unknown divisions after set_index.")

            # Make division dtypes compatible if needed.
            t_main = type(main_div[0]) if len(main_div) else None
            t_marg = type(marg_div[0]) if len(marg_div) else None
            if t_main is not None and t_marg is not None and t_main != t_marg:
                try:
                    import numpy as _np
                    main_div = tuple(_np.asarray(main_div, dtype=object).tolist())
                    marg_div = tuple(_np.asarray(marg_div, dtype=object).tolist())
                except Exception as e:
                    raise RuntimeError(
                        f"Incompatible division dtypes between main ({t_main}) and margin ({t_marg}): {e}"
                    ) from e

            # Build ordered union of divisions so both sides share the same split points.
            try:
                merged_divisions = tuple(sorted(set(main_div + marg_div)))
                if len(merged_divisions) < 2:
                    raise RuntimeError("Merged divisions must have at least two boundaries.")
            except Exception as e:
                raise RuntimeError(f"Could not build merged divisions: {e}") from e

            # Repartition both sides to the merged divisions.
            from dask import config as _dask_config
            with log_phase("deduplication", "repartition_to_merged_divisions", _base_logger()) as l2:
                with _dask_config.set({"dataframe.shuffle.method": "tasks"}):
                    main_ddf_aligned = main_ddf.repartition(divisions=merged_divisions, force=True)
                    margin_ddf_aligned = margin_ddf.repartition(divisions=merged_divisions, force=True)
                l2.info(
                    "Aligned: nparts(main)=%d nparts(margin)=%d",
                    main_ddf_aligned.npartitions,
                    margin_ddf_aligned.npartitions,
                )

            # Strict post-conditions
            if (
                main_ddf_aligned.npartitions != margin_ddf_aligned.npartitions
                or main_ddf_aligned.divisions != margin_ddf_aligned.divisions
            ):
                raise RuntimeError(
                    "Aligned repartition failed: main and margin differ in npartitions/divisions.\n"
                    f" main.npartitions={main_ddf_aligned.npartitions}, "
                    f"margin.npartitions={margin_ddf_aligned.npartitions}\n"
                    f" main.divisions[:3]={main_ddf_aligned.divisions[:3]} ... [-3:]={main_ddf_aligned.divisions[-3:]}\n"
                    f" margin.divisions[:3]={margin_ddf_aligned.divisions[:3]} ... [-3:]={margin_ddf_aligned.divisions[-3:]}"
                )

            # 1:1 zip between aligned partitions.
            with log_phase("deduplication", "map_partitions_with_margin", _base_logger()):
                labels_dd = main_ddf_aligned.map_partitions(
                    _dedup_local_with_margin,
                    margin_ddf_aligned,
                    None,  # diagnostics
                    meta=meta,
                    tiebreaking_priority=tiebreaking_priority,
                    instrument_type_priority=instrument_type_priority,
                    delta_z_threshold=float(delta_z_threshold),
                    crd_col=crd_col,
                    compared_col=compared_col,
                    z_col=z_col,
                    tie_col=tie_col,
                    edge_log=edge_log,
                    group_col=group_col
                )

        # Stable dtypes for downstream merges.
        assign_map = {
            crd_col: labels_dd[crd_col].astype("string[pyarrow]"),
            tie_col: labels_dd[tie_col].astype("Int8"),
        }
        if group_col:
            assign_map[group_col] = labels_dd[group_col].astype("Int64")
        labels_dd = labels_dd.assign(**assign_map)

        log.info("Labels dtypes stabilized.")
        return labels_dd
