# deduplication.py
from __future__ import annotations
"""
Deduplication module for Combine Redshift Catalogs (CRC).

This module groups rows into connected components using the undirected graph
implied by `CRD_ID <-> compared_to`, and resolves duplicates within each
component according to a configurable tiebreaking priority.

Notes
-----
- The column `compared_to` must be a comma-separated list of CRD_IDs (strings),
  or NA/empty when there are no neighbors.
- CRD_IDs are assumed to be globally unique within the input frame.
- Rows with `z_flag_homogenized == 6` are considered *stars*: they receive
  `tie_result = 3` and do not participate in tiebreaking.
"""

from typing import Iterable, Mapping, Sequence, Optional, Dict, List, Tuple
import math
import numpy as np
import pandas as pd

import re

def _canon_id_series(s: pd.Series) -> pd.Series:
    """Normalize CRD-like IDs: strip + remove zero-width chars/BOM."""
    t = s.astype("string")
    # remove ZWSP/ZWJ/ZWNJ/BOM
    t = t.str.replace(r"[\u200B-\u200D\uFEFF]", "", regex=True)
    return t.str.strip()
# =============================================================================
# Small helpers
# =============================================================================

def _norm_str(x) -> str | None:
    """Normalize value to stripped string; returns None for NA/empty."""
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
    # split by comma, strip tokens, drop empties
    return [t for t in (tok.strip() for tok in s.split(",")) if t]


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to float; non-castable values become NaN."""
    return pd.to_numeric(series, errors="coerce")


def _score_instrument_type(series: pd.Series, priority_map: Mapping[str, int]) -> pd.Series:
    """Map 'instrument_type_homogenized' to numeric priority (unknown -> 0)."""
    # normalize user map once
    norm_map = {str(k).strip().lower(): int(v) for k, v in priority_map.items()}

    def _score_one(v):
        v = _norm_str(v)
        return norm_map.get(v, 0) if v is not None else 0

    return series.map(_score_one).astype("int64")


# =============================================================================
# Graph building
# =============================================================================
# >>> NEW: super-rápido e vetorizado
def _split_cmp_vectorized(s: pd.Series) -> pd.Series:
    """
    Vectorized split/trim for compared_to.
    - Input: string dtype Series (can contain <NA>)
    - Output: Series of lists (may contain None/[])
    """
    s = s.astype("string")  # handles <NA> gracefully
    # split by comma
    lst = s.str.split(",")
    # explode + trim depois (mais rápido que map/apply no Python)
    return lst

def _build_edges_fast(
    df: pd.DataFrame,
    *,
    crd_col: str,
    compared_col: str,
    zf_series: Optional[pd.Series] = None,
):
    """
    Build undirected edges using only NON-STAR rows as sources,
    keeping neighbors only if they exist AND are NON-STAR too.
    """
    non_star_mask = pd.Series(True, index=df.index)
    if zf_series is not None:
        non_star_mask &= ~pd.to_numeric(zf_series, errors="coerce").eq(6)

    # reset index to ensure positional alignment
    A_df = df.loc[non_star_mask, [crd_col, compared_col]].copy().reset_index(drop=True)
    if A_df.empty:
        return pd.Index([], dtype="object"), np.empty((0, 2), dtype=np.int32)

    present_nonstar = set(
        _canon_id_series(df.loc[non_star_mask, crd_col]).dropna().unique()
    )

    cmp_lists = _split_cmp_vectorized(A_df[compared_col])
    edges_raw = (
        A_df[[crd_col]]
        .assign(v=cmp_lists)
        .rename(columns={crd_col: "u"})
        .explode("v", ignore_index=True)
    )

    # canonicalize both endpoints consistently
    edges_raw["u"] = _canon_id_series(edges_raw["u"])
    edges_raw["v"] = _canon_id_series(edges_raw["v"])

    # drop empties / NAs
    edges_raw = edges_raw.dropna(subset=["u", "v"])
    edges_raw = edges_raw[(edges_raw["u"] != "") & (edges_raw["v"] != "")]

    # keep only neighbors that exist and are non-star
    edges_raw = edges_raw[edges_raw["v"].isin(present_nonstar)]
    if edges_raw.empty:
        return pd.Index([], dtype="object"), np.empty((0, 2), dtype=np.int32)

    nodes_edge = pd.Index(
        pd.unique(pd.concat([edges_raw["u"], edges_raw["v"]], ignore_index=True))
    )

    id2ix = {cid: i for i, cid in enumerate(nodes_edge)}
    u = edges_raw["u"].map(id2ix).to_numpy(dtype=np.int32, copy=False)
    v = edges_raw["v"].map(id2ix).to_numpy(dtype=np.int32, copy=False)

    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    mask = lo != hi
    lo, hi = lo[mask], hi[mask]
    if lo.size == 0:
        return nodes_edge, np.empty((0, 2), dtype=np.int32)

    uv = np.stack([lo, hi], axis=1)
    view = uv.view([('x', uv.dtype), ('y', uv.dtype)])
    uv = np.unique(view).view(uv.dtype).reshape(-1, 2)
    return nodes_edge, uv



# >>> NEW: connected components com SciPy (muito rápido)
def _connected_components_scipy(n_nodes: int, edges_uv: np.ndarray) -> np.ndarray:
    """
    Return component labels for nodes [0..n_nodes-1], using SciPy.
    Singletons (sem aresta) aparecem como componentes distintos.
    """
    if n_nodes == 0:
        return np.array([], dtype=np.int64)
    if edges_uv.size == 0:
        # cada nó isolado
        return np.arange(n_nodes, dtype=np.int64)

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    data = np.ones(edges_uv.shape[0], dtype=np.int8)
    A = coo_matrix((data, (edges_uv[:, 0], edges_uv[:, 1])), shape=(n_nodes, n_nodes))
    # simetriza
    A = A + A.T
    _, labels = connected_components(A, directed=False, return_labels=True)
    return labels


def _build_edges_pdf(
    df: pd.DataFrame,
    crd_col: str,
    compared_col: str,
) -> pd.DataFrame:
    """Build undirected edges (u, v) from `compared_to`.

    Canonicalization is applied so that `u < v` lexicographically and self-edges
    are ignored.

    Args:
      df: Input dataframe containing at least `crd_col` and `compared_col`.
      crd_col: Name of the unique identifier column (CRD_ID).
      compared_col: Name of the column with comma-separated neighbor IDs.

    Returns:
      A pandas DataFrame with columns ["u", "v"] (string dtype), de-duplicated.
      Nodes not present in `df[crd_col]` are discarded from edges.
    """
    if df.empty:
        return pd.DataFrame({"u": pd.Series([], dtype="string"),
                             "v": pd.Series([], dtype="string")})

    # Present IDs (only edges pointing to present IDs are kept)
    present_ids = set(df[crd_col].astype(str))

    # Split/explode neighbors
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

    # Keep only neighbors that actually exist in the frame
    edges["v"] = edges["v"].astype(str)
    edges = edges[edges["v"].isin(present_ids)]

    # Canonicalize undirected pairs (u < v) and drop self-edges
    u = edges["u"].astype(str)
    v = edges["v"].astype(str)
    lo = np.minimum(u.values, v.values)
    hi = np.maximum(u.values, v.values)
    mask = lo != hi
    if not mask.any():
        return pd.DataFrame({"u": pd.Series([], dtype="string"),
                             "v": pd.Series([], dtype="string")})

    out = pd.DataFrame(
        {"u": pd.Series(lo[mask], dtype="string"),
         "v": pd.Series(hi[mask], dtype="string")}
    ).drop_duplicates(ignore_index=True)

    return out

def _collapse_within_dz(mask: pd.Series,
                        gid: pd.Series,
                        zvals: pd.Series,
                        crd_s: pd.Series,
                        threshold: float) -> pd.Series:
    """
    For each group (gid), sort by (gid, z, crd) and start a new cluster when the
    consecutive jump in z exceeds `threshold`. Keep 1 per cluster (min CRD).
    Rows with NaN z DO NOT participate in the clustering and remain as they are
    in `mask`. Uses positional indexing only.
    """
    thr = float(threshold or 0.0)
    if thr <= 0.0:
        return mask

    m = mask.to_numpy(dtype=bool, na_value=False)
    if not m.any():
        return mask

    n = len(mask)
    pos_all = np.arange(n, dtype=np.int64)
    pos = pos_all[m]  # candidate positions

    gid_arr = pd.Index(gid).to_numpy() if isinstance(gid, pd.Series) else np.asarray(gid)
    z_arr   = pd.to_numeric(zvals, errors="coerce").to_numpy()
    crd_arr = crd_s.astype(str).to_numpy()

    # Only candidates with defined z take part in clustering
    pos_def = pos[~np.isnan(z_arr[pos])]
    if pos_def.size == 0:
        # Nobody can be clustered; leave survivors as-is
        return mask

    # Sort defined-z candidates by (gid, z, crd)
    order = np.lexsort((crd_arr[pos_def], z_arr[pos_def], gid_arr[pos_def]))
    pos_sorted = pos_def[order]
    g_sorted   = gid_arr[pos_sorted]
    z_sorted   = z_arr[pos_sorted]
    crd_sorted = crd_arr[pos_sorted]

    # Cluster break: new gid OR z-jump > thr
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

    # Keep lexicographically smallest CRD within each (gid, clus)
    sub = pd.DataFrame({
        "pos": pos_sorted,
        "gid": g_sorted,
        "clus": clus,
        "crd": crd_sorted,
    })
    min_crd = sub.groupby(["gid", "clus"], sort=False)["crd"].transform("min")
    keep_mask = (sub["crd"] == min_crd)
    winners_pos = sub.loc[keep_mask].groupby(["gid", "clus"], sort=False).head(1)["pos"].to_numpy()

    # Rebuild final mask:
    # - turn off ONLY defined-z candidates (pos_def)
    # - turn on winners among them
    # - NaN-z candidates remain unchanged from the input `mask`
    out_np = m.copy()
    out_np[pos_def] = False
    out_np[winners_pos] = True
    return pd.Series(out_np, index=mask.index)


# =============================================================================
# Connected components (Union-Find / DSU)
# =============================================================================

class _DSU:
    """A lightweight Disjoint Set Union structure for connected components."""
    __slots__ = ("p",)

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Attach higher index under lower for slight determinism
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


def _connected_components(
    nodes: Iterable[str],
    edges: pd.DataFrame,
) -> Dict[str, int]:
    """Compute connected components over `nodes` using undirected `edges`.

    Args:
      nodes: Iterable of node IDs (strings) to cover (including singletons).
      edges: DataFrame with columns ["u", "v"] (strings), undirected pairs.

    Returns:
      Mapping from node ID to integer group ID in [0, n_groups).
    """
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

    # Compress, then relabel roots into dense [0..k)
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


# =============================================================================
# Per-group tiebreak resolution
# =============================================================================

# --- NEW helper: always return a 1-D Series, even if a duplicated-name selection produced a 2-D DataFrame ---
def _series_1d_from(df_or_series) -> pd.Series:
    """
    Ensure a 1-D Series. If pandas returned a DataFrame due to duplicated column
    names (e.g., selecting 'z_flag_homogenized' when it exists twice), take the
    FIRST column deterministically. This prevents 'phantom values' leaking in
    from secondary duplicates.
    """
    if isinstance(df_or_series, pd.DataFrame):
        return df_or_series.iloc[:, 0]
    return df_or_series

# --- PATCHED: accept Series OR single-col DataFrame; coerce to numeric safely ---
def _to_numeric(series_like) -> pd.Series:
    """
    Coerce to numeric (float). Accepts a Series or a single-column DataFrame.
    Non-castable values -> NaN. Critically, we normalize to 1-D first to avoid
    accidental bfill across duplicated columns.
    """
    s = _series_1d_from(series_like)
    return pd.to_numeric(s, errors="coerce")


def _resolve_group(
    g: pd.DataFrame,
    *,
    crd_col: str,
    z_col: str,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Optional[Mapping[str, int]],
    delta_z_threshold: float,
) -> pd.DataFrame:
    """Resolve ties within a single connected component.

    Rules:
      * Rows with `z_flag_homogenized == 6` are stars: set `tie_result_new = 3`
        and REMOVE them from tiebreaking.
      * For each column in `tiebreaking_priority` (in order), keep only rows
        that achieve the group's maximum. Missing values lose (as -inf).
        The only supported non-numeric column is `instrument_type_homogenized`,
        which is translated via `instrument_type_priority`.
      * If multiple candidates remain and `(max(z) - min(z)) <= delta_z_threshold`
        (considering defined z only), keep exactly one: the lexicographically
        smallest CRD_ID. Others are eliminated.
      * Final labels:
          - One survivor ⇒ 1
          - Multiple survivors ⇒ 2
          - Non-survivors ⇒ 0
          - Star ⇒ 3
    """
    crd = crd_col
    out = g[[crd]].copy()
    out["tie_result_new"] = 0  # default losers

    # ---------------------------------------------------------------------
    # Stars (strict, 1-D only) — exclude from tiebreaking but mark as 3
    # ---------------------------------------------------------------------
    star_mask = pd.Series(False, index=g.index)
    if "z_flag_homogenized" in g.columns:
        # `_to_numeric` here must be the patched 1-D-safe version
        zf_series = _to_numeric(g["z_flag_homogenized"])
        star_mask = zf_series.eq(6)
        out.loc[star_mask.index[star_mask], "tie_result_new"] = 3

    # Candidates for tiebreak (non-stars)
    cand = g[~star_mask].copy()
    if cand.empty:
        return out[[crd, "tie_result_new"]]

    # WORK ENTIRELY WITH POSITIONAL INDICES (avoids duplicate-label headaches)
    survivors_pos = np.arange(len(cand), dtype=np.int64)

    # ---------------------------------------------------------------------
    # Apply tiebreaking criteria in order; higher is better; missing loses
    # ---------------------------------------------------------------------
    for col in tiebreaking_priority:
        if survivors_pos.size <= 1:
            break

        # Positional slice for current survivors
        sub = cand.iloc[survivors_pos]

        # Defensive: force 1-D even if 'col' appears duplicated
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
        scores = scores.where(~scores.isna(), other=-np.inf)  # missing loses
        mx = scores.max()
        if np.isneginf(mx):
            # All missing -> nothing to prune at this stage
            continue

        # Boolean mask in the SAME positional order as `survivors_pos`
        keep_mask_np = scores.eq(mx).to_numpy()

        # Positional filtering (robust even with duplicate labels)
        survivors_pos = survivors_pos[keep_mask_np]

    # ---------------------------------------------------------------------
    # Optional Δz collapse among remaining survivors (near-identical z)
    # ---------------------------------------------------------------------
    if survivors_pos.size > 1 and (delta_z_threshold or 0.0) > 0.0:
        sub = cand.iloc[survivors_pos]
        zvals = _to_numeric(_series_1d_from(sub[z_col])).dropna()
        if not zvals.empty and (zvals.max() - zvals.min()) <= float(delta_z_threshold):
            # Keep exactly one: the lexicographically smallest CRD_ID
            crds = sub[crd].astype(str).to_numpy()
            # argmin over strings is lexicographic; gives local position in `sub`
            keep_local = np.argmin(crds)
            survivors_pos = np.array([survivors_pos[keep_local]], dtype=np.int64)

    # ---------------------------------------------------------------------
    # Label survivors (convert positions -> labels only at the end)
    # ---------------------------------------------------------------------
    if survivors_pos.size == 1:
        winner_label = cand.index[survivors_pos[0]]
        out.loc[winner_label, "tie_result_new"] = 1
    elif survivors_pos.size > 1:
        winner_labels = cand.index.take(survivors_pos)
        out.loc[winner_labels, "tie_result_new"] = 2

    return out[[crd, "tie_result_new"]]


# =============================================================================
# Public API (Pandas only)
# =============================================================================

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
) -> pd.DataFrame:
    """Graph-based dedup with vectorized per-group resolution and Δz clustering.

    This routine builds an undirected graph over non-star rows from
    ``CRD_ID <-> compared_to`` and resolves duplicates within each connected
    component according to a fixed priority order:

      1. ``z_flag_homogenized`` (higher is better; NaN loses)
      2. ``instrument_type_homogenized`` (optional; mapped to numeric via
         ``instrument_type_priority``; higher is better; unknown -> 0)
      3. ``Δz`` clustering: within each group of survivors, contiguous clusters
         along ``z`` whose consecutive gaps are ``<= delta_z_threshold`` are
         collapsed to a single survivor (lexicographically smallest ``CRD_ID``).

    If, after the above steps, a 2-node component still contains (1,1),
    a post-fix is applied:

      * First re-apply flag priority within the pair (NaN loses).
      * If still tied and type is in the configured priority, re-apply type
        priority.
      * If still tied:
          - if ``Δz <= threshold`` -> keep exactly one (min ``CRD_ID``),
          - else -> mark both as ties (2,2).

    Additional rules:
      * Rows with ``z_flag_homogenized == 6`` are stars: they do NOT participate
        in graph edges/tiebreaking. Star singletons get ``tie_result = 3``.
      * Consistency clamp: only ``flag==6`` may have ``tie_result=3``; any other
        accidental 3 becomes 1 (singleton) or 0 (in multi).

    Args:
      df: Input pandas DataFrame.
      tiebreaking_priority: Ordered list of column names used for priority. Only
        numeric columns and the special
        ``instrument_type_homogenized`` are supported.
      instrument_type_priority: Mapping name->score for
        ``instrument_type_homogenized``. Required iff that column appears in
        ``tiebreaking_priority``.
      delta_z_threshold: Non-negative threshold for Δz clustering.
      crd_col: Column name for unique ID (default: "CRD_ID").
      compared_col: Column name with comma-separated neighbor IDs (default:
        "compared_to").
      z_col: Column name for redshift (default: "z").
      tie_col: Output column for labels (default: "tie_result").

    Returns:
      A copy of ``df`` with an updated/created nullable Int8 column
      ``tie_col`` containing:
        - 1: single winner
        - 2: multiple winners (tie)
        - 0: losers
        - 3: star
    """
    required = {crd_col, compared_col, z_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()

    # Canonical CRD ids (string, stripped) used for mapping & deterministic tiebreaks
    crd_norm = out[crd_col].astype("string").str.strip()
    priority_set = set(tiebreaking_priority)

    # =============== 1) Build graph on non-stars + connected components (SciPy) ===============
    zf_series: Optional[pd.Series] = None
    if "z_flag_homogenized" in out.columns:
        zf_series = _to_numeric(out["z_flag_homogenized"])  # 1-D safe

    # Build edges among non-star sources, keep only non-star neighbors that exist
    nodes_edge, edges_uv = _build_edges_fast(
        out, crd_col=crd_col, compared_col=compared_col, zf_series=zf_series
    )
    labels_edge = _connected_components_scipy(len(nodes_edge), edges_uv)

    # Map canonical CRD -> connected component id from the main graph
    if labels_edge.size:
        s_map_index = pd.Index(nodes_edge).astype("string").str.strip()
        s_map = pd.Series(labels_edge.astype("int64"), index=s_map_index)
        mapped = crd_norm.map(s_map)  # NaN for rows not present in the main graph
    else:
        mapped = pd.Series(np.nan, index=out.index, dtype="float64")

    na_mask = mapped.isna().to_numpy()
    gids = np.empty(len(out), dtype=np.int64)

    # Fill group ids from the main graph where available
    if (~na_mask).any():
        gids[~na_mask] = mapped[~na_mask].to_numpy(dtype=np.int64, copy=False)

    # ---------- BRIDGE: inherit gid from already-mapped neighbors (before NA-fallback) ----------
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

    # ---------- Fallback: components inside the TRUE-NA subset (no index alignment) ----------
    if na_mask.any():
        pos_na = np.flatnonzero(na_mask)

        crd_arr_all = crd_norm.to_numpy()
        cmp_arr_all = out[compared_col].astype("string").str.strip().to_numpy()

        crd_arr = crd_arr_all[pos_na]
        cmp_arr = cmp_arr_all[pos_na]

        # Build edges only among the NA subset (default index; no reindexing)
        sub_df = pd.DataFrame({crd_col: crd_arr, compared_col: cmp_arr})
        edges_na = _build_edges_pdf(sub_df, crd_col=crd_col, compared_col=compared_col)
        nodes_na = list(pd.Index(sub_df[crd_col]).astype("string").unique())

        if nodes_na:
            if edges_na.empty:
                comp_map = {nid: i for i, nid in enumerate(nodes_na)}
            else:
                comp_map = _connected_components(nodes_na, edges_na)  # dict id->dense label

            labels_na = np.fromiter(
                (comp_map.get(str(cid), -1) for cid in crd_arr),
                dtype=np.int64,
                count=len(crd_arr),
            )
            start = int(labels_edge.max()) + 1 if labels_edge.size else 0
            gids[pos_na] = start + labels_na
        else:
            start = int(labels_edge.max()) + 1 if labels_edge.size else 0
            gids[pos_na] = np.arange(start, start + len(pos_na), dtype=np.int64)

    # Materialize in the DataFrame and define canonical CRD for later steps
    out["__group__"] = gids
    gid   = out["__group__"]
    crd_s = crd_norm  # canonical CRD for Δz clustering and pair fixers

    # =============== 2) Base labels: singletons=1; star-singletons=3 ===============
    group_sizes = pd.Series(1, index=out.index).groupby(gid).transform("sum")
    is_singleton = group_sizes.eq(1)

    if zf_series is None:
        is_star = pd.Series(False, index=out.index)
    else:
        # Pure boolean (NaN -> False)
        is_star = zf_series.eq(6)

    is_singleton_np = is_singleton.to_numpy(dtype=bool, na_value=False)
    is_star_np      = is_star.to_numpy(dtype=bool, na_value=False)

    tr = np.zeros(len(out), dtype=np.int8)
    tr[is_singleton_np] = 1
    tr[is_singleton_np & is_star_np] = 3  # only star singletons are 3 here

    # =============== 3) Tiebreak in multi-row groups (exclude stars) ===============
    is_multi = ~is_singleton
    non_star = ~is_star
    candidates = (is_multi & non_star)

    survivors = candidates.copy()  # boolean Series aligned to out.index

    # Precompute 1-D numeric scores
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

    # ---------- (B) Apply the configured priority list in order (col1, col2, ...) ----------
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
        s_eff  = scores.where(survivors,      other=-np.inf)
        gmax   = s_eff.groupby(gid).transform("max")
        survivors &= s_eff.eq(gmax)

    # ---------- (C) Δz (priority 3): cluster survivors by threshold, keep 1 per cluster ----------
    thr = float(delta_z_threshold or 0.0)
    if survivors.to_numpy(dtype=bool, na_value=False).any() and thr > 0.0:
        # Within each component, contiguous clusters in z with gaps <= thr are
        # collapsed to a single survivor (min CRD). Distinct clusters remain.
        zvals = _to_numeric(out[z_col]).astype("float64")
        survivors = _collapse_within_dz(survivors, gid, zvals, crd_s, thr)

    # ---------- (D) Final labels for multi-row groups ----------
    n_surv_final = survivors.groupby(gid).transform("sum")
    one_winner   = (survivors & n_surv_final.eq(1))
    multi_winner = (survivors & n_surv_final.ge(2))

    tr[one_winner.to_numpy(dtype=bool, na_value=False)]   = 1
    tr[multi_winner.to_numpy(dtype=bool, na_value=False)] = 2
    # Losers remain 0; multi-row stars are excluded earlier and handled by clamps later if needed.

    # =============== 6) Consistency clamp + robust pair fixers (POSITIONAL) ===============
    out[tie_col] = pd.Series(tr, index=out.index).astype("Int8")

    # Only flag==6 may have tie_result==3; downgrade accidental 3s deterministically
    if zf_series is not None:
        tr_num       = pd.to_numeric(out[tie_col], errors="coerce")
        eq3_np       = tr_num.eq(3.0).to_numpy(dtype=bool, na_value=False)
        is_star_np   = is_star.to_numpy(dtype=bool, na_value=False)
        is_single_np = is_singleton.to_numpy(dtype=bool, na_value=False)

        invalid_3_np = eq3_np & ~is_star_np
        if invalid_3_np.any():
            invalid_3 = pd.Series(invalid_3_np, index=out.index)
            single    = pd.Series(is_single_np, index=out.index)
            # Singletons: 3 -> 1; in multi: 3 -> 0
            out.loc[invalid_3 &  single, tie_col] = np.int8(1)
            out.loc[invalid_3 & ~single, tie_col] = np.int8(0)

    # Prepare common positional arrays/scalars for pair fixers
    pos_all = np.arange(len(out), dtype=np.int64)
    z_num = _to_numeric(out[z_col]).astype("float64")
    f_num = zf_num.fillna(-np.inf)
    t_num = (it_scores.fillna(0.0) if "instrument_type_homogenized" in priority_set else None)
    thr_local = float(delta_z_threshold or 0.0)

    # --- Strict (1,1) pair fixer (positional): resolve by flag → type → Δz deterministically ---
    is_pair  = group_sizes.eq(2)
    pair_one = out[tie_col].eq(1) & is_pair
    both_one = pair_one.groupby(gid).transform("sum").eq(2)
    bad_gids = pd.Index(gid[pair_one & both_one]).unique()

    if len(bad_gids) > 0:
        for gval in bad_gids:
            # positional selection to avoid duplicate-label traps
            mask = (gid.eq(gval) & pair_one)
            pos = np.flatnonzero(mask.to_numpy(dtype=bool, na_value=False))
            if pos.size != 2:
                continue
            p1, p2 = int(pos[0]), int(pos[1])

            # 1) Flag (NaN -> -inf)
            f1, f2 = float(f_num.iloc[p1]), float(f_num.iloc[p2])
            if f1 > f2:
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([1, 0], dtype=np.int8)
                continue
            if f2 > f1:
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([0, 1], dtype=np.int8)
                continue

            # 2) Type (if configured)
            if t_num is not None:
                t1, t2 = float(t_num.iloc[p1]), float(t_num.iloc[p2])
                if t1 > t2:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([1, 0], dtype=np.int8)
                    continue
                if t2 > t1:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array([0, 1], dtype=np.int8)
                    continue

            # 3) Δz
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

    # --- Strict (0,0) pair fixer (positional): resolve by flag → type → Δz; avoid 0–0 unless undecidable ---
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

            # Start from 0,0 and promote deterministically
            # 1) Flag
            f1, f2 = float(f_num.iloc[p1]), float(f_num.iloc[p2])
            if f1 > f2:
                out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(1)
                out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(0)
                continue
            if f2 > f1:
                out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(0)
                out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(1)
                continue

            # 2) Type
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

            # 3) Δz or true tie
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

    out.drop(columns=["__group__"], inplace=True)
    return out
