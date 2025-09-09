# validation_functions.py
from __future__ import annotations

# ==============================
# Standard library
# ==============================
import math
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

# ==============================
# Third-party libraries
# ==============================
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from IPython.display import display, Markdown


def _norm_str(s):
    """Normalize strings; NA-like -> None."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() in {"na", "nan", "<na>", "none", "null"}:
        return None
    return s

# =======================
# =======================
# Intra Source
# =======================
# =======================
def _parse_cmp_list_fast(s: str) -> list[str]:
    """Parse a comma-separated list into trimmed items; NA-like -> []."""
    if s is None:
        return []
    s = str(s).strip()
    if not s or s.lower() in {"na", "nan", "<na>", "none", "null"}:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def validate_intra_source_cells_fast(
    df_final: pd.DataFrame,
    ndp: int = 4,
    source_col: str = "source",
    emit_pairs: bool = True,
    limit_pairs: int | None = None,
):
    """Fast intra-source validator on (source, ra4, dec4) cells.

    Builds directed edges inside each cell using explode and counts pair directions.

    Args:
      df_final: Input dataframe.
      ndp: Decimals to round RA/DEC.
      source_col: Source/catalog column.
      emit_pairs: Whether to emit the pairs dataframe.
      limit_pairs: Optional cap to avoid materializing all missing pairs.

    Returns:
      Dict with pairs, cell_summary, totals, violations, diag_sources.
    """
    # Minimal normalization (avoid unnecessary copies)
    df = df_final[[source_col, "CRD_ID", "ra", "dec", "compared_to"]].copy()
    df["CRD_ID"] = df["CRD_ID"].astype(str)
    # normalize source
    df[source_col] = df[source_col].map(_norm_str).astype("string")
    # rounded cell
    df["ra4"]  = pd.to_numeric(df["ra"], errors="coerce").round(ndp)
    df["dec4"] = pd.to_numeric(df["dec"], errors="coerce").round(ndp)

    # Keep only cells with 2+ objects
    cell_sizes = (
        df.groupby([source_col, "ra4", "dec4"], dropna=False)
          .size().rename("n").reset_index()
    )
    multi_cells = cell_sizes[cell_sizes["n"] >= 2]
    if multi_cells.empty:
        cell_summary = pd.DataFrame(columns=["source","ra4","dec4","n_pairs","n_ok_bi","n_ok_any","n_missing","bi_cov","any_cov"])
        totals = {"n_cells": 0, "n_pairs": 0, "n_ok_bi": 0, "n_ok_any": 0, "n_missing": 0}
        return {
            "pairs": pd.DataFrame(columns=["source","ra4","dec4","A","B","A_in_B","B_in_A","status"]),
            "cell_summary": cell_summary,
            "totals": totals,
            "violations": pd.DataFrame(columns=["source","ra4","dec4","A","B","A_in_B","B_in_A","status"]),
            "diag_sources": pd.DataFrame(columns=["source","n_cells","cells_2plus"]),
        }

    # Subset with 2+ cells only
    df2 = df.merge(multi_cells[[source_col, "ra4", "dec4"]].drop_duplicates(),
                   on=[source_col, "ra4", "dec4"], how="inner")

    # Explode compared_to -> directed edges inside the cell
    df2["_cmp_list"] = df2["compared_to"].map(_parse_cmp_list_fast)
    edges = (
        df2[[source_col, "ra4", "dec4", "CRD_ID", "_cmp_list"]]
        .explode("_cmp_list", ignore_index=True)
        .rename(columns={"CRD_ID": "A", "_cmp_list": "B"})
    )
    edges = edges[edges["B"].notna() & (edges["B"] != "")]
    # Keep only edges whose B also belongs to the same cell
    members = df2[[source_col, "ra4", "dec4", "CRD_ID"]].rename(columns={"CRD_ID":"B"})
    edges = edges.merge(members, on=[source_col, "ra4", "dec4", "B"], how="inner")

    if edges.empty:
        # No directed edges -> all pairs are "missing".
        # Compute aggregates only (fast).
        n_pairs_series = (multi_cells["n"] * (multi_cells["n"] - 1) // 2)
        cell_summary = multi_cells.assign(
            n_pairs=n_pairs_series,
            n_ok_bi=0, n_ok_any=0, n_missing=n_pairs_series,
            bi_cov=0.0, any_cov=0.0
        ).rename(columns={source_col:"source"}).drop(columns=["n"])
        totals = {
            "n_cells":   int(cell_summary.shape[0]),
            "n_pairs":   int(cell_summary["n_pairs"].sum()),
            "n_ok_bi":   0,
            "n_ok_any":  0,
            "n_missing": int(cell_summary["n_pairs"].sum()),
        }
        diag_sources = (
            multi_cells.groupby(source_col)["n"]
            .agg(n_cells="size", cells_2plus=lambda s: (s >= 2).sum())
            .reset_index().rename(columns={source_col:"source"})
            .sort_values("cells_2plus", ascending=False)
        )
        return {
            "pairs": pd.DataFrame(columns=["source","ra4","dec4","A","B","A_in_B","B_in_A","status"]),
            "cell_summary": cell_summary,
            "totals": totals,
            "violations": pd.DataFrame(columns=["source","ra4","dec4","A","B","A_in_B","B_in_A","status"]),
            "diag_sources": diag_sources,
        }

    # Collapse to undirected pairs and count directions (1=one_way, 2=bi)
    a_le_b = (edges["A"] <= edges["B"])
    edges["A_und"] = np.where(a_le_b, edges["A"], edges["B"])
    edges["B_und"] = np.where(a_le_b, edges["B"], edges["A"])

    pair_counts = (
        edges.groupby([source_col, "ra4", "dec4", "A_und", "B_und"], dropna=False)
             .size().rename("n_dir").reset_index()
    )
    pair_counts["status"] = np.where(pair_counts["n_dir"] >= 2, "ok_bi", "ok_one_way")

    # Per-cell aggregates (fast)
    per_cell_found = (
        pair_counts.groupby([source_col, "ra4", "dec4"], dropna=False)
                   .agg(n_ok_bi=("status", lambda s: (s == "ok_bi").sum()),
                        n_ok_any=("status", "size"))
                   .reset_index()
    )
    cell_summary = multi_cells.merge(per_cell_found, on=[source_col, "ra4", "dec4"], how="left").fillna({"n_ok_bi":0, "n_ok_any":0})
    cell_summary["n_pairs"]   = (cell_summary["n"] * (cell_summary["n"] - 1) // 2)
    cell_summary["n_missing"] = cell_summary["n_pairs"] - cell_summary["n_ok_any"]
    cell_summary["bi_cov"]    = np.where(cell_summary["n_pairs"]>0, cell_summary["n_ok_bi"]/cell_summary["n_pairs"], 0.0)
    cell_summary["any_cov"]   = np.where(cell_summary["n_pairs"]>0, cell_summary["n_ok_any"]/cell_summary["n_pairs"], 0.0)
    cell_summary = (cell_summary
                    .rename(columns={source_col:"source"})
                    .drop(columns=["n"])
                    .reset_index(drop=True))

    totals = {
        "n_cells":   int(cell_summary.shape[0]),
        "n_pairs":   int(cell_summary["n_pairs"].sum()),
        "n_ok_bi":   int((pair_counts["status"] == "ok_bi").sum()),
        "n_ok_any":  int(pair_counts.shape[0]),
        "n_missing": int(cell_summary["n_missing"].sum()),
    }

    # Optional pair emission (include "missing" only if requested)
    if not emit_pairs:
        pairs = pd.DataFrame(columns=["source","ra4","dec4","A","B","A_in_B","B_in_A","status"])
        violations = pairs.copy()
    else:
        # Found pairs (fast)
        pairs_found = pair_counts.rename(columns={"A_und":"A", "B_und":"B"})
        # Rebuild direction flags via join on directed edges
        dir1 = edges[["A","B",source_col,"ra4","dec4"]].assign(dir=1)
        dir2 = edges.rename(columns={"A":"B","B":"A"})[["A","B",source_col,"ra4","dec4"]].assign(dir=2)
        both_dir = pd.concat([dir1, dir2], ignore_index=True)
        mark = both_dir.drop_duplicates([source_col,"ra4","dec4","A","B"]).assign(val=True)
        pairs_found = pairs_found.merge(
            mark.rename(columns={"A":"A","B":"B","val":"A_in_B"})[[source_col,"ra4","dec4","A","B","A_in_B"]],
            on=[source_col,"ra4","dec4","A","B"], how="left"
        )
        pairs_found = pairs_found.merge(
            mark.rename(columns={"A":"B","B":"A","val":"B_in_A"})[[source_col,"ra4","dec4","A","B","B_in_A"]],
            on=[source_col,"ra4","dec4","A","B"], how="left"
        )
        pairs_found["A_in_B"] = pairs_found["A_in_B"].fillna(False)
        pairs_found["B_in_A"] = pairs_found["B_in_A"].fillna(False)

        pairs_found = (pairs_found
                       .rename(columns={source_col:"source"})
                       [["source","ra4","dec4","A","B","A_in_B","B_in_A","status"]]
                       .reset_index(drop=True))

        # If "missing" are needed, materialize per cell in a controlled way
        missing_rows = []
        if limit_pairs is None or pairs_found.shape[0] < limit_pairs:
            found_key = set(
                (r.source, r.ra4, r.dec4, r.A, r.B)
                for r in pairs_found.itertuples(index=False)
            )
            for rec in multi_cells.itertuples(index=False):
                src, ra4, dec4, n = getattr(rec, source_col), rec.ra4, rec.dec4, rec.n
                ids = df2[(df2[source_col]==src) & (df2["ra4"].eq(ra4)) & (df2["dec4"].eq(dec4))]["CRD_ID"].tolist()
                ids.sort()
                for a, b in combinations(ids, 2):
                    key = (src, ra4, dec4, a, b)
                    if key not in found_key:
                        missing_rows.append({"source":src, "ra4":ra4, "dec4":dec4,
                                             "A":a, "B":b, "A_in_B":False, "B_in_A":False, "status":"missing"})
            missing_df = pd.DataFrame(missing_rows) if missing_rows else pd.DataFrame(columns=pairs_found.columns)
            pairs = pd.concat([pairs_found, missing_df], ignore_index=True)
        else:
            pairs = pairs_found

        violations = pairs[pairs["status"] != "ok_bi"].copy()

    # Diagnostics by source
    diag_sources = (
        multi_cells.groupby(source_col)["n"]
        .agg(n_cells="size", cells_2plus=lambda s: (s >= 2).sum())
        .reset_index().rename(columns={source_col:"source"})
        .sort_values("cells_2plus", ascending=False)
    )

    return {
        "pairs": pairs,
        "cell_summary": cell_summary,
        "totals": totals,
        "violations": violations,
        "diag_sources": diag_sources,
    }

def explain_intra_source_validation_output(
    res,
    top_k: int = 20,
    df_original: pd.DataFrame | None = None,
    ndp_used: int = 4,
    samples_per_source: int = 2,
    max_sources: int | None = 5,
    source_col: str = "source",
):
    """Render explanations and samples for the intra-source validator."""
    # --- TOTALS ---
    tot = res.get("totals", {})
    display(Markdown(
f"""
### Totals (`res["totals"]`)
- **`n_cells`**: number of cells (**`{source_col}` + `ra`/`dec` rounded to {ndp_used} decimals**) with **≥ 2 objects**.
- **`n_pairs`**: total number of A–B pairs evaluated in those cells.
- **`n_ok_bi`**: pairs with **bidirectionality** (A∈`compared_to`(B) **and** B∈`compared_to`(A)).
- **`n_ok_any`**: pairs with **at least one** direction present (A∈B **or** B∈A).
- **`n_missing`**: pairs with **no direction** present.

**Values:** `{tot}`
"""
    ))

    # === SAMPLES BY SOURCE (right after Totals) ===
    if df_original is not None:
        display(Markdown(
f"""
### Sample groups by `{source_col}`
Below, for each `{source_col}` with at least one cell containing ≥2 objects (at {ndp_used} decimals),
we show up to **{samples_per_source}** cells (the largest ones) and list their objects with:

`CRD_ID, ra, dec, z, z_flag_homogenized, instrument_type_homogenized, compared_to`.
"""
        ))

        # Prepare base DF with the same rounding used in validation
        df_base = df_original.copy()
        df_base[source_col] = df_base[source_col].map(_norm_str).astype("string")
        df_base["ra4"]  = pd.to_numeric(df_base["ra"], errors="coerce").round(ndp_used)
        df_base["dec4"] = pd.to_numeric(df_base["dec"], errors="coerce").round(ndp_used)

        cs = res.get("cell_summary")
        diag = res.get("diag_sources")
        if cs is None or diag is None or getattr(diag, "empty", True):
            display(Markdown("_No eligible cells found for samples._"))
        else:
            # pick sources with at least one cell of 2+ objects
            sources_to_show = diag.loc[diag["cells_2plus"] > 0, "source"].tolist()
            if max_sources is not None:
                sources_to_show = sources_to_show[:max_sources]

            # columns to display (only those present)
            col_candidates = [
                "CRD_ID", "ra", "dec", "z",
                "z_flag_homogenized",
                "instrument_type_homogenized",  # preferred
                "type_homogenized",             # fallback, if it exists
                "compared_to",
            ]
            cols_present = [c for c in col_candidates if c in df_base.columns]

            for src in sources_to_show:
                subcells = cs[cs["source"] == src]
                if subcells.empty:
                    continue

                # prioritize largest cells (more pairs => more objects)
                subcells_sorted = subcells.sort_values("n_pairs", ascending=False).head(samples_per_source)

                display(Markdown(f"#### Source: `{src}`"))
                for _, row in subcells_sorted.iterrows():
                    ra4, dec4 = row["ra4"], row["dec4"]
                    group = df_base[
                        (df_base[source_col] == src) &
                        (df_base["ra4"] == ra4) &
                        (df_base["dec4"] == dec4)
                    ].copy()

                    # light sorting for readability
                    if "z_flag_homogenized" in group.columns:
                        group = group.sort_values(["z_flag_homogenized", "CRD_ID"], ascending=[False, True])
                    else:
                        group = group.sort_values("CRD_ID")

                    meta = (
                        f"- Cell **ra4={ra4}**, **dec4={dec4}** — "
                        f"pairs: {int(row['n_pairs'])}, ok_bi: {int(row['n_ok_bi'])}, "
                        f"any: {int(row['n_ok_any'])}, missing: {int(row['n_missing'])}"
                    )
                    display(Markdown(meta))
                    display(group[cols_present].reset_index(drop=True))

    # --- DIAGNOSTIC BY SOURCE ---
    diag = res.get("diag_sources")
    display(Markdown(
"""
### Diagnostic by `source` (`res["diag_sources"]`)
- **`source`**: catalog/source.
- **`n_cells`**: total number of cells found in that `source` (includes cells with a single object).
- **`cells_2plus`**: number of cells with **≥ 2 objects** (i.e., generating pairs for validation).
"""
    ))
    if diag is not None and hasattr(diag, "empty") and not diag.empty:
        display(diag.head(top_k))
    else:
        display(Markdown("_No rows in `diag_sources`._"))

    # --- CELL SUMMARY ---
    cell_summary = res.get("cell_summary")
    display(Markdown(
f"""
### Coverage per cell (`res["cell_summary"]`)
- **`source`**: catalog.
- **`ra4`, `dec4`**: RA/DEC rounded to {ndp_used} decimals (10⁻⁴ deg ≈ 0.36″).
- **`n_pairs`**: number of A–B pairs in this cell.
- **`n_ok_bi`**: number of bidirectional pairs.
- **`n_ok_any`**: number of pairs with at least one direction present.
- **`n_missing`**: number of pairs with no direction present.
- **`bi_cov`**: bidirectional coverage = `n_ok_bi / n_pairs`.
- **`any_cov`**: “at least one direction” coverage = `n_ok_any / n_pairs`.
"""
    ))
    if cell_summary is not None and hasattr(cell_summary, "empty") and not cell_summary.empty:
        display(cell_summary.head(top_k))
    else:
        display(Markdown("_No rows in `cell_summary`._"))

    # --- PAIRS WITH PROBLEMS ---
    viol = res.get("violations")
    display(Markdown(
"""
### Non-bidirectional pairs (`res["violations"]`)
- **`A`, `B`**: CRD_IDs of the pair.
- **`A_in_B`**: `True` if A appears in B’s `compared_to`.
- **`B_in_A`**: `True` if B appears in A’s `compared_to`.
- **`status`**:
  - `ok_bi` → bidirectional (ideal),
  - `ok_one_way` → only one direction present,
  - `missing` → no direction present.
"""
    ))
    if viol is not None and hasattr(viol, "empty") and not viol.empty:
        display(viol.head(top_k))
    else:
        display(Markdown("_No violations — all pairs are `ok_bi`._"))

# =======================
# =======================
# Cross Sources
# =======================
# =======================
# ------------------------------
# Public API  (star-inclusive graph + star consistency)
# ------------------------------
def validate_tie_results_fast(
    df_final: pd.DataFrame,
    threshold: float = 0.0005,
    max_groups: int | None = None,
    include_rows: bool = False,
) -> Dict[str, object]:
    """
    Unified-graph validator (stars do not connect).

    * Build a single graph EXCLUDING any edge/node that touches a star.
    * Stars for validation are ONLY those with z_flag_homogenized == 6
      (do not infer stars from tie_result == 3).
    * Within each component (non-stars only), apply rules among non-stars.
      Stars do not connect groups nor influence 1/2.

    Args:
      df_final: Input dataframe.
      threshold: Delta-z threshold (pairs with Δz <= threshold count as "too close").
      max_groups: Optional cap on scanned components (non-star components).
      include_rows: Include violating rows (dataframes) in payload.

    Returns:
      Dict with summary and violations.
    """
    # ---------------------------
    # Input prep
    # ---------------------------
    df = df_final.copy()

    # Ensure columns exist
    req = [
        "CRD_ID",
        "compared_to",
        "z_flag_homogenized",
        "tie_result",
        "instrument_type_homogenized",
        "z",
    ]
    for c in req:
        if c not in df.columns:
            df[c] = np.nan

    # Canonical dtypes for core columns
    df["CRD_ID"] = df["CRD_ID"].astype(str)
    df["compared_to"] = df["compared_to"].astype("string")

    # Global numeric views
    zf_all = pd.to_numeric(df["z_flag_homogenized"], errors="coerce")
    tr_all = pd.to_numeric(df["tie_result"], errors="coerce")

    # Stars for VALIDATION = z_flag_homogenized == 6 ONLY
    is_star_row = zf_all.eq(6.0)
    star_ids = set(df.loc[is_star_row, "CRD_ID"].astype(str))

    violations: list[dict] = []

    # ---------------------------
    # Global star consistency checks
    # ---------------------------
    star_must_be_3 = df[is_star_row & ~tr_all.eq(3.0)]
    if not star_must_be_3.empty:
        violations.append(
            {
                "rule": "STAR_MUST_BE_3",
                "message": "Rows with z_flag_homogenized==6 must have tie_result==3.",
                "group_ids": tuple(star_must_be_3["CRD_ID"].astype(str).tolist()),
                "rows": star_must_be_3 if include_rows else None,
            }
        )

    nonstar_cant_be_3 = df[~is_star_row & tr_all.eq(3.0)]
    if not nonstar_cant_be_3.empty:
        violations.append(
            {
                "rule": "NONSTAR_MUST_NOT_BE_3",
                "message": "Rows with z_flag_homogenized!=6 must not have tie_result==3.",
                "group_ids": tuple(nonstar_cant_be_3["CRD_ID"].astype(str).tolist()),
                "rows": nonstar_cant_be_3 if include_rows else None,
            }
        )

    # ---------------------------
    # Build edges (EXCLUDE stars as A and as B, and drop B not in df)
    # ---------------------------
    base = df[["CRD_ID", "compared_to"]].copy()
    cmp_df = (
        base.assign(_cmp_list=base["compared_to"].map(_parse_cmp_list_fast))[["CRD_ID", "_cmp_list"]]
        .explode("_cmp_list", ignore_index=True)
        .rename(columns={"CRD_ID": "A", "_cmp_list": "B"})
    )

    cmp_df["A"] = cmp_df["A"].astype(str)
    cmp_df["B"] = cmp_df["B"].astype("string").str.strip()
    cmp_df = cmp_df[cmp_df["B"].notna() & (cmp_df["B"] != "")]

    # Keep only edges to B that actually exist in df (prevents phantom NaN nodes)
    valid_ids = set(df["CRD_ID"].astype(str))
    if valid_ids:
        cmp_df = cmp_df[cmp_df["B"].isin(valid_ids)]

    # EXCLUDE edges that touch a star (either endpoint)
    if star_ids:
        cmp_df = cmp_df[~cmp_df["A"].isin(star_ids) & ~cmp_df["B"].isin(star_ids)]

    # No edges → no non-star components to validate (still return star stats).
    if cmp_df.empty:
        summary = {
            "n_components": 0,  # only among non-stars
            "n_pairs": 0,
            "n_groups": 0,
            "n_violations": len(violations),
            "by_rule": (pd.Series([v["rule"] for v in violations]).value_counts().to_dict() if violations else {}),
            "n_stars_excluded": int(len(star_ids)),
        }
        return {"summary": summary, "violations": violations}

    # ---------------------------
    # Nodes, undirected unique edges, connected components (non-stars only)
    # ---------------------------
    nodes = pd.Index(pd.unique(pd.concat([cmp_df["A"], cmp_df["B"]], ignore_index=True)))
    id2ix = {cid: i for i, cid in enumerate(nodes)}
    ai = cmp_df["A"].map(id2ix).to_numpy()
    bi = cmp_df["B"].map(id2ix).to_numpy()

    lo = np.minimum(ai, bi)
    hi = np.maximum(ai, bi)
    und = np.stack([lo, hi], axis=1)
    und = und[und[:, 0] != und[:, 1]]

    if und.size == 0:
        # All edges were self-loops (after cleaning) → every node is its own component
        n_comp_all = nodes.size
        labels = np.arange(nodes.size, dtype=np.int64)
    else:
        # Unique undirected edges
        undv = und.view([("x", und.dtype), ("y", und.dtype)])
        und = np.unique(undv).view(und.dtype).reshape(-1, 2)

        n_nodes = nodes.size
        data = np.ones(len(und), dtype=np.int8)
        A = coo_matrix((data, (und[:, 0], und[:, 1])), shape=(n_nodes, n_nodes))
        A = A + A.T
        n_comp_all, labels = connected_components(A, directed=False, return_labels=True)

    # ---------------------------
    # Precompute arrays aligned to `nodes` (non-stars only set)
    # ---------------------------
    # node_df is a view of df aligned to ordered 'nodes'
    node_df = df.set_index(df["CRD_ID"].astype(str), drop=False).reindex(nodes, copy=False)

    zf = pd.to_numeric(node_df["z_flag_homogenized"], errors="coerce").to_numpy()
    tr = pd.to_numeric(node_df["tie_result"], errors="coerce").to_numpy()
    z  = pd.to_numeric(node_df["z"], errors="coerce").to_numpy()

    # Rank for flags (keep the mapping even though stars should not be present here);
    # star(6)=-2, NaN=-1, others numeric.
    zf_rank = np.where(np.isnan(zf), -1.0, zf)
    zf_rank[zf == 6] = -2.0

    # Instrument type priority (example mapping; adjust if you use a different one)
    type_map = {"s": 3, "g": 2, "p": 1}
    it = node_df["instrument_type_homogenized"].map(lambda t: type_map.get(_norm_str(t), 0)).astype(np.int16).to_numpy()

    # Non-star mask among nodes (redundant because we filtered edges to avoid stars,
    # but kept as a safety net if rows slip in).
    nonstar_mask_nodes = (zf != 6) & (tr != 3)

    # ---------------------------
    # Iterate components efficiently
    # ---------------------------
    order = np.argsort(labels, kind="mergesort")
    labels_sorted = labels[order]
    boundaries = np.flatnonzero(np.r_[True, labels_sorted[1:] != labels_sorted[:-1], True])

    n_pairs = 0
    n_groups = 0
    comps_scanned = 0

    for b in range(len(boundaries) - 1):
        if (max_groups is not None) and (comps_scanned >= max_groups):
            break

        start, end = boundaries[b], boundaries[b + 1]
        comp_idx = order[start:end]            # node indices (non-star nodes set)
        comp_ids_str = tuple(nodes[comp_idx].tolist())

        rows_payload_all = node_df.iloc[comp_idx].copy() if include_rows else None

        # Non-star guard (should be all non-stars anyway)
        ns_mask_local = nonstar_mask_nodes[comp_idx]
        ns_idx = comp_idx[ns_mask_local]

        # Components with <2 non-star nodes have no pair/group rules to check
        if ns_idx.size < 2:
            comps_scanned += 1
            continue

        # Slice arrays for the component (non-stars)
        tr_c = np.nan_to_num(tr[ns_idx], nan=-1.0).astype(np.int16)
        it_c = it[ns_idx]
        zf_c = zf_rank[ns_idx]
        z_c  = z[ns_idx]

        m = ns_idx.size
        if m == 2:
            n_pairs += 1
            a, b_ = 0, 1
            tr_a, tr_b = tr_c[a], tr_c[b_]
            zf_a, zf_b = zf_c[a], zf_c[b_]
            ts_a, ts_b = it_c[a], it_c[b_]

            za, zb = z_c[a], z_c[b_]
            dz = np.inf if (np.isnan(za) or np.isnan(zb)) else abs(za - zb)

            vios_local: List[Tuple[str, str]] = []
            if (tr_a in (0, 1)) and (tr_b in (0, 1)) and (tr_a != tr_b):
                # winner must dominate in flag or (if equal) in type;
                # if still tied in flag+type, Δz must be <= threshold to justify a single winner;
                # otherwise this is suspicious.
                win_is_a = (tr_a == 1)
                win_zf, los_zf = (zf_a, zf_b) if win_is_a else (zf_b, zf_a)
                win_ts, los_ts = (ts_a, ts_b) if win_is_a else (ts_b, ts_a)
                cond = (
                    (win_zf > los_zf) or
                    (win_zf == los_zf and win_ts > los_ts) or
                    (win_zf == los_zf and win_ts == los_ts and (dz <= threshold))
                )
                if not cond:
                    vios_local.append((
                        "PAIR_1v0_PRIORITY",
                        "Winner lacks higher flag/type, and Δz is not <= threshold when flag/type tie persists.",
                    ))
            elif (tr_a == 2) and (tr_b == 2):
                # a proper 2–2 requires equal flag/type; and if both z are defined, they should be > threshold apart
                cond = (zf_a == zf_b) and (ts_a == ts_b) and (np.isinf(dz) or (dz > threshold))
                if not cond:
                    vios_local.append((
                        "PAIR_2v2_TIE_CONSISTENCY",
                        "Tie (2,2) requires equal flag/type and Δz>threshold (or undefined z).",
                    ))
            elif (tr_a == 0) and (tr_b == 0):
                vios_local.append((
                    "PAIR_0v0_SUSPECT",
                    "Both eliminated (0,0); check upstream logic.",
                ))
            else:
                vios_local.append((
                    "PAIR_INVALID_TIE_PATTERN",
                    f"Unexpected tie_result pattern: ({tr_a},{tr_b}).",
                ))

            if vios_local:
                for rule, msg in vios_local:
                    violations.append(
                        {
                            "rule": rule,
                            "message": msg,
                            "group_ids": comp_ids_str,
                            "rows": rows_payload_all,
                        }
                    )

        else:
            n_groups += 1
            # Survivors among non-stars are 1 or 2
            surv = np.isin(tr_c, (1, 2))

            # Flag dominance (NaN -> -1 in zf_c; stars never appear here)
            max_flag = np.max(zf_c) if zf_c.size else -np.inf
            if np.any(surv & (zf_c < max_flag)):
                violations.append(
                    {
                        "rule": "GROUP_FLAG_DOMINANCE",
                        "message": "Survivors include members with lower z_flag than group max (rank: star=-2 < NaN=-1 < others).",
                        "group_ids": comp_ids_str,
                        "rows": rows_payload_all,
                    }
                )

            # Type dominance among top-flag
            cand_flag = (zf_c == max_flag)
            max_type = np.max(it_c[cand_flag]) if np.any(cand_flag) else -1
            if np.any(surv & (it_c < max_type)):
                violations.append(
                    {
                        "rule": "GROUP_TYPE_DOMINANCE",
                        "message": "Survivors include members with lower instrument_type than group max among top-flag.",
                        "group_ids": comp_ids_str,
                        "rows": rows_payload_all,
                    }
                )

            # Δz independence among top-flag & top-type survivors
            cand = cand_flag & (it_c == max_type)
            idx = np.where(cand & surv)[0]
            if idx.size >= 2:
                zS = z_c[idx]
                too_close = False
                for i in range(idx.size - 1):
                    zi = zS[i]
                    if np.isnan(zi):
                        continue
                    d = np.abs(zS[i + 1 :] - zi)
                    if np.any(~np.isnan(d) & (d <= threshold)):
                        too_close = True
                        break
                if too_close:
                    violations.append(
                        {
                            "rule": "GROUP_DELTZ_INDEPENDENCE",
                            "message": "Two survivors are <= threshold apart among max-flag & max-type.",
                            "group_ids": comp_ids_str,
                            "rows": rows_payload_all,
                        }
                    )

            # Survivor count (among non-stars)
            n_surv = int(np.sum(surv))
            if n_surv == 0:
                violations.append(
                    {
                        "rule": "GROUP_NO_SURVIVOR_SUSPECT",
                        "message": "No survivors among non-stars inside a star-excluded component.",
                        "group_ids": comp_ids_str,
                        "rows": rows_payload_all,
                    }
                )
            elif n_surv == 1:
                only_idx = int(np.where(surv)[0][0])
                if int(tr_c[only_idx]) != 1:
                    violations.append(
                        {
                            "rule": "GROUP_SINGLE_SURVIVOR_MUST_BE_1",
                            "message": "Exactly one survivor among non-stars but not labeled tie_result == 1.",
                            "group_ids": comp_ids_str,
                            "rows": rows_payload_all,
                        }
                    )
            else:
                if not np.all(tr_c[surv] == 2):
                    violations.append(
                        {
                            "rule": "GROUP_MULTI_SURVIVOR_MUST_BE_2",
                            "message": "Multiple survivors among non-stars but not all labeled tie_result == 2.",
                            "group_ids": comp_ids_str,
                            "rows": rows_payload_all,
                        }
                    )

        comps_scanned += 1

    summary = {
        "n_components": int(n_comp_all),  # components among non-stars only
        "n_pairs": int(n_pairs),
        "n_groups": int(n_groups),
        "n_violations": len(violations),
        "by_rule": (pd.Series([v["rule"] for v in violations]).value_counts().to_dict() if violations else {}),
        "n_stars_excluded": int(len(star_ids)),
    }
    return {"summary": summary, "violations": violations}


def explain_tie_validation_output(
    report: dict,
    show_per_rule: int = 3,
    prefer_cols: list[str] | None = None,
):
    """Explain and display results from validate_tie_results_fast().

    The validator builds a single graph **excluding any edge that touches a star**
    (rows where `z_flag_homogenized == 6` or `tie_result == 3`). All rules are
    applied only among **non-star** nodes inside each connected component.
    Stars must have `tie_result == 3`, never connect groups, and never influence
    1/2 decisions among non-stars.
    """
    if prefer_cols is None:
        prefer_cols = [
            "CRD_ID", "ra", "dec", "z",
            "z_flag_homogenized",
            "instrument_type_homogenized",  # preferred
            "type_homogenized",             # optional fallback if present
            "tie_result",
            "compared_to",
        ]

    summary = report.get("summary") or {}
    violations = report.get("violations") or []

    # ---------------------------
    # Header + Global Summary
    # ---------------------------
    display(Markdown("### Tie-results validation"))
    display(
        Markdown(
            f"""
#### Summary
- **Analyzed components (non-stars only)**: `{summary.get("n_components", 0)}`
- **Pairs (size = 2)**: `{summary.get("n_pairs", 0)}`
- **Groups (size ≥ 3)**: `{summary.get("n_groups", 0)}`
- **Total violations**: `{summary.get("n_violations", 0)}`
- **Stars excluded from the graph** (`z_flag_homogenized==6` or `tie_result==3`): `{summary.get("n_stars_excluded", 0)}`
"""
        )
    )

    # ---------------------------
    # Count by Rule
    # ---------------------------
    by_rule = summary.get("by_rule") or {}
    display(Markdown("#### Violations by rule"))
    if by_rule:
        by_rule_df = (
            pd.DataFrame([(k, v) for k, v in by_rule.items()], columns=["rule", "count"])
            .sort_values("count", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
        display(by_rule_df)
    else:
        display(Markdown("_No violations — everything consistent ✅_"))

    # If there are no violations, stop early
    if not violations:
        return

    # ---------------------------
    # Samples per Rule
    # ---------------------------
    display(Markdown("### Samples per rule"))

    # Group violations by rule
    vio_by_rule: dict[str, list[dict]] = {}
    for v in violations:
        vio_by_rule.setdefault(v.get("rule", "UNKNOWN_RULE"), []).append(v)

    # Order rules by frequency if available
    if by_rule:
        ordered_rules = [r for r, _ in sorted(by_rule.items(), key=lambda x: x[1], reverse=True)]
    else:
        ordered_rules = list(vio_by_rule.keys())

    for rule in ordered_rules:
        V = vio_by_rule.get(rule, [])
        display(Markdown(f"### Rule: `{rule}` — {len(V)} occurrence(s)"))

        # Typical message (from the first occurrence)
        msg = V[0].get("message", "") if V else ""
        if msg:
            display(Markdown(f"> _{msg}_"))

        # Show up to `show_per_rule` examples
        for i, viol in enumerate(V[:max(0, int(show_per_rule))], start=1):
            group_ids = viol.get("group_ids", ())
            rows_obj = viol.get("rows", None)
            rows = rows_obj.copy() if isinstance(rows_obj, pd.DataFrame) else pd.DataFrame()

            display(Markdown(f"**Example {i}** — `group_ids`: `{group_ids}`"))

            # Select columns that exist, preserving the preferred order
            cols_present = [c for c in prefer_cols if c in rows.columns]
            to_show = (rows[cols_present] if cols_present else rows).reset_index(drop=True)

            # Light sorting for readability
            sort_cols: list[tuple[str, bool]] = []
            if "z_flag_homogenized" in to_show.columns:
                sort_cols.append(("z_flag_homogenized", False))
            if "tie_result" in to_show.columns:
                sort_cols.append(("tie_result", False))
            if "CRD_ID" in to_show.columns:
                sort_cols.append(("CRD_ID", True))

            if sort_cols and len(to_show) > 0:
                by = [c for c, _ in sort_cols]
                asc = [a for _, a in sort_cols]
                to_show = to_show.sort_values(by=by, ascending=asc, kind="mergesort")

            display(to_show)

    # ---------------------------
    # Final Notes
    # ---------------------------
    display(
        Markdown(
            """
> **Notes**
>
> - **Star handling (graph construction)**  
>   Any edge that touches a star is removed. Stars are isolated from the
>   non-star connected components and must have `tie_result == 3`.
>
> - **`z_flag_homogenized` ranking (non-stars)**  
>   For non-stars inside a component, flags are ranked as:  
>   `NaN → -1` (lowest) < `other numeric flags` (their numeric value).  
>   (Stars are not present in the component graph.)
>
> - **PAIR_1v0_PRIORITY**  
>   For a 2-node component with tie pattern `(1,0)`, the winner must have a
>   strictly higher flag; if equal, a better `instrument_type`; if still tied,
>   require `Δz < threshold`.
>
> - **PAIR_2v2_TIE_CONSISTENCY**  
>   For a 2-node component with tie pattern `(2,2)`, both members must share
>   the same flag and `instrument_type`, and `Δz > threshold` (or undefined).
>
> - **GROUP_* rules (size ≥ 3, non-stars only)**  
>   Check flag/type dominance and Δz independence among survivors within the
>   same component.
"""
        )
    )


def render_na_compared_to_validation(
    df_final: pd.DataFrame,
    show_max: int = 10,
    cols_to_show: list[str] | None = None,
    assert_if_invalid: bool = False,
):
    """Render a Markdown report for three rules on `compared_to`-NA rows.

    Rules:
      A) If `compared_to` is <NA>, then `tie_result` must be 1.
         Exception: `tie_result` may be 3 only if `z_flag_homogenized == 6`.
      B) (commented out here; kept for reference)
      C) Global: `tie_result == 3` => `z_flag_homogenized == 6`, and optionally the converse.
    """
    # Pretty display imports (OK outside notebook)
    try:
        from IPython.display import display, Markdown
    except Exception:
        def display(x):  # type: ignore
            pass
        def Markdown(x):  # type: ignore
            return x

    # Required columns
    required = {"compared_to", "tie_result", "z_flag_homogenized"}
    missing = sorted(required - set(df_final.columns))
    if missing:
        raise KeyError(f"Missing required columns for validation: {missing}")

    if cols_to_show is None:
        cols_to_show = [
            "CRD_ID", "ra", "dec", "z", "z_flag", "z_err",
            "z_flag_homogenized", "instrument_type", "instrument_type_homogenized",
            "tie_result", "survey", "source", "compared_to",
        ]

    # Normalize compared_to to true NA
    df = df_final.copy()
    df["compared_to"] = (
        df["compared_to"]
          .astype("string").str.strip()
          .replace({
              "": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NA": pd.NA,
              "<NA>": pd.NA, "None": pd.NA, "null": pd.NA
          })
    )

    # Numeric helper
    def _num(s):
        return pd.to_numeric(s, errors="coerce")

    # =============================================================================
    # Rule A: compared_to is NA -> tie_result must be 1 (except flag==6 -> 3)
    # =============================================================================
    na_cmp = df[df["compared_to"].isna()].copy()

    tie_A   = _num(na_cmp["tie_result"]).fillna(-1).astype(int)
    zf_A    = _num(na_cmp["z_flag_homogenized"]).fillna(-1).astype(int)

    valid_A = tie_A.eq(1) | (tie_A.eq(3) & zf_A.eq(6))
    viol_A  = na_cmp.loc[~valid_A].copy()

    total_A   = int(len(na_cmp))
    ok_A      = int(valid_A.sum())
    bad_A     = int(len(viol_A))

    display(Markdown(
f"""
### Validation A: `compared_to` `<NA>` ➜ `tie_result` (with exception flag==6)

- Rows with `compared_to` `<NA>`: **{total_A}**  
- Valid: **{ok_A}**  
- **INVALID:** **{bad_A}**
"""
    ))

    tie_disp_A   = _num(na_cmp["tie_result"]).astype("Int64").astype("string").fillna("<NA>")
    zflag_disp_A = _num(na_cmp["z_flag_homogenized"]).astype("Int64").astype("string").fillna("<NA>")
    ctab_A = pd.crosstab(tie_disp_A, zflag_disp_A, dropna=False)

    display(Markdown("#### Crosstab A: `tie_result` × `z_flag_homogenized` (isolated; showing `<NA>`)"))
    display(ctab_A)

    if bad_A > 0:
        cols_present_A = [c for c in cols_to_show if c in viol_A.columns]
        display(Markdown(f"#### ⚠️ Examples (up to {show_max}) — Rule A"))
        display(viol_A[cols_present_A].head(show_max).reset_index(drop=True))
    else:
        display(Markdown("✅ Rule A OK: all isolated rows follow the rule."))

    # =============================================================================
    # Rule B (GLOBAL): star consistency
    #   B1) If tie_result == 3 → z_flag_homogenized must be 6
    #   B2) If z_flag_homogenized == 6 → tie_result must be 3
    # =============================================================================
    tie_all = _num(df["tie_result"])
    zf_all  = _num(df["z_flag_homogenized"])

    # B1: tie_result==3 without flag==6
    mask_B1 = tie_all.eq(3) & ~zf_all.eq(6)
    viol_B1 = df.loc[mask_B1].copy()

    total_B1 = int(tie_all.eq(3).sum())
    bad_B1   = int(mask_B1.sum())
    ok_B1    = int(total_B1 - bad_B1)

    display(Markdown(
f"""
### Validation B1 (Global): **`tie_result == 3` requires `z_flag_homogenized == 6`**

- Rows with `tie_result == 3`: **{total_B1}**  
- Valid (`flag == 6`): **{ok_B1}**  
- **INVALID (`flag != 6` or `<NA>`):** **{bad_B1}**
"""
    ))

    if bad_B1 > 0:
        cols_present_B1 = [c for c in cols_to_show if c in viol_B1.columns]
        display(Markdown(f"#### ⚠️ Examples (up to {show_max}) — Rule B1"))
        display(viol_B1[cols_present_B1].head(show_max).reset_index(drop=True))
    else:
        display(Markdown("✅ Rule B1 OK: every `tie_result == 3` has `flag == 6`."))

    # B2: flag==6 → tie_result must be 3
    mask_B2 = zf_all.eq(6) & ~tie_all.eq(3)
    viol_B2 = df.loc[mask_B2].copy()

    total_B2 = int(zf_all.eq(6).sum())
    bad_B2   = int(mask_B2.sum())
    ok_B2    = int(total_B2 - bad_B2)

    display(Markdown(
f"""
### Validation B2 (Global): **`z_flag_homogenized == 6` requires `tie_result == 3`**

- Rows with `flag == 6`: **{total_B2}**  
- Valid (`tie_result == 3`): **{ok_B2}**  
- **INVALID (`tie_result != 3`):** **{bad_B2}**
"""
    ))

    if bad_B2 > 0:
        cols_present_B2 = [c for c in cols_to_show if c in viol_B2.columns]
        display(Markdown(f"#### ⚠️ Examples (up to {show_max}) — Rule B2"))
        display(viol_B2[cols_present_B2].head(show_max).reset_index(drop=True))
    else:
        display(Markdown("✅ Rule B2 OK: every `flag == 6` has `tie_result == 3`."))

    # =============================================================================
    # Assertion (optional) — fail if ANY rule is violated
    # =============================================================================
    if assert_if_invalid and (bad_A > 0 or bad_B1 > 0 or bad_B2 > 0):
        raise AssertionError(
            "Validation failed:\n"
            f"- Rule A violations: {bad_A}\n"
            f"- Rule B1 (tie==3 -> flag==6) violations: {bad_B1}\n"
            f"- Rule B2 (flag==6 -> tie==3) violations: {bad_B2}\n"
            "See displayed tables for examples."
        )

    return {
        # Rule A
        "rule_na_cmp_total": total_A,
        "rule_na_cmp_valid": ok_A,
        "rule_na_cmp_invalid": bad_A,
        "rule_na_cmp_crosstab": ctab_A,
        "rule_na_cmp_violations": viol_A,

        # Rule B1 (global: tie==3 -> flag==6)
        "rule_global_tie3_total": total_B1,
        "rule_global_tie3_valid": ok_B1,
        "rule_global_tie3_invalid": bad_B1,
        "rule_global_tie3_violations": viol_B1,

        # Rule B2 (global: flag==6 -> tie==3)
        "rule_global_flag6_total": total_B2,
        "rule_global_flag6_valid": ok_B2,
        "rule_global_flag6_invalid": bad_B2,
        "rule_global_flag6_violations": viol_B2,
    }



# =======================
# =======================
# Manual Validation
# =======================
# =======================
def analyze_groups_by_compared_to_fast(
    df_final: pd.DataFrame,
    threshold: float = 0.0005,
    max_groups: int | None = 10000,
    max_examples_per_case: int = 5,
    desired_order: list[str] | None = None,
    final_columns: list[str] | None = None,
    render: bool = True,
) -> dict:
    """Fast group analyzer using sparse graph + connected_components.

    Preserves buckets:
      - CASE1_* (pairs) and CASE2_* (groups ≥3), with *_small/_large and *_same/_diff
      - TIE_FLAG_TYPE_BREAK_PAIR / _GROUP
      - SAME_FLAG_DIFF_TYPE
      - SAME_SOURCE_PAIR

    Args:
      df_final: Input dataframe.
      threshold: Δz threshold for small/large classification.
      max_groups: Max number of components to scan.
      max_examples_per_case: Max examples to render per bucket.
      desired_order: Preferred column order in rendering.
      final_columns: Final subset of columns for tables.
      render: Whether to display Markdown/tables.

    Returns:
      Dict with processed_groups, groups_by_case, case_descriptions, summary_counts.
    """
    
    # --------- display defaults
    if desired_order is None:
        desired_order = [
            "CRD_ID", "id", "ra", "dec", "z", "z_flag", "z_err", "instrument_type",
            "survey", "source", "z_flag_homogenized", "instrument_type_homogenized",
            "tie_result", "compared_to", "role",
        ]
    if final_columns is None:
        final_columns = [
            "CRD_ID", "ra", "dec", "z", "z_flag", "z_err",
            "z_flag_homogenized", "instrument_type", "instrument_type_homogenized",
            "tie_result", "survey", "source", "compared_to",
        ]

    # --------- filter rows with non-empty compared_to (string-aware)
    df = df_final.copy()
    cmp = (
        df[["CRD_ID", "compared_to"]]
        .assign(compared_to=lambda x: x["compared_to"].astype("string"))
    )
    nonempty_mask = cmp["compared_to"].notna() & (cmp["compared_to"].str.strip() != "")
    df_cmp = df.loc[nonempty_mask].copy()
    n_rows_with_cmp = int(len(df_cmp))
    if n_rows_with_cmp == 0:
        # Nothing to process
        out = {
            "processed_groups": 0,
            "groups_by_case": {k: [] for k in [
                "CASE1_small_same", "CASE1_small_diff", "CASE1_large_same", "CASE1_large_diff",
                "CASE2_small_same", "CASE2_small_diff", "CASE2_large_same", "CASE2_large_diff",
                "TIE_FLAG_TYPE_BREAK_PAIR", "TIE_FLAG_TYPE_BREAK_GROUP",
                "SAME_FLAG_DIFF_TYPE", "SAME_SOURCE_PAIR",
            ]},
            "case_descriptions": {},
            "summary_counts": {},
        }
        if render:
            try:
                from IPython.display import display, Markdown
            except Exception:
                def display(x): pass  # type: ignore
                def Markdown(x): return x  # type: ignore
            display(Markdown("### Manual group analysis via `compared_to`"))
            display(Markdown("- No rows with non-empty `compared_to`."))
        return out

    # --------- vectorized edges from compared_to
    tmp = df_cmp[["CRD_ID", "compared_to"]].copy()
    tmp["_nb_list"] = tmp["compared_to"].map(_parse_cmp_list_fast)
    edges = (
        tmp[["CRD_ID", "_nb_list"]]
        .explode("_nb_list", ignore_index=True)
        .rename(columns={"CRD_ID": "A", "_nb_list": "B"})
    )
    edges["A"] = edges["A"].astype(str)
    edges["B"] = edges["B"].astype("string").fillna("")
    edges = edges[edges["B"].str.strip() != ""]

    if edges.empty:
        # No real edges -> no groups
        out = {
            "processed_groups": 0,
            "groups_by_case": {k: [] for k in [
                "CASE1_small_same", "CASE1_small_diff", "CASE1_large_same", "CASE1_large_diff",
                "CASE2_small_same", "CASE2_small_diff", "CASE2_large_same", "CASE2_large_diff",
                "TIE_FLAG_TYPE_BREAK_PAIR", "TIE_FLAG_TYPE_BREAK_GROUP",
                "SAME_FLAG_DIFF_TYPE", "SAME_SOURCE_PAIR",
            ]},
            "case_descriptions": {},
            "summary_counts": {},
        }
        if render:
            try:
                from IPython.display import display, Markdown
            except Exception:
                def display(x): pass  # type: ignore
                def Markdown(x): return x  # type: ignore
            display(Markdown("### Manual group analysis via `compared_to`"))
            display(Markdown("- No edges after cleaning `compared_to`."))
        return out

    # --------- normalize nodes and deduplicate undirected edges
    nodes = pd.Index(pd.unique(pd.concat([edges["A"], edges["B"]], ignore_index=True)))
    id2ix = pd.Series(np.arange(nodes.size), index=nodes)  # Series for fast map
    ai = id2ix.loc[edges["A"]].to_numpy()
    bi = id2ix.loc[edges["B"]].to_numpy()
    lo = np.minimum(ai, bi)
    hi = np.maximum(ai, bi)
    und = np.stack([lo, hi], axis=1)
    und = und[und[:, 0] != und[:, 1]]
    if und.size == 0:
        # only self-loops (discarded)
        out = {
            "processed_groups": 0,
            "groups_by_case": {k: [] for k in [
                "CASE1_small_same", "CASE1_small_diff", "CASE1_large_same", "CASE1_large_diff",
                "CASE2_small_same", "CASE2_small_diff", "CASE2_large_same", "CASE2_large_diff",
                "TIE_FLAG_TYPE_BREAK_PAIR", "TIE_FLAG_TYPE_BREAK_GROUP",
                "SAME_FLAG_DIFF_TYPE", "SAME_SOURCE_PAIR",
            ]},
            "case_descriptions": {},
            "summary_counts": {},
        }
        if render:
            try:
                from IPython.display import display, Markdown
            except Exception:
                def display(x): pass  # type: ignore
                def Markdown(x): return x  # type: ignore
            display(Markdown("### Manual group analysis via `compared_to`"))
            display(Markdown("- Only self-loops; no useful groups."))
        return out

    # deduplicate edges
    und_view = und.view([('x', und.dtype), ('y', und.dtype)])
    und = np.unique(und_view).view(und.dtype).reshape(-1, 2)

    # --------- sparse graph and components
    n = nodes.size
    data = np.ones(len(und), dtype=np.int8)
    A = coo_matrix((data, (und[:, 0], und[:, 1])), shape=(n, n))
    A = A + A.T
    n_comp, labels = connected_components(A, directed=False, return_labels=True)

    # --------- fast indexing by CRD_ID
    base = df_cmp.set_index(df_cmp["CRD_ID"].astype(str), drop=False)
    present_mask = nodes.isin(base.index)
    comp_ids = [np.where(labels == k)[0] for k in range(n_comp)]

    # --------- quick helpers
    def _pairwise_max_and_all_leq(arr: np.ndarray, thr: float) -> tuple[float, bool]:
        """Return (max_pairwise_abs_diff, all_pairs_leq_thr) with early-stop."""
        m = arr.size
        if m <= 1:
            return 0.0, True
        max_d = 0.0
        all_leq = True
        for i in range(m - 1):
            d = np.abs(arr[i+1:] - arr[i])
            md = float(d.max(initial=0.0))
            if md > max_d:
                max_d = md
            if all_leq and np.any(d > thr):
                all_leq = False
        return max_d, all_leq

    def _same_survey_nonempty(vals: pd.Series) -> bool:
        vs = vals.astype(str).str.strip()
        vs = vs[vs != ""]
        return vs.nunique(dropna=True) == 1

    # --------- output buckets
    groups_by_case: dict[str, list[pd.DataFrame]] = {
        "CASE1_small_same": [],
        "CASE1_small_diff": [],
        "CASE1_large_same": [],
        "CASE1_large_diff": [],
        "CASE2_small_same": [],
        "CASE2_small_diff": [],
        "CASE2_large_same": [],
        "CASE2_large_diff": [],
        "TIE_FLAG_TYPE_BREAK_PAIR": [],
        "TIE_FLAG_TYPE_BREAK_GROUP": [],
        "SAME_FLAG_DIFF_TYPE": [],
        "SAME_SOURCE_PAIR": [],
    }
    case_descriptions = {
        "CASE1_small_same": f"pair with Δz ≤ {threshold} from the same survey",
        "CASE1_small_diff": f"pair with Δz ≤ {threshold} from different surveys",
        "CASE1_large_same": f"pair with Δz > {threshold} from the same survey",
        "CASE1_large_diff": f"pair with Δz > {threshold} from different surveys",
        "CASE2_small_same": f"group (≥3) with all Δz ≤ {threshold} from the same survey",
        "CASE2_small_diff": f"group (≥3) with all Δz ≤ {threshold} from different surveys",
        "CASE2_large_same": f"group (≥3) with some Δz > {threshold} from the same survey",
        "CASE2_large_diff": f"group (≥3) with some Δz > {threshold} from different surveys",
        "TIE_FLAG_TYPE_BREAK_PAIR": "pair with equal z_flag_homogenized, different instrument_type_homogenized, different surveys",
        "TIE_FLAG_TYPE_BREAK_GROUP": "group (≥3) with equal z_flag_homogenized, different instrument_type_homogenized, different surveys",
        "SAME_FLAG_DIFF_TYPE": "group (≥3) with same z_flag_homogenized but at least one differing instrument_type_homogenized",
        "SAME_SOURCE_PAIR": "pair with identical source (normalized, non-null)",
    }

    processed_groups = 0

    # --------- linear iteration by component (with early cap)
    for k in range(n_comp):
        if max_groups is not None and processed_groups >= max_groups:
            break

        gidx = comp_ids[k]                   # global node indices
        present = present_mask[gidx]         # nodes that exist in df_cmp
        # materialize the group with present rows
        group_ids = nodes[gidx[present]].tolist()
        if len(group_ids) < 2:
            continue  # ignore singletons

        group = base.loc[group_ids].copy()

        # "role" only to mark the seed row (display)
        start_id = group_ids[0]
        group["role"] = np.where(group["CRD_ID"].astype(str).eq(start_id), "principal", "compared")

        # quick metrics
        z_vals = pd.to_numeric(group["z"], errors="coerce").to_numpy()
        max_dz, all_leq = _pairwise_max_and_all_leq(z_vals[np.isfinite(z_vals)], threshold)
        same_survey = _same_survey_nonempty(group["survey"])

        # classification
        if len(group) == 2:
            key = (
                "CASE1_small_same" if (max_dz <= threshold and same_survey) else
                "CASE1_small_diff" if (max_dz <= threshold) else
                "CASE1_large_same" if (same_survey) else
                "CASE1_large_diff"
            )
        else:
            key = (
                "CASE2_small_same" if (all_leq and same_survey) else
                "CASE2_small_diff" if (all_leq) else
                "CASE2_large_same" if (same_survey) else
                "CASE2_large_diff"
            )

        # reorder columns for readability
        all_columns = list(group.columns)
        ordered_columns = (desired_order or []) + [c for c in all_columns if c not in (desired_order or [])]
        group = group.reindex(columns=ordered_columns)
        groups_by_case[key].append(group)

        # extra buckets
        flags = set(group["z_flag_homogenized"].dropna())
        types = set(group["instrument_type_homogenized"].dropna())
        surveys_in_group = set(group["survey"].dropna())
        if len(surveys_in_group) > 1 and len(flags) == 1 and len(types) > 1:
            if len(group) == 2:
                groups_by_case["TIE_FLAG_TYPE_BREAK_PAIR"].append(group)
            else:
                groups_by_case["TIE_FLAG_TYPE_BREAK_GROUP"].append(group)
        if len(flags) == 1 and len(types) > 1 and len(group) > 2:
            groups_by_case["SAME_FLAG_DIFF_TYPE"].append(group)
        if len(group) == 2:
            src = group["source"].dropna().astype(str).str.strip().str.lower()
            if src.size == 2 and src.nunique() == 1:
                groups_by_case["SAME_SOURCE_PAIR"].append(group)

        processed_groups += 1

    # --------- summary
    summary_counts = {k: len(v) for k, v in groups_by_case.items()}

    # --------- rendering (with diversity sampling by survey signature)
    if render:
        try:
            from IPython.display import display, Markdown
        except Exception:
            def display(x): pass  # type: ignore
            def Markdown(x): return x  # type: ignore

        def survey_signature(df: pd.DataFrame) -> tuple[str, ...]:
            vals = df["survey"].dropna().astype(str).unique().tolist()
            return tuple(sorted(vals)) if len(vals) else ("<MISSING>",)

        display(Markdown(f"### Manual group analysis via `compared_to` (fast)"))
        display(Markdown(
            f"- Rows with **non-empty** `compared_to`: **{n_rows_with_cmp}**  \n"
            f"- Unique connected groups processed: **{processed_groups}**  \n"
            f"- Δz threshold: **{threshold}**"
        ))

        if any(summary_counts.values()):
            summary_df = pd.DataFrame(
                sorted(summary_counts.items(), key=lambda x: (-x[1], x[0])),
                columns=["case", "count"]
            )
            display(Markdown("#### Groups per bucket"))
            display(summary_df.reset_index(drop=True))
        else:
            display(Markdown("_No groups found under the current filters._"))

        for case_name, groups in groups_by_case.items():
            if not groups:
                continue
            # diversity by survey signature
            seen_sigs = set()
            selection, leftovers = [], []
            for g in groups:
                sig = survey_signature(g)
                (selection if sig not in seen_sigs else leftovers).append(g)
                seen_sigs.add(sig)
                if len(selection) >= max_examples_per_case:
                    break
            i = 0
            while len(selection) < max_examples_per_case and i < len(leftovers):
                selection.append(leftovers[i]); i += 1

            desc = case_descriptions.get(case_name, case_name)
            display(Markdown(f"#### {case_name} — {desc}  \nFound: **{len(groups)}** group(s)"))
            for g in selection:
                to_show = g.copy()
                cols_present = [c for c in (final_columns or []) if c in to_show.columns]
                if cols_present:
                    to_show = to_show.reindex(columns=cols_present)
                display(to_show.reset_index(drop=True))
                display(Markdown("---"))

    return {
        "processed_groups": processed_groups,
        "groups_by_case": groups_by_case,
        "case_descriptions": case_descriptions,
        "summary_counts": summary_counts,
    }

def _parse_tokens(s: pd.Series) -> pd.Series:
    """Split CSV-like strings into lists of stripped tokens (''/NA -> [])."""
    s = s.astype("string")
    lst = s.str.split(",")
    out = []
    for tokens in lst:
        if not tokens:
            out.append([])
            continue
        toks = [t.strip() for t in tokens if t and t.strip()]
        out.append(toks)
    return pd.Series(out, index=s.index, dtype="object")

def _cmp_na_like_mask(df: pd.DataFrame, cmp_col: str) -> pd.Series:
    """True when compared_to is NA-like (NA or only whitespace)."""
    s = df[cmp_col].astype("string")
    return s.isna() | (s.str.strip() == "")

def _only_star_neighbors_mask(
    df_proc: pd.DataFrame,
    df_orig: pd.DataFrame,
    *,
    key: str,
    cmp_col: str,
    zflag_col: str = "z_flag_homogenized",
) -> pd.Series:
    """True when compared_to is non-empty and all neighbors are star IDs
    (as defined by df_original[z_flag_homogenized] == 6)."""
    if zflag_col not in df_orig.columns:
        # If we cannot tell who is a star, be conservative: no rows qualify
        return pd.Series(False, index=df_proc.index)

    # Build set of star IDs from original
    star_ids = set(
        df_orig.loc[
            pd.to_numeric(df_orig[zflag_col], errors="coerce").eq(6.0), key
        ].astype("string")
    )

    s = df_proc[cmp_col].astype("string")
    non_empty = ~(s.isna() | (s.str.strip() == ""))
    tokens = _parse_tokens(s.fillna(""))

    def _all_star(tok_list):
        if not tok_list:
            return False
        return all((t in star_ids) for t in tok_list)

    only_star = tokens.map(_all_star)
    return (non_empty & only_star).astype(bool)

def _check_preserve_block(
    df_original: pd.DataFrame,
    df_processed: pd.DataFrame,
    mask_proc: pd.Series,
    *,
    key: str,
    tr_col: str,
    max_examples: int,
) -> Dict:
    """Compare tie_result for the subset indicated by mask_proc."""
    proc_sub = df_processed.loc[mask_proc, [key, tr_col]].copy()
    proc_sub = proc_sub.rename(columns={tr_col: f"{tr_col}_proc"})

    orig_min = df_original[[key, tr_col]].rename(columns={tr_col: f"{tr_col}_orig"})
    merged = proc_sub.merge(orig_min, on=key, how="left", validate="m:1")

    missing_in_orig_mask = merged[f"{tr_col}_orig"].isna() & ~merged[f"{tr_col}_proc"].isna()
    n_missing_in_original = int(missing_in_orig_mask.sum())

    tr_proc = pd.to_numeric(merged[f"{tr_col}_proc"], errors="coerce")
    tr_orig = pd.to_numeric(merged[f"{tr_col}_orig"], errors="coerce")
    equal_mask = (tr_proc == tr_orig) | (tr_proc.isna() & tr_orig.isna())
    mismatch_mask = ~equal_mask & ~missing_in_orig_mask

    mismatches_df = merged.loc[mismatch_mask, [key, f"{tr_col}_proc", f"{tr_col}_orig"]].copy()
    if len(mismatches_df) > max_examples:
        mismatches_df = mismatches_df.head(max_examples)

    return {
        "n_checked": int(len(merged)),
        "n_missing_in_original": n_missing_in_original,
        "n_mismatches": int(mismatch_mask.sum()),
        "mismatches": mismatches_df,
    }

def _check_star_must_be_3(
    df_processed: pd.DataFrame,
    mask_proc: pd.Series,
    *,
    key: str,
    tr_col: str,
    max_examples: int,
) -> dict:
    """Check that stars under mask have tie_result == 3.

    Args:
        df_processed: Processed DataFrame.
        mask_proc: Boolean mask selecting candidate rows.
        key: ID column name.
        tr_col: Tie result column name.
        max_examples: Maximum number of example rows to return.

    Returns:
        Dict with counts and example rows where tie_result != 3.
    """
    sub = df_processed.loc[mask_proc, [key, tr_col]].copy()
    tr_proc = pd.to_numeric(sub[tr_col], errors="coerce")
    bad = ~(tr_proc == 3)

    examples = sub.loc[bad, [key, tr_col]].copy()
    examples = examples.rename(columns={tr_col: f"{tr_col}_proc"})
    if len(examples) > max_examples:
        examples = examples.head(max_examples)

    return {
        "n_checked": int(len(sub)),
        "n_bad": int(bad.sum()),
        "examples": examples,
    }


def validate_tie_preservation(
    df_original: pd.DataFrame,
    df_processed: pd.DataFrame,
    *,
    key: str = "CRD_ID",
    tr_col: str = "tie_result",
    cmp_col: str = "compared_to",
    max_examples: int = 20,
) -> dict:
    """Validate tie_result preservation in NA-like and star-neighbor cases.

    Rules:
        A) If compared_to is NA-like -> must preserve tie_result.
        B) If compared_to has only star neighbors:
           - Non-star row -> must preserve tie_result.
           - Star row     -> must have tie_result == 3.

    Args:
        df_original: Original DataFrame.
        df_processed: Processed DataFrame.
        key: ID column name.
        tr_col: Tie result column name.
        cmp_col: Compared-to column name.
        max_examples: Maximum number of mismatches to show.

    Returns:
        Dict with per-case results, totals, and examples.
    """
    # A) NA-like case
    mask_na = _cmp_na_like_mask(df_processed, cmp_col)

    # B) Only-star neighbors
    mask_only_star = _only_star_neighbors_mask(
        df_processed, df_original, key=key, cmp_col=cmp_col, zflag_col="z_flag_homogenized"
    )

    # Identify stars in processed
    zflag_proc = pd.to_numeric(df_processed.get("z_flag_homogenized"), errors="coerce")
    mask_is_star_proc = zflag_proc.eq(6.0)

    # Split into star / non-star cases
    mask_only_star_star = (mask_only_star & mask_is_star_proc).fillna(False)
    mask_only_star_nonstar = (mask_only_star & ~mask_is_star_proc).fillna(False)

    # Run validations
    res_na = _check_preserve_block(
        df_original, df_processed, mask_na, key=key, tr_col=tr_col, max_examples=max_examples
    )
    res_star_preserve = _check_preserve_block(
        df_original, df_processed, mask_only_star_nonstar, key=key, tr_col=tr_col, max_examples=max_examples
    )
    res_star_must3 = _check_star_must_be_3(
        df_processed, mask_only_star_star, key=key, tr_col=tr_col, max_examples=max_examples
    )

    # Combine examples
    mismatches_all = pd.concat(
        [res_na["mismatches"], res_star_preserve["mismatches"]], ignore_index=True
    )
    if len(mismatches_all) > max_examples:
        mismatches_all = mismatches_all.head(max_examples)

    ok = (
        (res_na["n_mismatches"] == 0 and res_na["n_missing_in_original"] == 0)
        and (res_star_preserve["n_mismatches"] == 0 and res_star_preserve["n_missing_in_original"] == 0)
        and (res_star_must3["n_bad"] == 0)
    )

    return {
        "ok": ok,
        "na_like": {
            "n_checked": res_na["n_checked"],
            "n_missing_in_original": res_na["n_missing_in_original"],
            "n_mismatches": res_na["n_mismatches"],
        },
        "only_star_neighbors_preserve": {
            "n_checked": res_star_preserve["n_checked"],
            "n_missing_in_original": res_star_preserve["n_missing_in_original"],
            "n_mismatches": res_star_preserve["n_mismatches"],
        },
        "only_star_neighbors_star3": {
            "n_checked": res_star_must3["n_checked"],
            "n_bad": res_star_must3["n_bad"],
        },
        "n_checked_total": int(
            res_na["n_checked"] + res_star_preserve["n_checked"] + res_star_must3["n_checked"]
        ),
        "n_missing_in_original_total": int(
            res_na["n_missing_in_original"] + res_star_preserve["n_missing_in_original"]
        ),
        "n_mismatches_total": int(
            res_na["n_mismatches"] + res_star_preserve["n_mismatches"]
        ),
        "n_star_must3_bad_total": int(res_star_must3["n_bad"]),
        "mismatches": mismatches_all,
        "star_must3_examples": res_star_must3["examples"],
    }


def explain_tie_preservation(result: dict, *, title: str = "Tie-result preservation check") -> str:
    """Render human-readable summary of preservation validation results.

    Args:
        result: Dict returned by validate_tie_preservation.
        title: Report title.

    Returns:
        Text summary.
    """
    lines = []
    lines.append(f"{title}")
    lines.append("-" * len(title))

    # Case A
    lines.append("Case A — compared_to NA-like (preserve tie_result)")
    lines.append(f"  Checked: {result['na_like']['n_checked']}")
    lines.append(f"  Missing in original: {result['na_like']['n_missing_in_original']}")
    lines.append(f"  Mismatches: {result['na_like']['n_mismatches']}")

    # Case B.1
    lines.append("Case B1 — only star neighbors & row is NON-STAR (preserve)")
    lines.append(f"  Checked: {result['only_star_neighbors_preserve']['n_checked']}")
    lines.append(f"  Missing in original: {result['only_star_neighbors_preserve']['n_missing_in_original']}")
    lines.append(f"  Mismatches: {result['only_star_neighbors_preserve']['n_mismatches']}")

    # Case B.2
    lines.append("Case B2 — only star neighbors & row IS STAR (require tie_result==3)")
    lines.append(f"  Checked: {result['only_star_neighbors_star3']['n_checked']}")
    lines.append(f"  Not equal to 3: {result['only_star_neighbors_star3']['n_bad']}")

    lines.append("")
    lines.append(f"Total checked: {result['n_checked_total']}")
    lines.append(f"Total missing in original (preserve-cases): {result['n_missing_in_original_total']}")
    lines.append(f"Total mismatches (preserve-cases): {result['n_mismatches_total']}")
    lines.append(f"Total star-not-3 (must-be-3 case): {result['n_star_must3_bad_total']}")
    lines.append(f"Status: {'OK ✅' if result['ok'] else 'FAIL ❌'}")

    # Examples
    mism = result.get("mismatches", None)
    if mism is not None and len(mism) > 0:
        lines.append("\nExamples (preserve mismatches: proc -> orig):")
        for _, row in mism.iterrows():
            lines.append(f"  {row.iloc[0]}: {row.iloc[1]} -> {row.iloc[2]}")

    star_bad = result.get("star_must3_examples", None)
    if star_bad is not None and len(star_bad) > 0:
        lines.append("\nExamples (star must be 3 — proc values):")
        for _, row in star_bad.iterrows():
            lines.append(f"  {row.iloc[0]}: proc={row.iloc[1]} (expected 3)")

    return "\n".join(lines)
