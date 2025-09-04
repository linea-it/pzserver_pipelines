# validation_functions.py
from __future__ import annotations

import math
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations

from IPython.display import display, Markdown

from IPython.display import display, Markdown

def _norm_str(s):
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
    if s is None:
        return []
    s = str(s).strip()
    if not s or s.lower() in {"na", "nan", "<na>", "none", "null"}:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def validate_intra_source_cells(df_final: pd.DataFrame, ndp: int = 4, source_col: str = "source"):
    """
    Para objetos de mesma 'source' (coluna `source_col`) e mesma célula (ra/dec arredondados a `ndp`),
    verifica se foram comparados entre si via 'compared_to'.
    """
    df = df_final.copy()

    # Normalizações
    df["CRD_ID"] = df["CRD_ID"].astype(str)
    df[source_col] = df[source_col].map(_norm_str).astype("string")
    df["compared_to"] = df["compared_to"].astype("string")
    df["ra4"]  = pd.to_numeric(df["ra"], errors="coerce").round(ndp)
    df["dec4"] = pd.to_numeric(df["dec"], errors="coerce").round(ndp)
    df["cmp_set"] = df["compared_to"].map(_parse_cmp_list_fast)

    rows = []
    # (1) agrupar por source
    for src, gsrc in df.groupby(source_col, dropna=False):
        # (2) dentro da source, agrupar por célula ra/dec arredondada
        for (ra4, dec4), sub in gsrc.groupby(["ra4", "dec4"], dropna=False):
            if len(sub) < 2:
                continue
            id_to_set = dict(zip(sub["CRD_ID"], sub["cmp_set"]))
            ids = list(sub["CRD_ID"])
            for a, b in combinations(ids, 2):
                a_in_b = a in id_to_set.get(b, set())
                b_in_a = b in id_to_set.get(a, set())
                status = "ok_bi" if (a_in_b and b_in_a) else ("ok_one_way" if (a_in_b or b_in_a) else "missing")
                rows.append({
                    "source": src, "ra4": ra4, "dec4": dec4,
                    "A": a, "B": b, "A_in_B": a_in_b, "B_in_A": b_in_a, "status": status
                })

    pairs = pd.DataFrame(rows)

    if pairs.empty:
        cell_summary = pd.DataFrame(columns=["source","ra4","dec4","n_pairs","n_ok_bi","n_ok_any","n_missing","bi_cov","any_cov"])
        totals = {"n_cells": 0, "n_pairs": 0, "n_ok_bi": 0, "n_ok_any": 0, "n_missing": 0}
    else:
        g = pairs.groupby(["source","ra4","dec4"], dropna=False)
        cell_summary = g.agg(
            n_pairs=("status","size"),
            n_ok_bi=("status", lambda s: (s == "ok_bi").sum()),
            n_ok_any=("status", lambda s: s.isin(["ok_bi", "ok_one_way"]).sum()),
            n_missing=("status", lambda s: (s == "missing").sum()),
        ).reset_index()
        cell_summary["bi_cov"]  = cell_summary["n_ok_bi"] / cell_summary["n_pairs"]
        cell_summary["any_cov"] = cell_summary["n_ok_any"] / cell_summary["n_pairs"]

        totals = {
            "n_cells":   int(cell_summary.shape[0]),
            "n_pairs":   int(pairs.shape[0]),
            "n_ok_bi":   int((pairs["status"] == "ok_bi").sum()),
            "n_ok_any":  int(pairs["status"].isin(["ok_bi","ok_one_way"]).sum()),
            "n_missing": int((pairs["status"] == "missing").sum()),
        }

    violations = pairs[pairs["status"] != "ok_bi"].copy()

    # extra: diagnóstico por source (qtd de células com 2+ objetos)
    cell_sizes = (
        df.groupby([source_col, "ra4", "dec4"], dropna=False)
          .size().reset_index(name="n")
    )
    diag_sources = (
        cell_sizes.groupby(source_col)["n"]
        .agg(n_cells="size", cells_2plus=lambda s: (s >= 2).sum())
        .reset_index()
        .rename(columns={source_col: "source"})
        .sort_values("cells_2plus", ascending=False)
    )

    return {
        "pairs": pairs,
        "cell_summary": cell_summary,
        "totals": totals,
        "violations": violations,
        "diag_sources": diag_sources,
    }


def validate_intra_source_cells_fast(
    df_final: pd.DataFrame,
    ndp: int = 4,
    source_col: str = "source",
    emit_pairs: bool = True,
    limit_pairs: int | None = None,   # limite opcional p/ evitar explosão
):
    """
    Versão acelerada:
      - Constrói arestas dirigidas por célula (source, ra4, dec4) via explode.
      - Conta direções por par (0,1,2) -> status: missing / ok_one_way / ok_bi.
      - Gera 'pairs' completo apenas se emit_pairs=True (pode ser limitado).
    """
    # Normalizações mínimas (evita cópias desnecessárias)
    df = df_final[[source_col, "CRD_ID", "ra", "dec", "compared_to"]].copy()
    df["CRD_ID"] = df["CRD_ID"].astype(str)
    # normaliza source como string "limpa"
    df[source_col] = df[source_col].map(_norm_str).astype("string")
    # célula arredondada
    df["ra4"]  = pd.to_numeric(df["ra"], errors="coerce").round(ndp)
    df["dec4"] = pd.to_numeric(df["dec"], errors="coerce").round(ndp)

    # Filtra só células com 2+ objetos
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

    # Subset só com células 2+
    df2 = df.merge(multi_cells[[source_col, "ra4", "dec4"]].drop_duplicates(),
                   on=[source_col, "ra4", "dec4"], how="inner")

    # Explode compared_to -> arestas dirigidas dentro da célula
    df2["_cmp_list"] = df2["compared_to"].map(_parse_cmp_list_fast)
    edges = (
        df2[[source_col, "ra4", "dec4", "CRD_ID", "_cmp_list"]]
        .explode("_cmp_list", ignore_index=True)
        .rename(columns={"CRD_ID": "A", "_cmp_list": "B"})
    )
    edges = edges[edges["B"].notna() & (edges["B"] != "")]
    # Mantém apenas arestas cuja B também pertence à mesma célula
    members = df2[[source_col, "ra4", "dec4", "CRD_ID"]].rename(columns={"CRD_ID":"B"})
    edges = edges.merge(members, on=[source_col, "ra4", "dec4", "B"], how="inner")

    if edges.empty:
        # Não há nenhuma aresta dirigida -> todos os pares são "missing".
        # Calcula somente agregados sem materializar todos os pares (rápido).
        # n_pairs esperado = sum C(n,2)
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

    # Colapsa para pares não-dirigidos e conta direções (1=one_way, 2=bi)
    a_le_b = (edges["A"] <= edges["B"])
    edges["A_und"] = np.where(a_le_b, edges["A"], edges["B"])
    edges["B_und"] = np.where(a_le_b, edges["B"], edges["A"])

    pair_counts = (
        edges.groupby([source_col, "ra4", "dec4", "A_und", "B_und"], dropna=False)
             .size().rename("n_dir").reset_index()
    )
    pair_counts["status"] = np.where(pair_counts["n_dir"] >= 2, "ok_bi", "ok_one_way")

    # Agregados por célula (rápido)
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

    # Emissão opcional dos pares (inclui "missing" apenas se solicitado)
    if not emit_pairs:
        pairs = pd.DataFrame(columns=["source","ra4","dec4","A","B","A_in_B","B_in_A","status"])
        violations = pairs.copy()
    else:
        # materializa só os encontrados (rápido)
        pairs_found = pair_counts.rename(columns={"A_und":"A", "B_und":"B"})
        # A_in_B/B_in_A indicam direções presentes
        # Reconstroi flags via junção nos dirigidos
        dir1 = edges[["A","B",source_col,"ra4","dec4"]].assign(dir=1)
        dir2 = edges.rename(columns={"A":"B","B":"A"})[["A","B",source_col,"ra4","dec4"]].assign(dir=2)
        both_dir = pd.concat([dir1, dir2], ignore_index=True)
        # Marca direções
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

        # Se precisar dos "missing", materializa por célula de forma controlada
        missing_rows = []
        if limit_pairs is None or pairs_found.shape[0] < limit_pairs:
            # cria conjunto de pares encontrados por célula
            found_key = set(
                (r.source, r.ra4, r.dec4, r.A, r.B)
                for r in pairs_found.itertuples(index=False)
            )
            # gera todos os pares possíveis por célula, só quando necessário
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

    # Diagnóstico por source
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
    """Print explanations and show validator outputs (with samples by source)."""
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
TYPE_PRIORITY = {"s": 3, "g": 2, "p": 1}  # others/NA -> 0

def _type_score(t):
    """Map instrument_type_homogenized to numeric priority."""
    t = _norm_str(t)
    return TYPE_PRIORITY.get(t, 0)


def _to_num(x):
    """Best-effort numeric cast -> float; returns np.nan on failure."""
    try:
        return float(x)
    except Exception:
        return np.nan


def _z_diff(a, b):
    """Absolute z difference; returns np.inf if either is NA (undefined)."""
    if a is None or b is None:
        return np.inf
    if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return np.inf
    return abs(a - b)


def _is_na(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


# ------------------------------
# Graph building from compared_to  (EXCLUDES STARS)
# ------------------------------

def build_components(df: pd.DataFrame, excluded_ids: set[str] | None = None) -> list[tuple[str, ...]]:
    """
    Build undirected components using 'CRD_ID' and 'compared_to' comma-list.

    - Only rows with non-empty compared_to contribute edges.
    - Any edge touching IDs in `excluded_ids` (e.g., stars) is dropped.
    - Nodes in `excluded_ids` are not included in the components.

    Args:
        df: DataFrame with columns CRD_ID, compared_to.
        excluded_ids: Optional set of IDs to exclude from edges/nodes.

    Returns:
        List of sorted tuples, one per connected component.
    """
    excluded_ids = excluded_ids or set()

    edges = []
    seen_nodes = set()

    for _, row in df.iterrows():
        crd = _norm_str(row.get("CRD_ID"))
        if not crd or crd in excluded_ids:
            continue
        cmp_raw = _norm_str(row.get("compared_to"))
        if not cmp_raw:
            continue

        for nb in cmp_raw.split(","):
            nb = _norm_str(nb)
            if not nb:
                continue
            if nb in excluded_ids:
                # drop edges touching excluded nodes
                continue
            edges.append((crd, nb))
            seen_nodes.add(crd)
            seen_nodes.add(nb)

    # Disjoint-set (Union-Find)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    for n in seen_nodes:
        find(n)

    comps = defaultdict(list)
    for n in parent.keys():
        comps[find(n)].append(n)

    return [tuple(sorted(v)) for v in comps.values()]


# ------------------------------
# Pair validation rules  (GUARDS AGAINST STARS)
# ------------------------------

def validate_pair(group_df: pd.DataFrame, threshold: float) -> list[dict]:
    """
    Validate a 2-row group against the pair rules.
    Returns a list of violation dicts (empty if valid).

    New behavior:
      - If any member is a star (z_flag_homogenized==6) OR tie_result==3,
        this pair should not exist (stars are excluded from comparisons).
        A violation 'PAIR_STAR_SHOULD_NOT_BE_COMPARED' is emitted and other
        checks for this pair are skipped.
      - Stars must carry tie_result==3; otherwise 'STAR_MUST_BE_3'.
    """
    vios = []

    a, b = group_df.iloc[0], group_df.iloc[1]

    tr_a, tr_b = int(_to_num(a["tie_result"])), int(_to_num(b["tie_result"]))
    zf_a = _to_num(a["z_flag_homogenized"])
    zf_b = _to_num(b["z_flag_homogenized"])

    # Star checks
    a_is_star = (zf_a == 6) or (tr_a == 3)
    b_is_star = (zf_b == 6) or (tr_b == 3)

    if a_is_star and tr_a != 3:
        vios.append({
            "rule": "STAR_MUST_BE_3",
            "message": "Row with z_flag_homogenized==6 must have tie_result==3.",
            "group_ids": tuple(group_df["CRD_ID"].tolist()),
            "rows": group_df.copy(),
        })
    if b_is_star and tr_b != 3:
        vios.append({
            "rule": "STAR_MUST_BE_3",
            "message": "Row with z_flag_homogenized==6 must have tie_result==3.",
            "group_ids": tuple(group_df["CRD_ID"].tolist()),
            "rows": group_df.copy(),
        })

    if a_is_star or b_is_star:
        vios.append({
            "rule": "PAIR_STAR_SHOULD_NOT_BE_COMPARED",
            "message": "Stars must be excluded from comparisons; this pair should not exist.",
            "group_ids": tuple(group_df["CRD_ID"].tolist()),
            "rows": group_df.copy(),
        })
        return vios  # skip further pair logic

    # --- original pair logic (unchanged) ---
    # For comparisons, ignore flag 6 as "quality"
    zf_a_eff = (-1 if zf_a == 6 else zf_a)
    zf_b_eff = (-1 if zf_b == 6 else zf_b)

    ts_a = _type_score(a["instrument_type_homogenized"])
    ts_b = _type_score(b["instrument_type_homogenized"])

    z_a = _to_num(a["z"])
    z_b = _to_num(b["z"])
    dz = _z_diff(z_a, z_b)  # np.inf if undefined

    # Case: (1,0) or (0,1)
    if {tr_a, tr_b} == {0, 1}:
        win = a if tr_a == 1 else b
        los = b if tr_a == 1 else a

        win_zf = _to_num(win["z_flag_homogenized"])
        los_zf = _to_num(los["z_flag_homogenized"])
        win_zf_eff = (-1 if win_zf == 6 else win_zf)
        los_zf_eff = (-1 if los_zf == 6 else los_zf)

        win_ts = _type_score(win["instrument_type_homogenized"])
        los_ts = _type_score(los["instrument_type_homogenized"])

        win_z = _to_num(win["z"])
        los_z = _to_num(los["z"])
        win_los_dz = _z_diff(win_z, los_z)

        cond = (
            (win_zf_eff > los_zf_eff) or
            (win_zf_eff == los_zf_eff and win_ts > los_ts) or
            (win_zf_eff == los_zf_eff and win_ts == los_ts and win_los_dz < threshold)
        )
        if not cond:
            vios.append({
                "rule": "PAIR_1v0_PRIORITY",
                "message": "Winner does not have higher z_flag (excl. 6), nor better type, nor Δz < threshold on tie.",
                "group_ids": tuple(group_df["CRD_ID"].tolist()),
                "rows": group_df.copy(),
            })

    # Case: (2,2)
    elif tr_a == 2 and tr_b == 2:
        cond_equal_quality = (zf_a_eff == zf_b_eff) and (ts_a == ts_b)
        cond_delta = (dz > threshold) or (math.isinf(dz))
        if not (cond_equal_quality and cond_delta):
            vios.append({
                "rule": "PAIR_2v2_TIE_CONSISTENCY",
                "message": "Tie (2,2) requires equal z_flag (excl. 6), equal type, and Δz > threshold (or undefined).",
                "group_ids": tuple(group_df["CRD_ID"].tolist()),
                "rows": group_df.copy(),
            })

    # Case: (0,0)
    elif tr_a == 0 and tr_b == 0:
        # With star exclusion, (0,0) should be rare; keep original check
        if not (zf_a == 6 and zf_b == 6):
            vios.append({
                "rule": "PAIR_0v0_BOTH_FLAG6",
                "message": "Both eliminated in a pair must have z_flag_homogenized == 6.",
                "group_ids": tuple(group_df["CRD_ID"].tolist()),
                "rows": group_df.copy(),
            })

    else:
        vios.append({
            "rule": "PAIR_INVALID_TIE_PATTERN",
            "message": f"Unexpected tie_result pattern for pair: ({tr_a},{tr_b}).",
            "group_ids": tuple(group_df["CRD_ID"].tolist()),
            "rows": group_df.copy(),
        })

    return vios


# ------------------------------
# Group validation rules (size >= 3)  (GUARDS AGAINST STARS)
# ------------------------------

def validate_group(group_df: pd.DataFrame, threshold: float) -> list[dict]:
    """
    Validate a group (>=3) with hierarchical constraints.

    New behavior:
      - If any member is a star (z_flag_homogenized==6 OR tie_result==3),
        this group should not exist (stars are excluded from comparisons).
        Emit 'GROUP_STAR_SHOULD_NOT_BE_COMPARED' and skip the rest.
      - Stars must carry tie_result==3; otherwise 'STAR_MUST_BE_3'.
    """
    vios = []
    df = group_df.copy()

    # Early star checks
    zf = pd.to_numeric(df["z_flag_homogenized"], errors="coerce")
    tr = pd.to_numeric(df["tie_result"], errors="coerce").fillna(-1).astype(int)
    star_mask = (zf == 6) | (tr == 3)
    if star_mask.any():
        # Any star must have tie_result==3
        bad_star = star_mask & (tr != 3)
        if bad_star.any():
            vios.append({
                "rule": "STAR_MUST_BE_3",
                "message": "Rows with z_flag_homogenized==6 must have tie_result==3.",
                "group_ids": tuple(df["CRD_ID"].tolist()),
                "rows": df.copy(),
            })
        vios.append({
            "rule": "GROUP_STAR_SHOULD_NOT_BE_COMPARED",
            "message": "Stars must be excluded from comparisons; this group should not exist.",
            "group_ids": tuple(df["CRD_ID"].tolist()),
            "rows": df.copy(),
        })
        return vios  # skip further group logic

    # --- original group logic (unchanged beyond star guard) ---
    it = df["instrument_type_homogenized"].map(_type_score).astype("Int64")
    z = pd.to_numeric(df["z"], errors="coerce")

    # Survivors = rows with tie_result in {1,2}
    survivors_mask = tr.isin({1, 2})
    survivors_idx = df.index[survivors_mask]
    losers_idx = df.index[~survivors_mask]

    # 0) All-flag-6 special case: all must be 0 (shouldn't happen now)
    if zf.notna().all() and (zf == 6).all():
        if survivors_mask.any():
            vios.append({
                "rule": "GROUP_ALL_FLAG6_ALL_ZERO",
                "message": "All members have flag 6 but there are survivors.",
                "group_ids": tuple(df["CRD_ID"].tolist()),
                "rows": df.copy(),
            })
        return vios

    # 1) Flag dominance (ignore 6 as quality)
    zf_eff = zf.where(zf != 6, other=-1)
    max_flag = zf_eff.max(skipna=True)
    bad_flag_survivors = survivors_idx[zf_eff.loc[survivors_idx] < max_flag]
    if len(bad_flag_survivors) > 0:
        vios.append({
            "rule": "GROUP_FLAG_DOMINANCE",
            "message": "Survivors include members with lower z_flag than group max (excluding 6).",
            "group_ids": tuple(df["CRD_ID"].tolist()),
            "rows": df.copy(),
        })

    # 2) Type dominance among max-flag candidates
    max_flag_mask = (zf_eff == max_flag)
    if max_flag_mask.any():
        max_type = it[max_flag_mask].max(skipna=True)
        bad_type_survivors = survivors_idx[it.loc[survivors_idx] < max_type]
        if len(bad_type_survivors) > 0:
            vios.append({
                "rule": "GROUP_TYPE_DOMINANCE",
                "message": "Survivors include members with lower instrument_type than group max.",
                "group_ids": tuple(df["CRD_ID"].tolist()),
                "rows": df.copy(),
            })

    # 3) Δz-threshold independence among survivors at max-flag & max-type
    cand_mask = (zf_eff == max_flag) & (it == it[max_flag_mask].max(skipna=True))
    cand_idx = df.index[cand_mask]
    crd = df.loc[cand_idx, "CRD_ID"].astype(str).tolist()
    zvals = z.loc[cand_idx].to_numpy()

    survivors_crd = set(df.loc[survivors_idx, "CRD_ID"].astype(str))

    for i in range(len(crd)):
        for j in range(i + 1, len(crd)):
            zi, zj = zvals[i], zvals[j]
            if not (np.isnan(zi) or np.isnan(zj)):
                if abs(zi - zj) < threshold and (crd[i] in survivors_crd) and (crd[j] in survivors_crd):
                    vios.append({
                        "rule": "GROUP_DELTZ_INDEPENDENCE",
                        "message": "Two survivors are closer than threshold (both z defined). Survivors must form an independent set.",
                        "group_ids": tuple(df["CRD_ID"].tolist()),
                        "rows": df.copy(),
                    })
                    break

    # 4) Final labeling consistency
    n_survivors = int(survivors_mask.sum())
    if n_survivors == 0:
        if not (zf.notna().all() and (zf == 6).all()):
            vios.append({
                "rule": "GROUP_NO_SURVIVOR_SUSPECT",
                "message": "No survivors but group not all flag 6.",
                "group_ids": tuple(df["CRD_ID"].tolist()),
                "rows": df.copy(),
            })
    elif n_survivors == 1:
        lone_idx = survivors_idx[0]
        if int(df.loc[lone_idx, "tie_result"]) != 1:
            vios.append({
                "rule": "GROUP_SINGLE_SURVIVOR_MUST_BE_1",
                "message": "Exactly one survivor but not labeled with tie_result == 1.",
                "group_ids": tuple(df["CRD_ID"].tolist()),
                "rows": df.copy(),
            })
    else:
        if not (df.loc[survivors_idx, "tie_result"] == 2).all():
            vios.append({
                "rule": "GROUP_MULTI_SURVIVOR_MUST_BE_2",
                "message": "Multiple survivors but not all labeled tie_result == 2.",
                "group_ids": tuple(df["CRD_ID"].tolist()),
                "rows": df.copy(),
            })

    return vios


# ------------------------------
# Public API  (excludes stars from graph + enforces tie_result==3)
# ------------------------------
def validate_tie_results(
    df_final: pd.DataFrame,
    threshold: float = 0.0005,
    max_groups: int | None = None,
) -> dict:
    """
    Main entry point.
    - Excludes stars (z_flag_homogenized==6 or tie_result==3) from graph.
    - Builds components from 'compared_to' among NON-stars.
    - Validates pairs and groups with the declared rules.
    - Validates that all stars have tie_result==3.
    """
    df = df_final.copy()
    df["CRD_ID"] = df["CRD_ID"].astype(str)
    df["compared_to"] = df["compared_to"].astype("string")

    # Identify stars & enforce tie_result==3
    zf = pd.to_numeric(df.get("z_flag_homogenized"), errors="coerce")
    tr = pd.to_numeric(df.get("tie_result"), errors="coerce")
    star_mask = (zf == 6) | (tr == 3)
    star_ids = set(df.loc[star_mask, "CRD_ID"].astype(str))
    # Pre-collect violations for stars with wrong tie_result
    violations = []
    wrong_tr_mask = (zf == 6) & (tr != 3)
    if wrong_tr_mask.any():
        rows = df.loc[wrong_tr_mask].copy()
        violations.append({
            "rule": "STAR_MUST_BE_3",
            "message": "Rows with z_flag_homogenized==6 must have tie_result==3.",
            "group_ids": tuple(rows["CRD_ID"].tolist()),
            "rows": rows,
        })

    # Build components EXCLUDING stars
    components = build_components(df, excluded_ids=star_ids)
    if max_groups is not None:
        components = components[:max_groups]

    n_pairs = n_groups = 0

    for comp in components:
        group_df = df[df["CRD_ID"].isin(comp)].copy()
        if len(group_df) == 2:
            n_pairs += 1
            violations.extend(validate_pair(group_df, threshold))
        elif len(group_df) > 2:
            n_groups += 1
            violations.extend(validate_group(group_df, threshold))
        # singleton components are irrelevant here

    summary = {
        "n_components": len(components),
        "n_pairs": n_pairs,
        "n_groups": n_groups,
        "n_violations": len(violations),
        "by_rule": pd.Series([v["rule"] for v in violations]).value_counts().to_dict() if violations else {},
        "n_stars_excluded": int(star_mask.sum()),
    }
    return {"summary": summary, "violations": violations}


def validate_tie_results_fast(
    df_final: pd.DataFrame,
    threshold: float = 0.0005,
    max_groups: int | None = None,
    include_rows: bool = False,
):
    """
    Fast + compat com a semântica do build_components:
      - Só conta nós que aparecem em alguma aresta (como o baseline).
      - Mantém vizinhos "pendurados" (citados em compared_to mas sem linha),
        exceto se estiverem em star_ids (iguais ao baseline).
      - Componentes podem existir mesmo que virem singletons após filtrar
        por linhas presentes; ainda assim contam em n_components (como baseline).
      - Validação só roda quando há >=2 linhas PRESENTES no DF.
    """
    import numpy as np
    import math
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    df = df_final.copy()

    # colunas necessárias (se faltar, preenche com NaN)
    cols = ["CRD_ID", "compared_to", "z_flag_homogenized", "tie_result",
            "instrument_type_homogenized", "z"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df["CRD_ID"] = df["CRD_ID"].astype(str)
    df["compared_to"] = df["compared_to"].astype("string")

    # estrelas: z_flag==6 OU tie_result==3  (mesma definição)
    zf = pd.to_numeric(df["z_flag_homogenized"], errors="coerce")
    tr = pd.to_numeric(df["tie_result"], errors="coerce")
    star_mask = (zf == 6) | (tr == 3)
    star_ids = set(df.loc[star_mask, "CRD_ID"].astype(str))

    violations = []
    wrong_tr_mask = (zf == 6) & (tr != 3)
    if wrong_tr_mask.any():
        rows = df.loc[wrong_tr_mask, cols].copy() if include_rows else None
        violations.append({
            "rule": "STAR_MUST_BE_3",
            "message": "Rows with z_flag_homogenized==6 must have tie_result==3.",
            "group_ids": tuple(df.loc[wrong_tr_mask, "CRD_ID"].astype(str).tolist()),
            "rows": rows,
        })

    # Base para gerar arestas: SOMENTE nós A não-estrelas (como baseline)
    base = df.loc[~df["CRD_ID"].isin(star_ids), ["CRD_ID", "compared_to"]].copy()
    if base.empty:
        summary = {
            "n_components": 0, "n_pairs": 0, "n_groups": 0,
            "n_violations": len(violations),
            "by_rule": pd.Series([v["rule"] for v in violations]).value_counts().to_dict() if violations else {},
            "n_stars_excluded": int(star_mask.sum()),
        }
        return {"summary": summary, "violations": violations}

    # explode compared_to -> arestas dirigidas (A,B)
    cmp_df = (
        base.assign(_cmp_list=base["compared_to"].map(_parse_cmp_list_fast))
            [["CRD_ID","_cmp_list"]]
            .explode("_cmp_list", ignore_index=True)
            .rename(columns={"CRD_ID":"A", "_cmp_list":"B"})
    )
    # remove vazios
    cmp_df = cmp_df[cmp_df["B"].notna() & (cmp_df["B"] != "")]
    # baseline: descarta apenas vizinhos que estão em star_ids;
    # vizinhos ausentes no DF mas não-estrela (desconhecidos) PERMANECEM.
    cmp_df = cmp_df[~cmp_df["B"].isin(star_ids)]

    # Se não há arestas → baseline teria seen_nodes vazio → 0 componentes
    if cmp_df.empty:
        summary = {
            "n_components": 0,           # compat: NÃO conta singletons sem arestas
            "n_pairs": 0, "n_groups": 0,
            "n_violations": len(violations),
            "by_rule": pd.Series([v["rule"] for v in violations]).value_counts().to_dict() if violations else {},
            "n_stars_excluded": int(star_mask.sum()),
        }
        return {"summary": summary, "violations": violations}

    # NÓS DO GRAFO = união(A ∪ B) das arestas (como seen_nodes do baseline)
    nodes = pd.Index(pd.unique(pd.concat([cmp_df["A"], cmp_df["B"]], ignore_index=True)))
    id2ix = {cid: i for i, cid in enumerate(nodes)}
    n = nodes.size

    # Arestas não-dirigidas deduplicadas
    a_ix = cmp_df["A"].map(id2ix).to_numpy()
    b_ix = cmp_df["B"].map(id2ix).to_numpy()
    lo = np.minimum(a_ix, b_ix)
    hi = np.maximum(a_ix, b_ix)
    und = np.stack([lo, hi], axis=1)
    und = und[und[:, 0] != und[:, 1]]
    und_view = und.view([('x', und.dtype), ('y', und.dtype)])
    und = np.unique(und_view).view(und.dtype).reshape(-1, 2)

    # Grafo esparso + componentes (conta TODOS os componentes do grafo)
    data = np.ones(und.shape[0], dtype=np.int8)
    A = coo_matrix((data, (und[:, 0], und[:, 1])), shape=(n, n))
    A = A + A.T
    n_comp, labels = connected_components(A, directed=False, return_labels=True)

    # Índices dos nós PRESENTES no DF (não-estrelas + possíveis B citados que existam no DF)
    present = df.set_index("CRD_ID").reindex(nodes, copy=False)
    idx_vals = np.asarray(present.index)  # garante ndarray
    present_mask = (~pd.isna(idx_vals)) & (pd.Index(idx_vals).isin(df["CRD_ID"].astype(str).values))
    present_idx = np.flatnonzero(present_mask)  # equivale a np.where(...)[0]

    # Arrays para os presentes (para validação); nós ausentes não são usados na validação
    sub = present.iloc[present_idx][["z_flag_homogenized","tie_result","instrument_type_homogenized","z"]]
    zf_arr = pd.to_numeric(sub["z_flag_homogenized"], errors="coerce").to_numpy()
    tr_arr = pd.to_numeric(sub["tie_result"], errors="coerce").fillna(-1).astype(int).to_numpy()
    type_map = {"s": 3, "g": 2, "p": 1}
    it_arr = sub["instrument_type_homogenized"].map(lambda t: type_map.get(_norm_str(t), 0)).astype(int).to_numpy()
    z_arr = pd.to_numeric(sub["z"], errors="coerce").to_numpy()

    # mapa: global_ix -> local_ix (apenas presentes)
    gix_to_lix = {gix: lix for lix, gix in enumerate(present_idx)}

    def zdiff(a, b):
        mask = np.isnan(a) | np.isnan(b)
        out = np.abs(a - b)
        out[mask] = np.inf
        return out

    n_pairs = 0
    n_groups = 0

    # Opcionalmente limitar número de componentes analisados (como baseline faz slice)
    comp_order = np.arange(n_comp)
    if max_groups is not None and max_groups < n_comp:
        comp_order = comp_order[:max_groups]

    for cid in comp_order:
        # nós globais no componente
        gidx = np.where(labels == cid)[0]
        # filtra só nós PRESENTES no DF
        lix = [gix_to_lix[g] for g in gidx if g in gix_to_lix]
        m = len(lix)
        if m <= 1:
            continue  # nada a validar

        comp_ids_str = tuple(nodes[gidx].tolist())  # para contagem/identificação (compat)
        zf_c = zf_arr[lix]
        tr_c = tr_arr[lix]
        it_c = it_arr[lix]
        z_c  = z_arr[lix]

        if m == 2:
            n_pairs += 1
            a, b = 0, 1
            tr_a, tr_b = tr_c[a], tr_c[b]
            zf_a_eff = (-1 if zf_c[a] == 6 else zf_c[a])
            zf_b_eff = (-1 if zf_c[b] == 6 else zf_c[b])
            ts_a, ts_b = it_c[a], it_c[b]
            dz = zdiff(np.array([z_c[a]]), np.array([z_c[b]]))[0]

            vios_local = []
            if {tr_a, tr_b} == {0, 1}:
                win_is_a = (tr_a == 1)
                win_zf = zf_a_eff if win_is_a else zf_b_eff
                los_zf = zf_b_eff if win_is_a else zf_a_eff
                win_ts = ts_a if win_is_a else ts_b
                los_ts = ts_b if win_is_a else ts_a
                cond = (
                    (win_zf > los_zf) or
                    (win_zf == los_zf and win_ts > los_ts) or
                    (win_zf == los_zf and win_ts == los_ts and dz < threshold)
                )
                if not cond:
                    vios_local.append(("PAIR_1v0_PRIORITY", "Winner lacks higher flag/type/Δz<thr on tie."))
            elif tr_a == 2 and tr_b == 2:
                cond = (zf_a_eff == zf_b_eff) and (ts_a == ts_b) and (dz > threshold or math.isinf(dz))
                if not cond:
                    vios_local.append(("PAIR_2v2_TIE_CONSISTENCY", "Tie (2,2) requires equal flag/type and Δz>thr (or undefined)."))
            elif tr_a == 0 and tr_b == 0:
                vios_local.append(("PAIR_0v0_BOTH_FLAG6", "Both eliminated (0,0) but not both flag 6."))
            else:
                vios_local.append(("PAIR_INVALID_TIE_PATTERN", f"Unexpected tie_result pattern: ({tr_a},{tr_b})."))

            if vios_local:
                rows_payload = df[df["CRD_ID"].isin(sub.index[lix])].copy() if include_rows else None
                for rule, msg in vios_local:
                    violations.append({
                        "rule": rule, "message": msg,
                        "group_ids": comp_ids_str,
                        "rows": rows_payload,
                    })
        else:
            n_groups += 1
            surv = np.isin(tr_c, (1, 2))
            zf_eff = np.where(zf_c == 6, -1, zf_c)
            # nanmax pode avisar se tudo NaN; nessa situação, tratamos como "sem dominância"
            with np.errstate(all='ignore'):
                max_flag = np.nanmax(zf_eff)
            if not np.isfinite(max_flag):
                max_flag = -np.inf  # ninguém domina; evita falsos positivos

            if np.any(surv & (zf_eff < max_flag)):
                rows_payload = df[df["CRD_ID"].isin(sub.index[lix])].copy() if include_rows else None
                violations.append({
                    "rule": "GROUP_FLAG_DOMINANCE",
                    "message": "Survivors include members with lower z_flag than group max (excluding 6).",
                    "group_ids": comp_ids_str,
                    "rows": rows_payload,
                })

            cand_flag = (zf_eff == max_flag)
            if np.any(cand_flag):
                max_type = np.max(it_c[cand_flag]) if np.any(cand_flag) else -1
                if np.any(surv & (it_c < max_type)):
                    rows_payload = df[df["CRD_ID"].isin(sub.index[lix])].copy() if include_rows else None
                    violations.append({
                        "rule": "GROUP_TYPE_DOMINANCE",
                        "message": "Survivors include members with lower instrument_type than group max.",
                        "group_ids": comp_ids_str,
                        "rows": rows_payload,
                    })

            # 3) Δz independence em candidatos (max-flag & max-type)
            cand = cand_flag & (it_c == (np.max(it_c[cand_flag]) if np.any(cand_flag) else -1))
            
            # ✅ considere apenas os sobreviventes entre os candidatos
            cand_surv_idx = np.where(cand & surv)[0]
            if cand_surv_idx.size >= 2:
                zS = z_c[cand_surv_idx]
                # matriz de |Δz| apenas entre sobreviventes de topo
                D = np.abs(zS[:, None] - zS[None, :])
                iu = np.triu_indices_from(D, k=1)
                if (D[iu] < threshold).any():
                    rows_payload = df[df["CRD_ID"].isin(sub.index[lix])].copy() if include_rows else None
                    violations.append({
                        "rule": "GROUP_DELTZ_INDEPENDENCE",
                        "message": "Two survivors closer than threshold among max-flag&max-type.",
                        "group_ids": comp_ids_str,
                        "rows": rows_payload,
                    })


            n_surv = int(np.sum(surv))
            if n_surv == 0:
                rows_payload = df[df["CRD_ID"].isin(sub.index[lix])].copy() if include_rows else None
                violations.append({
                    "rule": "GROUP_NO_SURVIVOR_SUSPECT",
                    "message": "No survivors but group not all flag 6 (stars already excluded).",
                    "group_ids": comp_ids_str,
                    "rows": rows_payload,
                })
            elif n_surv == 1:
                only_idx = np.where(surv)[0][0]
                if int(tr_c[only_idx]) != 1:
                    rows_payload = df[df["CRD_ID"].isin(sub.index[lix])].copy() if include_rows else None
                    violations.append({
                        "rule": "GROUP_SINGLE_SURVIVOR_MUST_BE_1",
                        "message": "Exactly one survivor but not labeled tie_result == 1.",
                        "group_ids": comp_ids_str,
                        "rows": rows_payload,
                    })
            else:
                if not np.all(tr_c[surv] == 2):
                    rows_payload = df[df["CRD_ID"].isin(sub.index[lix])].copy() if include_rows else None
                    violations.append({
                        "rule": "GROUP_MULTI_SURVIVOR_MUST_BE_2",
                        "message": "Multiple survivors but not all labeled tie_result == 2.",
                        "group_ids": comp_ids_str,
                        "rows": rows_payload,
                    })

    summary = {
        # compat: conta TODOS os componentes do grafo (inclui aqueles que,
        # após filtrar presentes, não geraram validação)
        "n_components": int(n_comp),
        "n_pairs": n_pairs,
        "n_groups": n_groups,
        "n_violations": len(violations),
        "by_rule": pd.Series([v["rule"] for v in violations]).value_counts().to_dict() if violations else {},
        "n_stars_excluded": int(star_mask.sum()),
    }
    return {"summary": summary, "violations": violations}


def explain_tie_validation_output(
    report: dict,
    show_per_rule: int = 3,
    prefer_cols: list[str] | None = None,
):
    """
    Explain and display the results of `validate_tie_results`.

    Parameters
    ----------
    report : dict
        Output from validate_tie_results(df_final, ...).
        Expected keys: "summary" and "violations".
        Each item in "violations" is a dict with:
            - "rule": str
            - "message": str
            - "group_ids": tuple[str, ...]
            - "rows": pd.DataFrame (the group involved)
    show_per_rule : int
        Maximum number of examples to show per rule.
    prefer_cols : list[str] | None
        Preferred columns to display in each violation example.
        If None, a default set is used.
    """
    if prefer_cols is None:
        prefer_cols = [
            "CRD_ID", "ra", "dec", "z",
            "z_flag_homogenized",
            "instrument_type_homogenized",  # preferred name
            "type_homogenized",             # fallback, if it exists
            "tie_result",
            "compared_to",
        ]

    summary = report.get("summary", {}) or {}
    violations = report.get("violations", []) or []

    # ---------------------------
    # Header + Global Summary
    # ---------------------------
    display(Markdown("### Tie-results validation"))
    display(Markdown(
f"""
#### Summary
- **Analyzed components**: `{summary.get("n_components", 0)}`
- **Pairs (size=2)**: `{summary.get("n_pairs", 0)}`
- **Groups (size≥3)**: `{summary.get("n_groups", 0)}`
- **Total violations**: `{summary.get("n_violations", 0)}`
- **Stars excluded from graph** (`z_flag_homogenized==6` or `tie_result==3`): `{summary.get("n_stars_excluded", 0)}`
"""
    ))

    # ---------------------------
    # Count by Rule
    # ---------------------------
    by_rule = summary.get("by_rule", {}) or {}
    display(Markdown("#### Violations by rule"))
    if by_rule:
        by_rule_df = pd.DataFrame(
            [(k, v) for k, v in by_rule.items()],
            columns=["rule", "count"]
        ).sort_values("count", ascending=False, kind="mergesort")
        display(by_rule_df.reset_index(drop=True))
    else:
        display(Markdown("_No violations — everything consistent ✅_"))

    # If there are no violations, stop early
    if not violations:
        return

    # ---------------------------
    # Samples per Rule
    # ---------------------------
    display(Markdown("### Samples per rule"))
    # group violations by "rule"
    vio_by_rule = {}
    for v in violations:
        vio_by_rule.setdefault(v.get("rule", "UNKNOWN_RULE"), []).append(v)

    # order rules by count (same order as by_rule_df, if it exists)
    ordered_rules = list(vio_by_rule.keys())
    if by_rule:
        ordered_rules = [r for r, _ in sorted(by_rule.items(), key=lambda x: x[1], reverse=True)]

    for rule in ordered_rules:
        V = vio_by_rule[rule]
        display(Markdown(f"### Rule: `{rule}`  —  {len(V)} occurrence(s)"))
        # “typical” message of the rule (take the first one)
        msg = V[0].get("message", "")
        if msg:
            display(Markdown(f"> _{msg}_"))

        # show up to `show_per_rule` examples
        for i, viol in enumerate(V[:show_per_rule], start=1):
            group_ids = viol.get("group_ids", ())
            rows_obj = viol.get("rows", None)
            rows = rows_obj.copy() if isinstance(rows_obj, pd.DataFrame) else pd.DataFrame()

            display(Markdown(f"**Example {i}** — `group_ids`: `{group_ids}`"))

            # select columns present in `rows`, preserving preferred order
            cols_present = [c for c in prefer_cols if c in rows.columns]
            if cols_present:
                to_show = rows[cols_present].reset_index(drop=True)
            else:
                # fallback: show all if none of the preferred exist
                to_show = rows.reset_index(drop=True)

            # light sorting for readability (if columns exist)
            sort_cols = []
            if "z_flag_homogenized" in to_show.columns:
                sort_cols.append(("z_flag_homogenized", False))
            if "tie_result" in to_show.columns:
                sort_cols.append(("tie_result", False))
            if "CRD_ID" in to_show.columns:
                sort_cols.append(("CRD_ID", True))

            if sort_cols:
                by = [c for c, _ in sort_cols]
                asc = [a for _, a in sort_cols]
                to_show = to_show.sort_values(by=by, ascending=asc, kind="mergesort")

            display(to_show)

    # ---------------------------
    # Final Notes
    # ---------------------------
    display(Markdown(
"""
> **Useful Notes**
>
> - **PAIR_STAR_SHOULD_NOT_BE_COMPARED** / **GROUP_STAR_SHOULD_NOT_BE_COMPARED**  
>   indicate that a member was a star (`z_flag_homogenized==6` or `tie_result==3`) and therefore  
>   should not appear in the graph components.
>
> - **STAR_MUST_BE_3**  
>   enforces that rows with `z_flag_homogenized==6` must have `tie_result==3`.
>
> - **PAIR_1v0_PRIORITY**  
>   when there is (1,0), the winner must have higher `z_flag` (excluding 6) **or** better type  
>   (`instrument_type_homogenized`) **or** `Δz < threshold` in case of a tie.
>
> - **PAIR_2v2_TIE_CONSISTENCY**  
>   ties (2,2) require equal quality (excluding 6), equal type, and `Δz > threshold` (or undefined).
>
> - **GROUP_* rules**  
>   check flag/type dominance among survivors and `Δz` independence  
>   when multiple survive at the top.
"""
    ))

# --- Validation: when compared_to is <NA>, tie_result must be 1
# --- EXCEPTION: tie_result may be 3 ONLY if z_flag_homogenized == 6


def render_na_compared_to_validation(
    df_final: pd.DataFrame,
    show_max: int = 10,
    cols_to_show: list[str] | None = None,
    assert_if_invalid: bool = False,
):
    """
    Render a Markdown report for the rule:
      - If `compared_to` is <NA>, then `tie_result` must be 1.
      - Exception: `tie_result` may be 3 only if `z_flag_homogenized == 6`.

    Parameters
    ----------
    df_final : pd.DataFrame
        DataFrame containing at least: compared_to, tie_result, z_flag_homogenized.
    show_max : int
        Max number of violating rows to display.
    cols_to_show : list[str] | None
        Columns to display for violating rows. If None, a sensible default is used.
    assert_if_invalid : bool
        If True, raises AssertionError when violations are found.

    Returns
    -------
    dict with:
      - total_na (int), valid_count (int), invalid_count (int)
      - crosstab (pd.DataFrame)
      - violations (pd.DataFrame)
    """
    if cols_to_show is None:
        cols_to_show = [
            "CRD_ID", "ra", "dec", "z", "z_flag", "z_err",
            "z_flag_homogenized", "instrument_type", "instrument_type_homogenized",
            "tie_result", "survey", "source", "compared_to",
        ]

    # Normalize `compared_to` to true NA
    df = df_final.copy()
    df["compared_to"] = (
        df["compared_to"]
          .astype("string").str.strip()
          .replace({
              "": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NA": pd.NA,
              "<NA>": pd.NA, "None": pd.NA, "null": pd.NA
          })
    )

    # 1) Subset rows where compared_to is truly NA
    na_cmp = df[df["compared_to"].isna()].copy()

    # 2) Parse tie_result & z_flag_homogenized (keep -1 for logic)
    tie_as_int   = pd.to_numeric(na_cmp["tie_result"], errors="coerce").fillna(-1).astype(int)
    zflag_as_int = pd.to_numeric(na_cmp["z_flag_homogenized"], errors="coerce").fillna(-1).astype(int)

    # 3) Valid if:
    #    - tie_result == 1
    #    - OR tie_result == 3 and z_flag_homogenized == 6
    valid_mask = tie_as_int.eq(1) | (tie_as_int.eq(3) & zflag_as_int.eq(6))
    violations = na_cmp.loc[~valid_mask].copy()

    # 4) Summary numbers
    total_na      = int(len(na_cmp))
    valid_count   = int(valid_mask.sum())
    invalid_count = int(len(violations))

    # Header + rule description
    display(Markdown(
f"""
### Validation: `compared_to` is `<NA>` ➜ `tie_result` rule (with flag-6 exception)

**Rule**  
- If `compared_to` is `<NA>`, then **`tie_result` must be `1`**.  
- **Exception:** `tie_result` may be **`3` only if `z_flag_homogenized == 6`**.

**Summary**  
- Rows with `compared_to` `<NA>`: **{total_na}**  
- Valid: **{valid_count}**  
- **INVALID:** **{invalid_count}**
"""
    ))

    # 4b) Crosstab for display (without -1)
    tie_disp   = pd.to_numeric(na_cmp["tie_result"], errors="coerce").astype("Int64").astype("string").fillna("<NA>")
    zflag_disp = pd.to_numeric(na_cmp["z_flag_homogenized"], errors="coerce").astype("Int64").astype("string").fillna("<NA>")
    ctab = pd.crosstab(tie_disp, zflag_disp, dropna=False)

    display(Markdown("#### Crosstab: `tie_result` × `z_flag_homogenized` (display values; `<NA>` shown)"))
    display(ctab)

    # 5) Examples of violations (if any)
    if invalid_count > 0:
        cols_present = [c for c in cols_to_show if c in violations.columns]
        display(Markdown(f"#### ⚠️ Examples of violations (up to {show_max})"))
        display(violations[cols_present].head(show_max).reset_index(drop=True))
    else:
        display(Markdown("✅ All rows with `compared_to` `<NA>` satisfy the rule (either `tie_result == 1`, or `tie_result == 3` with `flag == 6`)."))

    # 6) Optional hard assertion
    if assert_if_invalid and invalid_count > 0:
        raise AssertionError(
            "Found rows where `compared_to` is <NA> but `tie_result` is invalid "
            "(not 1, nor 3 with flag==6)."
        )

    return {
        "total_na": total_na,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "crosstab": ctab,
        "violations": violations,
    }

# =======================
# =======================
# Manual Validation
# =======================
# =======================
def analyze_groups_by_compared_to(
    df_final: pd.DataFrame,
    threshold: float = 0.0005,
    max_groups: int | None = 10000,
    max_examples_per_case: int = 5,
    desired_order: list[str] | None = None,
    final_columns: list[str] | None = None,
    render: bool = True,
) -> dict:
    """
    Analyze connected groups induced by 'compared_to' (undirected), classify them into cases,
    and optionally render an explanatory Markdown report with sampled examples.

    Buckets created:
      - CASE1_* : pairs (size == 2)
      - CASE2_* : groups (size >= 3)
          *_small_* -> all pairwise Δz <= threshold
          *_large_* -> some pairwise Δz > threshold
          *_same    -> all rows share the same 'survey'
          *_diff    -> at least two distinct 'survey' values

      - TIE_FLAG_TYPE_BREAK_PAIR   : pair with equal z_flag_homogenized, different instrument_type_homogenized, different surveys
      - TIE_FLAG_TYPE_BREAK_GROUP  : group (>=3) with equal z_flag_homogenized, different instrument_type_homogenized, different surveys
      - SAME_FLAG_DIFF_TYPE        : group (>=3) with same z_flag_homogenized but at least one differing instrument_type_homogenized
      - SAME_SOURCE_PAIR           : pair with identical normalized 'source'

    Parameters
    ----------
    df_final : pd.DataFrame
        Input table with (at least) columns:
        CRD_ID, compared_to, z, survey, source, z_flag_homogenized, instrument_type_homogenized, tie_result, ra, dec...
    threshold : float
        Δz threshold used for “small/large” classification.
    max_groups : int | None
        Max number of connected components to process (None = no cap).
    max_examples_per_case : int
        How many example groups to display per bucket.
    desired_order : list[str] | None
        Column order to use for readability when showing each group.
    final_columns : list[str] | None
        Final subset of columns to display in the examples (fallback to all if None or missing).
    render : bool
        If True, display a Markdown report with summary + examples. If False, only return artifacts.

    Returns
    -------
    dict with:
      - processed_groups: int
      - groups_by_case: dict[str, list[pd.DataFrame]]
      - case_descriptions: dict[str, str]
      - summary_counts: dict[str, int]
    """
    from collections import defaultdict, deque

    # Defaults for column ordering/visibility
    if desired_order is None:
        desired_order = [
            "CRD_ID", "id", "ra", "dec", "z", "z_flag", "z_err", "instrument_type", "survey", "source",
            "z_flag_homogenized", "instrument_type_homogenized", "tie_result", "compared_to", "role",
        ]
    if final_columns is None:
        final_columns = [
            "CRD_ID", "ra", "dec", "z", "z_flag", "z_err",
            "z_flag_homogenized", "instrument_type", "instrument_type_homogenized",
            "tie_result", "survey", "source", "compared_to",
        ]

    # Keep only rows with non-empty compared_to (string-aware)
    df_cmp = df_final[(df_final["compared_to"].notnull()) & (df_final["compared_to"].astype(str).str.strip() != "")]
    n_rows_with_cmp = int(len(df_cmp))

    # ---- Build undirected adjacency from 'compared_to'
    adjacency: dict[str, set[str]] = defaultdict(set)
    for _, row in df_cmp.iterrows():
        crd_id = str(row["CRD_ID"])
        for neighbor in str(row["compared_to"]).split(","):
            nb = neighbor.strip()
            if not nb:
                continue
            adjacency[crd_id].add(nb)
            adjacency[nb].add(crd_id)

    def get_connected_group(start_id: str) -> tuple[str, ...]:
        """BFS component from start_id over adjacency."""
        visited = set()
        queue = deque([start_id])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            queue.extend(adjacency[cur] - visited)
        return tuple(sorted(visited))

    # Buckets
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
        "TIE_FLAG_TYPE_BREAK_PAIR": "pair with equal z_flag_homogenized, different instrument_type_homogenized, and different surveys",
        "TIE_FLAG_TYPE_BREAK_GROUP": "group (≥3) with equal z_flag_homogenized, different instrument_type_homogenized, and different surveys",
        "SAME_FLAG_DIFF_TYPE": "group (≥3) with same z_flag_homogenized but at least one differing instrument_type_homogenized",
        "SAME_SOURCE_PAIR": "pair with identical source (normalized, non-null)",
    }

    processed_groups = 0
    seen_components: set[tuple[str, ...]] = set()

    for _, row in df_cmp.iterrows():
        if max_groups is not None and processed_groups >= max_groups:
            break
        start_id = str(row["CRD_ID"])
        comp_ids = get_connected_group(start_id)
        if comp_ids in seen_components:
            continue
        seen_components.add(comp_ids)

        group_df = df_cmp[df_cmp["CRD_ID"].astype(str).isin(comp_ids)].copy()
        if len(group_df) < 2:
            continue

        # Mark the row used to discover the group (display-only)
        group_df["role"] = np.where(group_df["CRD_ID"].astype(str) == start_id, "principal", "compared")

        # Pairwise Δz and survey checks
        z_vals = pd.to_numeric(group_df["z"], errors="coerce").to_numpy()
        surveys = group_df["survey"].astype(str).replace({"nan": ""}).values

        delta_z_matrix = np.abs(z_vals[:, None] - z_vals[None, :])
        pairwise_dz = delta_z_matrix[np.triu_indices(len(z_vals), k=1)]
        max_delta_z = float(np.max(pairwise_dz)) if len(pairwise_dz) else 0.0
        all_pairs_below_thresh = bool(np.all(pairwise_dz <= threshold)) if len(pairwise_dz) else True
        same_survey = (len(set([s for s in surveys if s.strip() != ""])) == 1)

        # Classify
        if len(group_df) == 2:
            key = (
                "CASE1_small_same" if (max_delta_z <= threshold and same_survey) else
                "CASE1_small_diff" if (max_delta_z <= threshold) else
                "CASE1_large_same" if (same_survey) else
                "CASE1_large_diff"
            )
        else:
            key = (
                "CASE2_small_same" if (all_pairs_below_thresh and same_survey) else
                "CASE2_small_diff" if (all_pairs_below_thresh) else
                "CASE2_large_same" if (same_survey) else
                "CASE2_large_diff"
            )

        # Reorder columns for readability
        all_columns = list(group_df.columns)
        ordered_columns = (desired_order or []) + [c for c in all_columns if c not in (desired_order or [])]
        group_df = group_df.reindex(columns=ordered_columns)

        groups_by_case[key].append(group_df)

        # Extra buckets
        flags = set(group_df["z_flag_homogenized"].dropna())
        types = set(group_df["instrument_type_homogenized"].dropna())
        surveys_in_group = set(group_df["survey"].dropna())

        if len(surveys_in_group) > 1 and len(flags) == 1 and len(types) > 1:
            if len(group_df) == 2:
                groups_by_case["TIE_FLAG_TYPE_BREAK_PAIR"].append(group_df)
            elif len(group_df) > 2:
                groups_by_case["TIE_FLAG_TYPE_BREAK_GROUP"].append(group_df)

        if len(flags) == 1 and len(types) > 1 and len(group_df) > 2:
            groups_by_case["SAME_FLAG_DIFF_TYPE"].append(group_df)

        if len(group_df) == 2:
            src_valid = group_df["source"].dropna().astype(str).str.strip().str.lower()
            if len(src_valid) == 2 and len(set(src_valid)) == 1:
                groups_by_case["SAME_SOURCE_PAIR"].append(group_df)

        processed_groups += 1

    # Summary counts
    summary_counts = {k: len(v) for k, v in groups_by_case.items()}

    # Helper for diversity sampling by survey signature
    def survey_signature(df: pd.DataFrame) -> tuple[str, ...]:
        vals = df["survey"].dropna().astype(str).unique().tolist()
        return tuple(sorted(vals)) if len(vals) else ("<MISSING>",)

    # Optional rendering
    if render:
        display(Markdown(f"### Manual group analysis via `compared_to`"))
        display(Markdown(
            f"- Rows with **non-empty** `compared_to`: **{n_rows_with_cmp}**  \n"
            f"- Unique connected groups processed: **{processed_groups}**  \n"
            f"- Δz threshold: **{threshold}**"
        ))

        # Summary table (counts per case)
        if any(summary_counts.values()):
            summary_df = pd.DataFrame(
                sorted(summary_counts.items(), key=lambda x: (-x[1], x[0])),
                columns=["case", "count"]
            )
            display(Markdown("#### Groups per bucket"))
            display(summary_df.reset_index(drop=True))
        else:
            display(Markdown("_No groups found under the current filters._"))

        # Show examples per case
        for case_name, groups in groups_by_case.items():
            if not groups:
                continue
            desc = case_descriptions.get(case_name, case_name)
            display(Markdown(f"#### {case_name} — {desc}  \nFound: **{len(groups)}** group(s)"))

            # Diversity-first selection by survey signature
            seen_sigs = set()
            selection = []
            leftovers = []

            for g in groups:
                sig = survey_signature(g)
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    selection.append(g)
                else:
                    leftovers.append(g)
                if len(selection) >= max_examples_per_case:
                    break

            # Fill remaining slots if needed
            i = 0
            while len(selection) < max_examples_per_case and i < len(leftovers):
                selection.append(leftovers[i])
                i += 1

            # Display chosen examples
            for g in selection:
                to_show = g.copy()
                # Final visible columns
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