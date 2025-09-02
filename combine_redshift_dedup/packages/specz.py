# specz.py (top-level imports)
import os
import re
import ast
import glob
import json
import math
import shutil
import logging
import pathlib
import difflib
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import get_client

import lsdb
import hats
from hats_import.pipeline import ImportArguments, pipeline_with_client
from hats_import.catalog.file_readers import ParquetReader
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from combine_redshift_dedup.packages.product_handle import ProductHandle

# -----------------------
# Constants and lookups
# -----------------------
DP1_REGIONS = [
    (6.02,   -72.08, 2.5),  # 47 Tuc
    (37.86,    6.98, 2.5),  # Rubin SV 38 7
    (40.00,  -34.45, 2.5),  # Fornax dSph
    (53.13,  -28.10, 2.5),  # ECDFS
    (59.10,  -48.73, 2.5),  # EDFS
    (95.00,  -25.00, 2.5),  # Rubin SV 95 -25
    (106.23, -10.51, 2.5),  # Seagull
]

JADES_LETTER_TO_SCORE = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "E": 0.0}
VIMOS_FLAG_TO_SCORE  = {
    "LRB_X": 0.0, "MR_X": 0.0,
    "LRB_B": 1.0, "LRB_C": 1.0, "MR_C": 1.0,
    "MR_B": 3.0,
    "LRB_A": 4.0, "MR_A": 4.0
}

# -----------------------
# Logging helper
# -----------------------
def _build_logger(logs_dir: str, name: str, file_name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(pathlib.Path(logs_dir) / file_name)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    return logger

# -----------------------
# YAML mapping validation
# -----------------------
def _validate_and_rename(df: dd.DataFrame, entry: dict, logger: logging.Logger) -> dd.DataFrame:
    product_name = entry["internal_name"]
    columns_cfg = (entry.get("columns") or {})
    non_null_map = {std: src for std, src in columns_cfg.items() if src not in (None, "", "null")}
    input_cols = list(map(str, df.columns))

    missing_sources = [src for src in non_null_map.values() if src not in input_cols]
    if missing_sources:
        suggestions = {src: difflib.get_close_matches(src, input_cols, n=3, cutoff=0.6) for src in missing_sources}
        raise ValueError(
            f"[{product_name}] Missing mapped source columns in input parquet: {missing_sources}\n"
            f"Configured (non-null) mapping: {non_null_map}\n"
            f"Closest matches: {suggestions}\n"
            f"Available columns (sample): {sorted(input_cols)[:30]} ..."
        )

    # --- NEW: pre-resolve target collisions ---
    # If a target (std) already exists in df and its source (src) is different,
    # move the existing target aside to <target>__orig, <target>__orig1, ...
    for std, src in non_null_map.items():
        tgt = std
        if src != tgt and tgt in df.columns:
            # pick a non-conflicting parking name
            base = f"{tgt}__orig"
            parked = base
            i = 1
            existing = set(map(str, df.columns))
            while parked in existing:
                parked = f"{base}{i}"
                i += 1
            logger.info(
                f"{product_name} Resolving rename collision: target '{tgt}' already exists; "
                f"renaming existing '{tgt}' → '{parked}' so '{src}' can become '{tgt}'."
            )
            df = df.rename(columns={tgt: parked})

    # Build rename map: source -> standard
    col_map = {src: std for std, src in non_null_map.items()}
    if col_map:
        logger.info(f"{product_name} rename map OK (up to 6): {list(col_map.items())[:6]}")
        df = df.rename(columns=col_map)
    else:
        logger.info(f"{product_name} no non-null column mappings; proceeding without renaming.")

    # Tag source e garanta colunas padrão com dtypes anuláveis
    df["source"] = product_name
    
    base_schema = {
        "id": "string",
        "instrument_type": "string",
        "survey": "string",
        "ra": "Float64",
        "dec": "Float64",
        "z": "Float64",
        "z_flag": "Float64",
        "z_err": "Float64",
    }
    for col, pd_dtype in base_schema.items():
        if col not in df.columns:
            df = _add_missing_with_dtype(df, col, pd_dtype)
    
    return df

# -----------------------
# Type Helpers
# -----------------------

def _add_missing_with_dtype(_df: dd.DataFrame, col: str, pd_dtype: str) -> dd.DataFrame:
    """
    Cria coluna faltante com dtype anulável do pandas e meta correto.
    Preenche com <NA> (inclusive strings).
    """
    meta_added = _df._meta.assign(**{col: pd.Series(pd.array([], dtype=pd_dtype))})

    def _adder(part: pd.DataFrame) -> pd.DataFrame:
        p = part.copy()
        p[col] = pd.Series(pd.NA, index=p.index, dtype=pd_dtype)
        return p

    return _df.map_partitions(_adder, meta=meta_added)


def _normalize_string_series_to_na(s: pd.Series) -> pd.Series:
    """
    StringDtype com NA: trim e converte ''/None/'none'/'null'/'nan' -> <NA>.
    """
    s = s.astype("string").str.strip()
    low = s.fillna("").str.lower()
    mask_empty = (low == "") | low.isin(["none", "null", "nan"])
    return s.mask(mask_empty, pd.NA).astype("string")


def _to_nullable_boolean_strict(s: pd.Series) -> pd.Series:
    """
    Mantém apenas True/False; todo o resto vira <NA>; dtype 'boolean'.
    """
    if s.dtype == object or str(s.dtype).startswith("string"):
        s = s.astype("string").str.strip()
        low = s.fillna("").str.lower()
        s = s.mask((low == "") | low.isin(["none", "null", "nan"]), pd.NA)

    vals = s.astype("object")
    mask_true  = vals.apply(lambda v: isinstance(v, (bool, np.bool_)) and v is True)
    mask_false = vals.apply(lambda v: isinstance(v, (bool, np.bool_)) and v is False)

    out = pd.Series(pd.array([pd.NA] * len(vals), dtype="boolean"), index=vals.index)
    out[mask_true] = True
    out[mask_false] = False
    return out


def _normalize_schema_hints(hints: dict | None) -> dict:
    """
    Normaliza dtypes descritos no YAML para {col: 'int'|'float'|'str'|'bool'}.
    Aceita sinônimos (int64/Int64, float64/Float64/double, bool/boolean, str/string).
    Ignora entradas com dtype desconhecido (logicamente você pode logar se quiser).
    """
    if not hints:
        return {}
    norm = {}
    for col, dt in hints.items():
        if dt is None:
            continue
        k = str(col)
        v = str(dt).strip().lower()
        if v in {"int", "int64"}:
            norm[k] = "int"
        elif v in {"float", "float64", "double"}:
            norm[k] = "float"
        elif v in {"str", "string"}:
            norm[k] = "str"
        elif v in {"bool", "boolean"}:
            norm[k] = "bool"
        # else: desconhecido → ignora (ou faça um logger.warning aqui)
    return norm

def _normalize_types(df: dd.DataFrame, product_name: str, logger: logging.Logger) -> dd.DataFrame:
    """
    Normalize dtypes and do lightweight value cleaning.

    Assumes _honor_user_homogenized_mapping() has already run so that
    user-provided homogenized columns (if any) are present under the
    canonical names: 'z_flag_homogenized' and 'instrument_type_homogenized'.

    Note: 'type' is optional; failures there emit a warning (do not fail fast).
    """

    # --- 1) String-like columns (fail fast on errors) ---
    # Use pandas StringDtype ("string") for safe NA handling and consistent downstream behavior.
    string_like = ["id", "instrument_type", "survey", "instrument_type_homogenized", "source"]
    for col in string_like:
        if col in df.columns:
            try:
                df[col] = df[col].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype="string")),
                )
                if col == "survey":
                    df[col] = df[col].str.upper()
            except Exception as e:
                raise ValueError(
                    f"[{product_name}] Failed to convert column '{col}' to pandas StringDtype. "
                    f"Original error: {e}"
                ) from e

    # --- 2) Optional auxiliary 'type' (fast path support) ---
    type_cast_ok = False
    if "type" in df.columns:
        try:
            df["type"] = df["type"].map_partitions(
                _normalize_string_series_to_na,
                meta=pd.Series(pd.array([], dtype="string")),
            ).str.lower()
            type_cast_ok = True
        except Exception as e:
            logger.warning(f"⚠️ {product_name} Failed to normalize 'type' to lower-case: {e}")
            type_cast_ok = False

    # --- 3) Special-case mapping for raw 'z_flag' (JADES/VIMOS string codes -> numeric) ---
    # This only affects the raw 'z_flag', never the homogenized column.
    def _map_special_partition(partition: pd.DataFrame) -> pd.DataFrame:
        p = partition.copy()
        if "survey" not in p or "z_flag" not in p:
            return p

        # Identify surveys where string codes exist
        survey_uc = p["survey"].astype(str).str.upper()
        mask_jades = survey_uc == "JADES"
        mask_vimos = survey_uc == "VIMOS"
        if not (mask_jades.any() or mask_vimos.any()):
            return p

        # Preserve numeric entries; only map non-numeric strings
        zf_num = pd.to_numeric(p["z_flag"], errors="coerce")
        nonnum_mask = zf_num.isna()
        z_flag_str = p["z_flag"].astype(str)

        def map_jades(val):
            s = str(val).strip().upper()
            return JADES_LETTER_TO_SCORE.get(s, np.nan)

        def map_vimos(val):
            s = str(val).strip().upper()
            return VIMOS_FLAG_TO_SCORE.get(s, np.nan)

        idx_j = nonnum_mask & mask_jades
        if idx_j.any():
            z_flag_str.loc[idx_j] = z_flag_str.loc[idx_j].map(map_jades)

        idx_v = nonnum_mask & mask_vimos
        if idx_v.any():
            z_flag_str.loc[idx_v] = z_flag_str.loc[idx_v].map(map_vimos)

        mapped_numeric = pd.to_numeric(z_flag_str, errors="coerce")
        zf = zf_num.copy()
        zf[nonnum_mask] = mapped_numeric[nonnum_mask]
        p["z_flag"] = zf
        return p

    if "z_flag" in df.columns and "survey" in df.columns:
        df = df.map_partitions(_map_special_partition)

    # --- 4) Float-like columns (fail fast with detailed diagnostics) ---
    float_like = ["ra", "dec", "z", "z_err", "z_flag", "z_flag_homogenized"]
    for col in float_like:
        if col in df.columns:
            try:
                # use dd.to_numeric with errors='coerce' to detect non-numeric entries
                coerced = dd.to_numeric(df[col], errors="coerce")
    
                # IMPORTANT: use top-level dd.isna(...) instead of .notna() to avoid dask_expr issues
                invalid_mask = dd.isna(coerced) & ~dd.isna(df[col])
    
                # count & sample a few offending values
                invalid_count = dask.compute(invalid_mask.sum())[0]
                if invalid_count > 0:
                    # safer sampling: use .loc[mask] then head
                    sample_vals = df[col].loc[invalid_mask].head(5, compute=True).tolist()
                    raise ValueError(
                        f"[{product_name}] Failed to convert column '{col}' to float64: "
                        f"{invalid_count} non-numeric value(s) detected. "
                        f"Example(s): {sample_vals}. "
                        f"Tip: fix or drop non-numeric entries before running."
                    )
    
                # Safe to cast now
                df[col] = coerced.map_partitions(
                    lambda s: s.astype("Float64"),
                    meta=pd.Series(pd.array([], dtype="Float64")),
                )

            except ValueError:
                raise
            except Exception as e:
                raise ValueError(
                    f"[{product_name}] Unexpected failure converting '{col}' to float64. "
                    f"Original error: {e}"
                ) from e

    return df, type_cast_ok

# -----------------------
# CRD_ID generation
# -----------------------
def _generate_crd_ids(df: dd.DataFrame, product_name: str, temp_dir: str):
    m = re.match(r"(\d+)_", product_name)
    if not m:
        raise ValueError(f"❌ Could not extract numeric prefix from internal_name '{product_name}'")
    catalog_prefix = m.group(1)

    sizes = df.map_partitions(len).compute().tolist()
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)

    def _add_crd(part: pd.DataFrame, start: int) -> pd.DataFrame:
        p = part.copy()
        n = len(p)
        p["CRD_ID"] = [f"CRD{catalog_prefix}_{start + i + 1}" for i in range(n)]
        return p

    parts = [
        df.get_partition(i).map_partitions(
            _add_crd, offset,
            meta=df._meta.assign(CRD_ID=pd.Series(pd.array([], dtype="string")))
        )
        for i, offset in enumerate(offsets)
    ]
    df = dd.concat(parts)
    compared_to_path = os.path.join(temp_dir, f"compared_to_dict_{catalog_prefix}.json")
    return df, compared_to_path

# -----------------------
# Homogenization (wrap your current logic)
# -----------------------
def _honor_user_homogenized_mapping(
    df: dd.DataFrame,
    entry: dict,
    product_name: str,
    logger: logging.Logger,
) -> dd.DataFrame:
    """
    If YAML maps standard columns to already-homogenized sources, recreate the canonical
    homogenized columns so downstream logic can detect them.
    """

    def _ensure_single_series(df: dd.DataFrame, name: str):
        if list(df.columns).count(name) > 1:
            raise ValueError(
                f"[{product_name}] Detected duplicate '{name}' columns after YAML rename. "
                f"This typically happens when the file already had '{name}' and you also mapped "
                f"another source to '{name}'. The pipeline expects unique column names. "
                f"Tip: let the pipeline resolve collisions (see logs) or adjust the YAML."
            )
        return df[name]

    cols_cfg = (entry.get("columns") or {})
    norm = lambda x: (str(x).strip().lower() if x is not None else None)

    # z_flag <- z_flag_homogenized ?
    if norm(cols_cfg.get("z_flag")) == "z_flag_homogenized":
        if "z_flag" in df.columns:
            df["z_flag_homogenized"] = _ensure_single_series(df, "z_flag")
            logger.info(f"{product_name} YAML maps 'z_flag' <- 'z_flag_homogenized'; using user-provided homogenized z_flag.")
        else:
            logger.warning(f"⚠️ {product_name} YAML claims 'z_flag' comes from 'z_flag_homogenized', but 'z_flag' is missing after rename.")

    # instrument_type <- instrument_type_homogenized ?
    if norm(cols_cfg.get("instrument_type")) == "instrument_type_homogenized":
        if "instrument_type" in df.columns:
            df["instrument_type_homogenized"] = _ensure_single_series(df, "instrument_type")
            logger.info(f"{product_name} YAML maps 'instrument_type' <- 'instrument_type_homogenized'; using user-provided homogenized instrument_type.")
        else:
            logger.warning(f"⚠️ {product_name} YAML claims 'instrument_type' comes from 'instrument_type_homogenized', but 'instrument_type' is missing after rename.")

    return df


def _homogenize(
    df: dd.DataFrame,
    translation_config: dict,
    product_name: str,
    logger: logging.Logger,
    type_cast_ok: bool,
):
    """
    Compute homogenized columns needed for tie-breaking.

    Returns:
        tuple[
            dd.DataFrame,
            bool,                    # used_type_fastpath
            list,                    # tiebreaking_priority
            dict,                    # instrument_type_priority
            dict,                    # translation_rules_uc
        ]
    """
    tiebreaking_priority = translation_config.get("tiebreaking_priority", [])
    instrument_type_priority = translation_config.get("instrument_type_priority", {})
    translation_rules_uc = {k.upper(): v for k, v in translation_config.get("translation_rules", {}).items()}

    # -----------------------
    # Helper: row-wise translator driven by YAML rules
    # -----------------------
    def apply_translation(row: pd.Series, key: str):
        """
        - Looks up rules by upper-cased survey name.
        - Supports direct lookup and conditional 'expr' rules.
        - Replaces pd.NA in the eval context by None to avoid ambiguous boolean errors.
        - If the observed value is numeric, attempts to coerce rule keys to floats for matching.
        - Fallback:
            * for 'instrument_type' → "" (will be normalized to <NA> later if empty)
            * for 'z_flag'          → np.nan
            * otherwise uses "" for strings, np.nan for numerics
        """
        # Survey rules
        survey_raw = row.get("survey")
        survey_key = (str(survey_raw).upper() if pd.notna(survey_raw) else "")
        ruleset = translation_rules_uc.get(survey_key, {}) or {}
        rule = ruleset.get(f"{key}_translation", {}) or {}
        if not isinstance(rule, dict):
            rule = {}

        # Observed value + default type
        val = row.get(key)
        if key == "instrument_type":
            default_missing = ""
        elif key == "z_flag":
            default_missing = np.nan
        else:
            default_missing = "" if isinstance(val, str) else np.nan

        # Normalize observed value
        if pd.isna(val):
            val_norm = None
        elif isinstance(val, (int, float, np.number)) and not isinstance(val, (bool, np.bool_)):
            val_norm = float(val)
        else:
            val_norm = str(val).strip().lower()

        # Normalize rule keys for matching
        normalized_rule = {}
        for rk, rv in rule.items():
            if rk in {"conditions", "default"}:
                continue
            if isinstance(val_norm, (int, float)):
                try:
                    nk = float(rk)
                except Exception:
                    nk = rk  # non-numeric rule key won't match numeric val_norm
            else:
                nk = str(rk).strip().lower()
            normalized_rule[nk] = rv

        # Direct lookup
        if val_norm in normalized_rule:
            return normalized_rule[val_norm]

        # Conditional expressions (safe context: pd.NA -> None)
        if "conditions" in rule:
            context = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        
            # Whitelisted builtins only
            safe_globals = {
                "__builtins__": {},
                "len": len,
                "int": int,
                "float": float,
                "str": str,
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "math": math,
                "np": np,
            }
        
            try:
                for cond in rule["conditions"]:
                    expr = cond.get("expr")
                    if expr and eval(expr, safe_globals, context):
                        return cond.get("value", default_missing)
            except Exception as e:
                raise ValueError(
                    f"Error evaluating condition '{expr}' for survey '{survey_key}': {e}"
                )

        # Fallback
        return rule.get("default", default_missing)

    # -----------------------
    # z_flag_homogenized
    # -----------------------
    def can_use_zflag_as_quality() -> bool:
        """Decide if z_flag looks like a [0,1] quality score with fractional values present."""
        if "z_flag" not in df.columns:
            return False
        try:
            cnt = df["z_flag"].count()
            minv = df["z_flag"].min()
            maxv = df["z_flag"].max()
            frac_cnt = df[(df["z_flag"] > 0.0) & (df["z_flag"] < 1.0)]["z_flag"].count()
            cnt, minv, maxv, frac_cnt = dask.compute(cnt, minv, maxv, frac_cnt)
            if cnt == 0 or not np.isfinite(minv) or not np.isfinite(maxv):
                return False
            if not (minv >= 0.0 and maxv <= 1.0):
                return False
            return frac_cnt > 0
        except Exception as e:
            logger.warning(f"⚠️ {product_name} Could not validate 'z_flag' as quality-like: {e}")
            return False

    def quality_like_to_flag(x):
        """Map a [0,1] quality score to VVDS-like flags {0..4}."""
        if pd.isna(x):
            return np.nan
        x = float(x)
        if x == 0.0:
            return 0.0
        if 0.0 < x < 0.7:
            return 1.0
        if 0.7 <= x < 0.9:
            return 2.0
        if 0.9 <= x < 0.99:
            return 3.0
        if 0.99 <= x <= 1.0:
            return 4.0
        return np.nan

    if "z_flag_homogenized" in tiebreaking_priority:
        if "z_flag_homogenized" not in df.columns:
            if can_use_zflag_as_quality():
                logger.info(f"{product_name} Using 'z_flag' (quality-like) fast path for z_flag_homogenized.")
                df["z_flag_homogenized"] = df["z_flag"].map_partitions(
                    lambda s: s.apply(quality_like_to_flag).astype("Float64"),
                    meta=pd.Series(pd.array([], dtype="Float64")),
                )
            else:
                logger.info(f"{product_name} z_flag not quality-like; using YAML translation for z_flag_homogenized.")
                df["z_flag_homogenized"] = df.map_partitions(
                    lambda p: p.apply(lambda row: apply_translation(row, "z_flag"), axis=1).astype("Float64"),
                    meta=pd.Series(pd.array([], dtype="Float64")),
                )
        else:
            logger.warning(f"{product_name} Column 'z_flag_homogenized' already exists. Skipping recompute.")

    # -----------------------
    # instrument_type_homogenized
    # -----------------------
    used_type_fastpath = False

    def can_use_type_for_instrument() -> bool:
        # Respect normalization outcome from _normalize_types (type is string-lowered with NA)
        if not type_cast_ok or "type" not in df.columns:
            return False
        try:
            allowed = {"s", "g", "p"}
            uniques = df["type"].dropna().unique().compute().tolist()
            uniques = [u for u in uniques if isinstance(u, (str, np.str_)) and u != ""]
            if len(uniques) == 0:
                return False
            return all(u in allowed for u in uniques)
        except Exception as e:
            logger.warning(f"⚠️ {product_name} Could not validate 'type' values: {e}")
            return False

    if "instrument_type_homogenized" in tiebreaking_priority:
        if "instrument_type_homogenized" not in df.columns:
            if can_use_type_for_instrument():
                logger.info(f"{product_name} Using 'type' fast path for instrument_type_homogenized.")
                # Ensure StringDtype with NA and lower-cased values
                df["instrument_type_homogenized"] = df["type"].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype="string")),
                ).str.lower()
                used_type_fastpath = True
            else:
                logger.info(f"{product_name} 'type' not suitable; using YAML translation for instrument_type_homogenized.")
                df["instrument_type_homogenized"] = df.map_partitions(
                    lambda p: p.apply(lambda row: apply_translation(row, "instrument_type"), axis=1),
                    meta=pd.Series(pd.array([], dtype="string")),
                )
                # Normalize to StringDtype with NA and lower-case
                df["instrument_type_homogenized"] = df["instrument_type_homogenized"].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype="string")),
                ).str.lower()
        else:
            logger.warning(f"{product_name} Column 'instrument_type_homogenized' already exists. Skipping recompute.")

    return df, used_type_fastpath, tiebreaking_priority, instrument_type_priority, translation_rules_uc

# -----------------------
# Tie-breaking / duplicates
# -----------------------
def _apply_tiebreaking_and_collect(
    df: dd.DataFrame,
    translation_config: dict,
    tiebreaking_priority: list,
    instrument_type_priority: dict,
    compared_to_dict_solo: dict,
    product_name: str,
    logger: logging.Logger,
) -> dd.DataFrame:
    """
    Group by RA/DEC (rounded to 6 decimals), apply tie-breaking rules, and update
    `tie_result` per CRD_ID. Also populates `compared_to_dict_solo` with pairwise links
    for rows that collided at the same sky position.
    """
    # --- config ---
    delta_z_threshold = translation_config.get("delta_z_threshold", 0.0)

    # --- validate tiebreaking configuration ---
    if not tiebreaking_priority:
        logger.warning(
            f"⚠️ {product_name} tiebreaking_priority is empty. Will rely on delta_z_threshold only."
        )
    else:
        for col in tiebreaking_priority:
            if col not in df.columns:
                raise ValueError(
                    f"Tiebreaking column '{col}' is missing in catalog '{product_name}'."
                )

            # Determine if the column is effectively empty (all NA/empty for string dtypes).
            # For numeric dtypes we only check NA.
            col_dtype = df[col].dtype
            is_str = pd.api.types.is_string_dtype(col_dtype)
            empty_mask = df[col].isna()
            if is_str:
                empty_mask = empty_mask | (df[col] == "")
            if empty_mask.all().compute():
                raise ValueError(
                    f"Tiebreaking column '{col}' is invalid in catalog '{product_name}' (all values are NaN/empty)."
                )

            # Except for 'instrument_type_homogenized', all tiebreak columns must be numeric.
            if col != "instrument_type_homogenized":
                # Use Pandas dtype check to handle nullable extension dtypes (Float64/Int64)
                if not pd.api.types.is_numeric_dtype(col_dtype):
                    try:
                        df[col] = dd.to_numeric(df[col], errors="coerce").astype("float64")
                        logger.info(f"ℹ️ {product_name} cast '{col}' to float for tie-breaking.")
                    except Exception as e:
                        raise ValueError(
                            f"Tiebreaking column '{col}' must be numeric (except 'instrument_type_homogenized'). "
                            f"Attempted cast to float but failed. Error: {e}"
                        )

    if not tiebreaking_priority and (delta_z_threshold is None or delta_z_threshold == 0.0):
        raise ValueError(
            f"Cannot deduplicate catalog '{product_name}': tiebreaking_priority is empty and "
            f"delta_z_threshold is not set or is zero. Please provide at least one criterion."
        )

    # --- find RA/DEC collisions (rounded grid to 1e-6 deg) ---
    valid_coord_mask = df["ra"].notnull() & df["dec"].notnull()
    df_valid = df[valid_coord_mask]
    df_valid["ra_dec_key"] = (
        df_valid["ra"].round(6).astype(str) + "_" +
        df_valid["dec"].round(6).astype(str)
    )

    dup_counts = df_valid.groupby("ra_dec_key").size().compute()
    keys_dup = dup_counts[dup_counts > 1].index.tolist()
    tie_updates = []

    if keys_dup:
        # Pull only the duplicate rows to Pandas for local decision making
        df_dup_local = df_valid[df_valid["ra_dec_key"].isin(keys_dup)].compute()

        for key in keys_dup:
            group = df_dup_local[df_dup_local["ra_dec_key"] == key].copy()
            group["tie_result"] = 0  # start eliminated
            surviving = group.copy()

            # Apply columns in priority order
            for priority_col in tiebreaking_priority:
                if priority_col == "instrument_type_homogenized":
                    # Map categorical to numeric priority
                    surviving["_priority_value"] = (
                        surviving["instrument_type_homogenized"]
                        .map(instrument_type_priority)
                        .fillna(-np.inf)
                    )
                else:
                    surviving["_priority_value"] = surviving[priority_col].fillna(-np.inf)

                # Special rule: if z_flag_homogenized marks stars (== 6), eliminate them
                if priority_col == "z_flag_homogenized":
                    ids_to_eliminate = surviving.loc[
                        surviving["z_flag_homogenized"] == 6, "CRD_ID"
                    ].tolist()
                    if ids_to_eliminate:
                        group.loc[group["CRD_ID"].isin(ids_to_eliminate), "tie_result"] = 0
                        surviving = surviving[surviving["z_flag_homogenized"] != 6]

                if surviving.empty:
                    break

                # Keep only the best score for this priority
                max_val = surviving["_priority_value"].max()
                surviving = surviving[surviving["_priority_value"] == max_val]
                surviving = surviving.drop(columns=["_priority_value"], errors="ignore")

                if len(surviving) == 1:
                    break  # single survivor found

            # Mark current survivors as ties (to be resolved below)
            group.loc[group["CRD_ID"].isin(surviving["CRD_ID"]), "tie_result"] = 2

            # Delta-z disambiguation among survivors
            if len(surviving) > 1 and (delta_z_threshold or 0) > 0:
                # Convert z to float NumPy array (coerce invalids to NaN) to avoid pd.NA ambiguity
                z_vals = pd.to_numeric(surviving["z"], errors="coerce").to_numpy(dtype=float, copy=False)
                ids = surviving["CRD_ID"].astype(str).to_numpy(copy=False)

                remaining_ids = set(ids)

                for i in range(len(ids)):
                    if ids[i] not in remaining_ids:
                        continue
                    for j in range(i + 1, len(ids)):
                        if ids[j] not in remaining_ids:
                            continue
                        zi, zj = z_vals[i], z_vals[j]
                        # Skip pairs with non-finite z (NaN/inf)
                        if not (np.isfinite(zi) and np.isfinite(zj)):
                            continue
                        if abs(zi - zj) <= float(delta_z_threshold):
                            # Prefer i, eliminate j
                            group.loc[group["CRD_ID"] == ids[i], "tie_result"] = 2
                            group.loc[group["CRD_ID"] == ids[j], "tie_result"] = 0
                            remaining_ids.discard(ids[j])

            # Final clean-up: convert 2->1 if a single survivor remains
            survivors = group[group["tie_result"] == 2]
            if len(survivors) == 1:
                group.loc[group["tie_result"] == 2, "tie_result"] = 1
            elif len(survivors) == 0:
                non_elim = group[group["tie_result"] != 0]
                if len(non_elim) == 1:
                    group.loc[group["CRD_ID"] == non_elim.iloc[0]["CRD_ID"], "tie_result"] = 1

            # Record updates to apply back on Dask df
            tie_updates.append(group[["CRD_ID", "tie_result"]].copy())

            # Populate compared_to_dict_solo with all pairwise links inside this collision group
            ids_all = group["CRD_ID"].tolist()
            for a in range(len(ids_all)):
                for b in range(a + 1, len(ids_all)):
                    i, j = ids_all[a], ids_all[b]
                    compared_to_dict_solo[i].add(j)
                    compared_to_dict_solo[j].add(i)

    # Merge tie updates (if any) back to Dask
    if tie_updates:
        combined_update = pd.concat(tie_updates, ignore_index=True)
        tie_update_dd = dd.from_pandas(combined_update, npartitions=1)

        # Ensure consistent string types
        df["CRD_ID"] = df["CRD_ID"].astype("string")
        tie_update_dd["CRD_ID"] = tie_update_dd["CRD_ID"].astype("string")

        # Replace tie_result with updated values; default to 1 if not present
        df = (
            df.drop("tie_result", axis=1, errors="ignore")
              .merge(tie_update_dd, on="CRD_ID", how="left")
              .fillna({"tie_result": 1})
        )

        df["tie_result"] = df["tie_result"].astype("Int8")

    return df

# -----------------------
# RA/DEC strict validation (fail fast)
# -----------------------
def _validate_ra_dec_or_fail(df: dd.DataFrame, product_name: str):
    def _isfinite_series(s: pd.Series) -> pd.Series:
        arr = s.astype("float64")  # Float64 -> float64 (NaN onde <NA>)
        return pd.Series(np.isfinite(arr), index=s.index)

    isfinite_ra  = df["ra"].map_partitions(_isfinite_series,  meta=("ra", "bool"))
    isfinite_dec = df["dec"].map_partitions(_isfinite_series, meta=("dec", "bool"))

    ra64  = df["ra"].astype("float64")
    dec64 = df["dec"].astype("float64")

    in_range = (
        (ra64 >= 0.0) & (ra64 < 360.0) &
        (dec64 >= -90.0) & (dec64 <= 90.0)
    )
    invalid_mask = ~(isfinite_ra & isfinite_dec & in_range)

    na_ra, na_dec = df["ra"].isna().sum(), df["dec"].isna().sum()
    nonfinite_ra, nonfinite_dec = (~isfinite_ra).sum(), (~isfinite_dec).sum()
    oor_ra_low,  oor_ra_high  = (ra64 < 0.0).sum(), (ra64 >= 360.0).sum()
    oor_dec_low, oor_dec_high = (dec64 < -90.0).sum(), (dec64 > 90.0).sum()
    invalid_total = invalid_mask.sum()

    (na_ra, na_dec, nonfinite_ra, nonfinite_dec,
     oor_ra_low, oor_ra_high, oor_dec_low, oor_dec_high, invalid_total) = dask.compute(
        na_ra, na_dec, nonfinite_ra, nonfinite_dec,
        oor_ra_low, oor_ra_high, oor_dec_low, oor_dec_high, invalid_total
    )

    if invalid_total > 0:
        cols = [c for c in ["CRD_ID", "id", "source", "survey", "ra", "dec"] if c in df.columns]
        sample_records = df[invalid_mask][cols].head(5, compute=True).to_dict(orient="records")
        raise ValueError(
            f"[{product_name}] Found invalid RA/DEC rows before DP1 flagging: {invalid_total}\n"
            f"  - RA NaN: {na_ra}, DEC NaN: {na_dec}\n"
            f"  - RA non-finite (±inf): {nonfinite_ra}, DEC non-finite (±inf): {nonfinite_dec}\n"
            f"  - RA out-of-range (<0): {oor_ra_low}, RA out-of-range (>=360): {oor_ra_high}\n"
            f"  - DEC out-of-range (<-90): {oor_dec_low}, DEC out-of-range (>90): {oor_dec_high}\n"
            f"  - Sample of bad rows (up to 5): {sample_records}"
        )

# -----------------------
# DP1 flagging
# -----------------------
def _flag_dp1(df: dd.DataFrame) -> dd.DataFrame:
    ra_centers  = np.deg2rad([r[0] for r in DP1_REGIONS])
    dec_centers = np.deg2rad([r[1] for r in DP1_REGIONS])
    radii       = [r[2] for r in DP1_REGIONS]

    def _compute(part: pd.DataFrame) -> pd.DataFrame:
        p = part.copy()
        ra_rad  = np.deg2rad(p["ra"].to_numpy(dtype=float, copy=False))
        dec_rad = np.deg2rad(p["dec"].to_numpy(dtype=float, copy=False))
        in_any = np.zeros(len(p), dtype=bool)
        for ra_c, dec_c, rdeg in zip(ra_centers, dec_centers, radii):
            cos_ang = (np.sin(dec_c) * np.sin(dec_rad) +
                       np.cos(dec_c) * np.cos(dec_rad) * np.cos(ra_rad - ra_c))
            ang_deg = np.rad2deg(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
            in_any |= (ang_deg <= rdeg)
        p["is_in_DP1_fields"] = in_any.astype(np.int64)
        return p

    return df.map_partitions(_compute, meta=df._meta.assign(is_in_DP1_fields=np.int64()))

# -----------------------
# Column selection
# -----------------------
def _extract_variables_from_expr(expr: str):
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return set()

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.vars = set()
        def visit_Name(self, node):
            self.vars.add(node.id)
            self.generic_visit(node)

    v = _Visitor()
    v.visit(tree)
    return v.vars

def _select_output_columns(
    df: dd.DataFrame,
    translation_rules_uc: dict,
    tiebreaking_priority: list,
    used_type_fastpath: bool,
    save_expr_columns: bool = False,
    schema_hints: dict | None = None,
) -> dd.DataFrame:
    final_cols = [
        "CRD_ID", "id", "ra", "dec", "z", "z_flag", "z_err",
        "instrument_type", "survey", "source", "tie_result", "is_in_DP1_fields"
    ]
    if "z_flag_homogenized" in df.columns:
        final_cols.append("z_flag_homogenized")
    if "instrument_type_homogenized" in df.columns:
        final_cols.append("instrument_type_homogenized")

    # Se usamos o fastpath baseado em 'type', mantenha dtype string com NA
    if used_type_fastpath:
        df["instrument_type"] = df["type"].astype("string")

    # Quais colunas de tie-breaking também devem ir na saída
    extra = [c for c in tiebreaking_priority if c not in final_cols and c in df.columns]
    final_cols += extra

    # Expr columns só se habilitado
    extra_expr_cols = set()
    if save_expr_columns:
        for ruleset in translation_rules_uc.values():
            for key in ["z_flag_translation", "instrument_type_translation"]:
                rule = ruleset.get(key, {})
                for cond in rule.get("conditions", []):
                    expr = cond.get("expr", "")
                    vars_in_expr = _extract_variables_from_expr(expr)
                    extra_expr_cols.update({v for v in vars_in_expr if v in df.columns})

    standard = {"id", "ra", "dec", "z", "z_flag", "z_err", "instrument_type", "survey"}
    already = set(final_cols)

    # Precisamos delas (mesmo se não existirem no df; serão criadas como <NA> com dtype correto)
    needed = [c for c in extra_expr_cols if c not in standard and c not in already]
    if save_expr_columns:
        final_cols += needed

    # Hints vindos do YAML (normalizados antes via _normalize_schema_hints)
    schema_hints = schema_hints or {}
    pandas_dtype = {"int": "Int64", "float": "Float64", "str": "string", "bool": "boolean"}

    if save_expr_columns and schema_hints:
        # Só normalizamos/coagimos as colunas de expr que vamos salvar e que têm hint
        target_cols = [c for c in needed if c in schema_hints]
        for col in target_cols:
            kind = schema_hints[col]
            pd_dtype = pandas_dtype[kind]

            if col in df.columns:
                # Normalizar/forçar dtype conforme o hint
                if kind == "str":
                    df[col] = df[col].map_partitions(
                        _normalize_string_series_to_na,
                        meta=pd.Series(pd.array([], dtype="string")),
                    )
                elif kind == "float":
                    coerced = dd.to_numeric(df[col], errors="coerce")
                    df[col] = coerced.map_partitions(
                        lambda s: s.astype("Float64"),
                        meta=pd.Series(pd.array([], dtype="Float64")),
                    )
                elif kind == "int":
                    coerced = dd.to_numeric(df[col], errors="coerce")
                    df[col] = coerced.map_partitions(
                        lambda s: s.astype("Int64"),
                        meta=pd.Series(pd.array([], dtype="Int64")),
                    )
                elif kind == "bool":
                    df[col] = df[col].map_partitions(
                        _to_nullable_boolean_strict,
                        meta=pd.Series(pd.array([], dtype="boolean")),
                    )
            else:
                # Coluna não existe → cria com <NA> e dtype anulável correto
                df = _add_missing_with_dtype(df, col, pd_dtype)

    # Ordenação final e subsetting
    final_cols = list(dict.fromkeys(final_cols))
    df = df[[c for c in final_cols if c in df.columns]]
    return df

# -----------------------
# Save parquet + HATS/margin
# -----------------------
def _save_parquet(df: dd.DataFrame, temp_dir: str, product_name: str) -> str:
    out_path = os.path.join(temp_dir, f"prepared_{product_name}")
    df.to_parquet(out_path, write_index=False, engine="pyarrow")
    return out_path

def import_catalog(path, ra_col, dec_col, artifact_name, output_path, logs_dir, logger, client, size_threshold_mb=500):
    """
    Import a Parquet catalog into HATS format.
    Uses lightweight method for small catalogs (< size_threshold_mb).
    
    Args:
        path (str): Path to input Parquet files or directory.
        ra_col (str): RA column name.
        dec_col (str): DEC column name.
        artifact_name (str): Output HATS catalog name.
        output_path (str): Directory to save HATS output.
        client (Client): Dask client for large catalog import.
        size_threshold_mb (int): Threshold to decide method (in MB).
    """
    
    # Detect parquet file paths
    base_path = os.path.join(path, "base")
    if os.path.exists(base_path):
        parquet_files = glob.glob(os.path.join(base_path, "*.parquet"))
    else:
        parquet_files = glob.glob(os.path.join(path, "*.parquet"))

    if not parquet_files:
        raise ValueError(f"No Parquet files found at {path}")

    # Compute total size
    total_size_mb = sum(os.path.getsize(f) for f in parquet_files) / 1024**2

    hats_path = os.path.join(output_path, artifact_name)

    if total_size_mb <= size_threshold_mb:
        logger.info(f"⚡ Small catalog detected ({total_size_mb:.1f} MB). Using direct `to_hats()` method.")
        df = pd.read_parquet(parquet_files)
        catalog = lsdb.from_dataframe(
            df,
            catalog_name=artifact_name,
            ra_column=ra_col,
            dec_column=dec_col
        )
        catalog.to_hats(hats_path, overwrite=True)
        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: import_catalog id={artifact_name}")
    else:
        logger.info(f"🧱 Large catalog detected ({total_size_mb:.1f} MB). Using distributed HATS import.")
        file_reader = ParquetReader()
        args = ImportArguments(
            ra_column=ra_col,
            dec_column=dec_col,
            input_file_list=parquet_files,
            file_reader=file_reader,
            output_artifact_name=artifact_name,
            output_path=output_path,
        )
        pipeline_with_client(args, client)
        logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: import_catalog id={artifact_name}")

def generate_margin_cache_safe(hats_path, output_path, artifact_name, logs_dir, logger, client):
    """
    Generate margin cache if partitions > 1; otherwise, skip gracefully.
    If resume fails due to missing critical files, clean and regenerate.
    
    Args:
        hats_path (str): Path to HATS catalog.
        output_path (str): Path to save margin cache.
        artifact_name (str): Name of margin cache artifact.
        client (Client): Dask client.

    Returns:
        str or None: Path to margin cache or None if skipped.
    """

    margin_dir = os.path.join(output_path, artifact_name)
    intermediate_dir = os.path.join(margin_dir, "intermediate")
    critical_file = os.path.join(intermediate_dir, "margin_pair.csv")

    # Check for broken resumption state
    if os.path.exists(intermediate_dir) and not os.path.exists(critical_file):
        logger.info(f"⚠️ Detected incomplete margin cache at {margin_dir}. Deleting to force regeneration...")
        try:
            shutil.rmtree(margin_dir)
        except Exception as e:
            logger.info(f"❌ Failed to delete corrupted margin cache at {margin_dir}: {e}")
            raise

    try:
        catalog = hats.read_hats(hats_path)
        info = catalog.partition_info.as_dataframe().astype(int)
        if len(info) > 1:
            args = MarginCacheArguments(
                input_catalog_path=hats_path,
                output_path=output_path,
                margin_threshold=1.0,
                output_artifact_name=artifact_name
            )
            pipeline_with_client(args, client)
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: generate_margin_cache id={artifact_name}")
            return margin_dir
        else:
            logger.info(f"⚠️ Margin cache skipped: single partition for {artifact_name}")
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: generate_margin_cache id={artifact_name}")
            return None
    except ValueError as e:
        if "Margin cache contains no rows" in str(e):
            logger.info(f"⚠️ {e} Proceeding without margin cache.")
            logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: generate_margin_cache id={artifact_name}")
            return None
        else:
            raise

def _maybe_hats_and_margin(out_path: str, product_name: str, logs_dir: str, temp_dir: str,
                           logger: logging.Logger, client, combine_mode: str):
    hats_path, margin_cache_path = "", ""
    if combine_mode != "concatenate":
        if client is None:
            raise ValueError(
                f"❌ Dask client is required for combine_mode='{combine_mode}' "
                f"(needed to run import_catalog and generate_margin_cache)."
            )
        m = re.match(r"(\d+)_", product_name)
        if not m:
            raise ValueError(f"❌ Could not extract numeric prefix from internal_name '{product_name}'")
        prefix = m.group(1)
        art_hats, art_margin = f"cat{prefix}_hats", f"cat{prefix}_margin"

        import_catalog(out_path, "ra", "dec", art_hats, temp_dir, logs_dir, logger, client)
        generate_margin_cache_safe(os.path.join(temp_dir, art_hats), temp_dir, art_margin, logs_dir, logger, client)

        hats_path = os.path.join(temp_dir, art_hats)
        margin_cache_path = os.path.join(temp_dir, art_margin)
    return hats_path, margin_cache_path

# =======================
# Main orchestrator
# =======================
def prepare_catalog(entry, translation_config, logs_dir, temp_dir, combine_mode="concatenate_and_mark_duplicates"):
    client = get_client()

    product_name = entry["internal_name"]
    logger = _build_logger(logs_dir, "prepare_catalog_logger", "prepare_all.log")
    logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Starting: prepare_catalog id={product_name}")

    # Load
    ph = ProductHandle(entry["path"])
    df = ph.to_ddf()

    # Validate/rename
    df = _validate_and_rename(df, entry, logger)
    
    # >>> HONOR already homogenized columns coming via YAML <<<
    df = _honor_user_homogenized_mapping(df, entry, product_name, logger)
    
    # Type normalization (includes coercion of *_homogenized dtypes, if any)
    df, type_cast_ok = _normalize_types(df, product_name, logger)

    # IDs + compared_to
    df, compared_to_path = _generate_crd_ids(df, product_name, temp_dir)
    df["tie_result"] = 1
    compared_to_dict_solo = defaultdict(set)

    # Homogenization (your existing logic wrapped)
    df, used_type_fastpath, tiebreaking_priority, instrument_type_priority, translation_rules_uc = _homogenize(
        df, translation_config, product_name, logger, type_cast_ok=type_cast_ok
    )

    # Tie-breaking / duplicates (only in mark/remove modes)
    if combine_mode in ["concatenate_and_mark_duplicates", "concatenate_and_remove_duplicates"]:
        df = _apply_tiebreaking_and_collect(
            df, translation_config, tiebreaking_priority, instrument_type_priority,
            compared_to_dict_solo, product_name, logger
        )

    # Save compared_to (even if empty)
    with open(compared_to_path, "w") as f:
        json.dump({k: sorted(v) for k, v in compared_to_dict_solo.items()}, f)

    # Strict validation on RA/DEC to catch bad inputs early
    _validate_ra_dec_or_fail(df, product_name)

    # DP1 flag
    df = _flag_dp1(df)

    # Column selection and save
    df = _select_output_columns(
        df,
        translation_rules_uc,
        tiebreaking_priority,
        used_type_fastpath,
        save_expr_columns=translation_config.get("save_expr_columns", False),
        schema_hints=_normalize_schema_hints(translation_config.get("expr_column_schema")),
    )
    out_path = _save_parquet(df, temp_dir, product_name)

    # HATS + margin cache (optional)
    hats_path, margin_cache_path = _maybe_hats_and_margin(out_path, product_name, logs_dir, temp_dir, logger, client, combine_mode)

    logger.info(f"{datetime.now():%Y-%m-%d-%H:%M:%S.%f}: Finished: prepare_catalog id={product_name}")
    return hats_path, "ra", "dec", product_name, margin_cache_path, compared_to_path