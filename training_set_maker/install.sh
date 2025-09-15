#!/bin/bash --login
set -Eeuo pipefail

# ----------------------------
# Helpers
# ----------------------------
log(){ echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*"; }

# ---------- Conda initialization with fallbacks ----------
init_conda() {
  # 1) Fast path: conda on PATH + normal base
  if command -v conda >/dev/null 2>&1; then
    if source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null; then
      log "Conda initialized via conda info --base."
      return 0
    else
      log "Fallback: conda.sh not found under \$(conda info --base)."
    fi
  else
    log "conda not found in PATH."
  fi

  # 2) System-wide fallback
  if [ -f /etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    if source /etc/profile.d/conda.sh 2>/dev/null; then
      log "Conda initialized from /etc/profile.d/conda.sh."
      return 0
    fi
  fi

  # 3) From CONDA_EXE if present
  if [ -n "${CONDA_EXE:-}" ]; then
    local _base
    _base="$(dirname "$(dirname "$CONDA_EXE")")"
    if [ -f "$_base/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      if source "$_base/etc/profile.d/conda.sh" 2>/dev/null; then
        log "Conda initialized via CONDA_EXE at $_base."
        return 0
      fi
    fi
  fi

  # 4) Cluster-specific fallbacks: $SCRATCH/miniconda and $SCRATCH/miniconda3
  local _scratch="${SCRATCH:-}"
  if [ -n "$_scratch" ]; then
    local candidates=(
      "$_scratch/miniconda"
      "$_scratch/miniconda3"
    )
    for cand in "${candidates[@]}"; do
      if [ -f "$cand/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        if source "$cand/etc/profile.d/conda.sh" 2>/dev/null; then
          log "Conda initialized from $cand."
          return 0
        fi
      fi
    done
  else
    log "SCRATCH is not set; skipping $SCRATCH/miniconda* fallbacks."
  fi

  # If we reach here, we failed
  log "Failed to source conda.sh from all known locations."
  return 1
}

# Call it and stop on failure
init_conda || { echo "Failed to source conda.sh"; exit 1; }
# --------------------------------------------------------

env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

# Normalize package names to a canonical pip-style form (lowercase, _ -> -)
normalize_name() {
  tr '[:upper:]' '[:lower:]' | sed 's/_/-/g'
}

# Extract pinned specs (conda or pip-style) from environment.yaml.
# Emits one line per constrained dep: "name|op|version"
# Compatible with POSIX awk (no gawk-only features).
read_pinned_specs() {
  local yaml="$1"
  awk '
    /^[[:space:]]*-[[:space:]]*/ {
      line = $0
      sub(/#.*/, "", line)                        # strip comments
      sub(/^[[:space:]]*-[[:space:]]*/, "", line) # strip "- "

      idx = match(line, /(==|>=|<=|>|<|=)/)       # first operator
      if (idx) {
        op   = substr(line, RSTART, RLENGTH)
        name = substr(line, 1, RSTART-1)
        ver  = substr(line, RSTART+RLENGTH)

        gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", ver)

        if (name != "" && ver != "")
          print name "|" op "|" ver
      }
    }
  ' "$yaml"
}

# ---------- Version detection (conda + pip) ----------

# From conda: return installed version or empty (case-insensitive)
conda_installed_version() {
  local env="$1" pkg="$2"
  local want
  want="$(printf '%s' "$pkg" | normalize_name)"

  conda list -n "$env" \
  | awk -v want="$want" 'BEGIN{IGNORECASE=1}
       /^#/ || NF < 2 { next }
       {
         name=$1
         gsub(/_/, "-", name)
         if (name == want) { print $2; exit }
       }'
}

# Cache pip versions from the env via "pip list --format=json"
declare -A PIP_VERS_CACHE
PIP_VERS_LOADED=""

load_pip_versions() {
  local env="$1"
  if [ "$PIP_VERS_LOADED" = "$env" ]; then return; fi
  PIP_VERS_CACHE=()
  while IFS= read -r line; do
    # NAME==VERSION
    local name="${line%%==*}"
    local ver="${line#*==}"
    [ -z "$name" ] && continue
    name="$(printf '%s' "$name" | normalize_name)"
    PIP_VERS_CACHE["$name"]="$ver"
  done < <(
    conda run -n "$env" python - <<'PY'
import json, sys, subprocess
out = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"], text=True)
for item in json.loads(out):
    name = item.get("name") or ""
    ver  = item.get("version") or ""
    if name and ver:
        print(f"{name}=={ver}")
PY
  )
  PIP_VERS_LOADED="$env"
}

pip_installed_version_cached() {
  local name_norm="$1"
  printf '%s' "${PIP_VERS_CACHE[$name_norm]:-}"
}

# Combined installed version: prefer conda, then pip
installed_version() {
  local env="$1" raw_pkg="$2"
  local conda_ver pip_ver
  conda_ver="$(conda_installed_version "$env" "$raw_pkg" || true)"
  if [ -n "$conda_ver" ]; then
    printf '%s' "$conda_ver"
    return 0
  fi
  local name_norm
  name_norm="$(printf '%s' "$raw_pkg" | normalize_name)"
  load_pip_versions "$env"
  pip_ver="$(pip_installed_version_cached "$name_norm")"
  printf '%s' "$pip_ver"
}

# ---------- Version comparison ----------

normalize_version() {
  local v="$1"
  v="${v//[^0-9.]/.}"
  echo "$v" | sed -E 's/\.+/./g; s/^\.//; s/\.$//'
}

# Echo -1 if a<b; 0 if equal; 1 if a>b
vercmp() {
  local a="$(normalize_version "${1:-0}")"
  local b="$(normalize_version "${2:-0}")"
  local IFS=.
  local -a A B
  read -r -a A <<< "$a"
  read -r -a B <<< "$b"
  local len="${#A[@]}"; [ "${#B[@]}" -gt "$len" ] && len="${#B[@]}"
  local i ai bi
  for ((i=0; i<len; i++)); do
    ai="${A[i]:-0}"; bi="${B[i]:-0}"
    ai=$((10#$ai)); bi=$((10#$bi))
    if (( ai < bi )); then echo -1; return 0; fi
    if (( ai > bi )); then echo  1; return 0; fi
  done
  echo 0
}

# '=' means prefix/subversion match (e.g., 0.6 accepts 0.6.4; 3.12 accepts 3.12.x)
ver_satisfies() {
  local have="$1" op="$2" want="$3"

  if [ "$op" = "=" ]; then
    if [[ "$want" == *".*" ]]; then
      local pre="${want%.*}"
      [[ "$(normalize_version "$have")" == "$(normalize_version "$pre")".* ]] && return 0 || return 1
    fi
    local segs
    segs="$(awk -F. '{print NF}' <<<"$(normalize_version "$want")")"
    if [ "$segs" -ge 3 ]; then
      [ "$(normalize_version "$have")" = "$(normalize_version "$want")" ] && return 0 || return 1
    else
      [[ "$(normalize_version "$have")" == "$(normalize_version "$want")"* ]] && return 0 || return 1
    fi
  fi

  if [ "$op" = "==" ]; then
    [ "$(normalize_version "$have")" = "$(normalize_version "$want")" ] && return 0 || return 1
  fi

  local cmp
  cmp="$(vercmp "$have" "$want")"
  case "$op" in
    ">=") [ "$cmp" -ge 0 ] ;;
    "<=") [ "$cmp" -le 0 ] ;;
    ">")  [ "$cmp" -gt 0 ] ;;
    "<")  [ "$cmp" -lt 0 ] ;;
    *)    return 1 ;;
  esac
}

# ----------------------------
# Preconditions
# ----------------------------
if [ ! -d "${PIPELINES_DIR:-}" ]; then
  echo "Error: PIPELINES_DIR not defined."
  exit 1
fi

# --- Accept Anaconda TOS when supported by this conda; otherwise skip ---
conda_has_cmd() {
  conda commands 2>/dev/null | awk '{print $1}' | grep -qx "$1"
}

if conda_has_cmd tos; then
  log "conda 'tos' available → accepting ToS for required channels…"
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true
else
  log "conda 'tos' not available → skipping ToS acceptance (not needed on this setup)."
fi

PIPE_BASE="$PIPELINES_DIR/training_set_maker"
ENV_YAML="${PIPE_BASE}/environment.yaml"
ENV_NAME="pipe_tsm"

[ -f "$ENV_YAML" ] || { echo "Error: environment.yaml not found at: $ENV_YAML"; exit 1; }

# ----------------------------
# Decide: create / recreate / skip
# ----------------------------
NEED_CREATE=0
NEED_RECREATE=0

if ! env_exists "$ENV_NAME"; then
  log "Conda env '$ENV_NAME' does not exist → will create."
  NEED_CREATE=1
else
  log "Conda env '$ENV_NAME' exists → checking pinned constraints (conda + pip)…"

  # Safer capture: temp file so parser errors cannot be masked
  TMP_PINS="$(mktemp)"
  if ! read_pinned_specs "$ENV_YAML" > "$TMP_PINS"; then
    echo "Failed to parse pinned specs from $ENV_YAML"; rm -f "$TMP_PINS"; exit 1
  fi
  mapfile -t PINNED < "$TMP_PINS"
  rm -f "$TMP_PINS"

  if [ "${#PINNED[@]}" -eq 0 ]; then
    log "No pinned packages with version constraints found → skipping reinstall."
  else
    log "Pinned constraints detected:"
    for l in "${PINNED[@]}"; do log "  - $l"; done

    for line in "${PINNED[@]}"; do
      IFS='|' read -r raw_pkg op want <<< "$line"
      have="$(installed_version "$ENV_NAME" "$raw_pkg" || true)"

      if [ -z "${have:-}" ]; then
        log "Constraint not satisfied: '$raw_pkg' is missing in env."
        NEED_RECREATE=1
        break
      fi

      if ! ver_satisfies "$have" "$op" "$want"; then
        log "Constraint not satisfied: $raw_pkg (installed=$have) does not satisfy '$op $want'"
        NEED_RECREATE=1
        break
      else
        log "OK: $raw_pkg (installed=$have) satisfies '$op $want'"
      fi
    done
  fi
fi

# ----------------------------
# Apply action
# ----------------------------
if [ "$NEED_CREATE" -eq 1 ]; then
  log "Creating env '$ENV_NAME' from $ENV_YAML…"
  conda env create -f "$ENV_YAML"
elif [ "$NEED_RECREATE" -eq 1 ]; then
  log "Recreating env '$ENV_NAME' to satisfy updated constraints…"
  conda env remove -n "$ENV_NAME" -y || true
  conda env create -f "$ENV_YAML"
else
  log "Pinned constraints satisfied → skipping reinstall."
fi

# ----------------------------
# Activate env and export PATH/PYTHONPATH for TSM
# ----------------------------
conda activate "$ENV_NAME" || { echo "Failed to activate $ENV_NAME"; exit 1; }

# Export PATH for TSM scripts
export PATH="$PATH:${PIPE_BASE}/scripts/"

# Ensure Python sees local TSM packages
if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${PYTHONPATH}:${PIPE_BASE}/packages/"
else
  export PYTHONPATH="${PIPE_BASE}/packages/"
fi

echo "Conda Environment: $CONDA_DEFAULT_ENV"

