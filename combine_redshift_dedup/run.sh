#!/bin/bash

# =========================================
# Strict flags, errtrace, and quiet-by-default
# =========================================
set -Eeuo pipefail
set -o errtrace
export PS4='+ [$(date "+%Y-%m-%d %H:%M:%S")] (${BASH_SOURCE##*/}:${LINENO}) '

# ---- HARD KILL any inherited xtrace right away ----
{ set +x; } 2>/dev/null
exec 9>/dev/null
# Force xtrace output to /dev/null unless we explicitly rewire it
export BASH_XTRACEFD=9

# Helpers to (re)enable / disable xtrace safely.
_enable_xtrace() {
  # Enable xtrace ONLY if explicitly requested, and only on interactive TTYs.
  if [ "${DEBUG:-0}" = "1" ] && [ "${ALLOW_XTRACE:-0}" = "1" ] && [ -t 2 ]; then
    export BASH_XTRACEFD=2
    set -x
  fi
}
_disable_xtrace() {
  { set +x; } 2>/dev/null
  export BASH_XTRACEFD=9
}

# =========================
# Helpers
# =========================

# Compact call stack printer (source:line in order)
_print_call_stack() {
  local i
  for ((i=0; i<${#BASH_SOURCE[@]}; i++)); do
    local src="${BASH_SOURCE[$i]}"
    local func="${FUNCNAME[$i]:-main}"
    local line="${BASH_LINENO[$((i-1))]:-?}"
    printf '   #%d %s:%s in %s()\n' "$i" "${src:-?}" "${line}" "${func}"
  done
}

# Timestamped logging helpers
log()  { echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*"; }
warn() { echo "[$(date "+%Y-%m-%d %H:%M:%S")] [WARN] $*"  >&2; }
err()  { echo "[$(date "+%Y-%m-%d %H:%M:%S")] [ERROR] $*" >&2; }

# Best-effort detection of the driver's primary IPv4 address.
detect_driver_ip() {
  if command -v hostname >/dev/null 2>&1; then
    # shellcheck disable=SC2207
    local addrs=($(hostname -I 2>/dev/null || true))
    for a in "${addrs[@]:-}"; do
      if [[ "$a" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] && [[ ! "$a" =~ ^127\. ]]; then
        echo "$a"; return 0
      fi
    done
  fi
  if command -v ip >/dev/null 2>&1; then
    local r src last_ipv4
    r="$(ip route get 1.1.1.1 2>/dev/null || true)"
    if [[ -n "$r" ]]; then
      src="$(awk '/src/ {for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}' <<<"$r" || true)"
      if [[ "$src" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] && [[ ! "$src" =~ ^127\. ]]; then echo "$src"; return 0; fi
      last_ipv4="$(grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' <<<"$r" | tail -n1 || true)"
      if [[ "$last_ipv4" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] && [[ ! "$last_ipv4" =~ ^127\. ]]; then echo "$last_ipv4"; return 0; fi
    fi
  fi
  if command -v ifconfig >/dev/null 2>&1; then
    local cand
    cand="$(ifconfig 2>/dev/null | grep -Eo 'inet ([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $2}' | grep -v '^127\.' | head -n1 || true)"
    if [[ "$cand" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then echo "$cand"; return 0; fi
  fi
  echo ""; return 1
}

# Pick next available process directory under a root (defaults to .)
pick_next_process_dir() {
  local root="${1:-.}"
  local i=1 name path
  while :; do
    printf -v name "process%03d" "$i"
    path="${root}/${name}"
    if [ ! -e "$path" ]; then echo "$path"; return 0; fi
    i=$((i+1))
  done
}

# =========================
# Error Trap
# =========================
trap '{
  code=$?
  { set +x; } 2>/dev/null

  local_cmd=${BASH_COMMAND}
  local_src=${BASH_SOURCE[0]}
  local_line=${LINENO}

  err "Exit code: ${code}"
  err "While running: ${local_cmd}"
  err "At: ${local_src}:${local_line}"

  if [ ${#BASH_SOURCE[@]} -gt 1 ]; then
    err "Call stack (most recent first):"
    _print_call_stack | sed -e "s/^/[$(date "+%Y-%m-%d %H:%M:%S")] /" >&2
  fi

  err "Context:"
  err "  CONFIG_PATH='${CONFIG_PATH-}'"
  err "  BASE_DIR_OVERRIDE='${BASE_DIR_OVERRIDE-}'"
  err "  PIPELINES_DIR='${PIPELINES_DIR-}'"
  err "  INSTALL_PIPE='${INSTALL_PIPE-}'"
  err "  CRC_LOG_COLLECTOR='${CRC_LOG_COLLECTOR-}'"

  if [ -n "${LOG_FILE-}" ] && [ -f "${LOG_FILE-}" ]; then
    err "---- tail of ${LOG_FILE} ----"
    tail -n 200 "$LOG_FILE" >&2 || true
    err "-----------------------------"
  fi

  err "❌ Fail"
  exit $code
}' ERR

# =========================
# Args & basic checks
# =========================
if [ $# -lt 1 ]; then
  err "Usage: ./run.sh <config.yaml> [run_dir]"
  err "If [run_dir] is not provided, the script will pick process001, process002, ..."
  exit 1
fi

CONFIG_PATH="$1"
BASE_DIR_OVERRIDE="${2:-}"

if [ -z "${PIPELINES_DIR:-}" ] || [ ! -d "${PIPELINES_DIR:-/nonexistent}" ]; then
  err "Error: PIPELINES_DIR not defined or not a directory."
  exit 1
fi

INSTALL_PIPE="$PIPELINES_DIR/combine_redshift_dedup/install.sh"
PIPE_BASE="$PIPELINES_DIR/combine_redshift_dedup"

if [ ! -f "$INSTALL_PIPE" ]; then
  err "Error: Installation script not found at: $INSTALL_PIPE"
  exit 1
fi

# Auto-pick run directory if not supplied
if [ -z "$BASE_DIR_OVERRIDE" ]; then
  BASE_DIR_OVERRIDE="$(pick_next_process_dir ".")"
  log "No run directory provided. Using '${BASE_DIR_OVERRIDE}'."
fi
mkdir -p "$BASE_DIR_OVERRIDE"

# =========================
# Logs wiring (shell level)
# =========================
LOGS_DIR="${BASE_DIR_OVERRIDE}/process_info"
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/process.log"

# Replicate logs: both stdout and stderr go to terminal AND file
exec > >(tee -a "$LOG_FILE") 2>&1

# =========================
# Install pipeline deps
# =========================
log "Installing pipeline..."
# Don’t trust external DEBUG/BASH_XTRACEFD unless explicitly allowed
_enable_xtrace
# shellcheck disable=SC1090
. "$INSTALL_PIPE"
_disable_xtrace

# =========================
# CRC_LOG_COLLECTOR auto-detection
# =========================
: "${CRC_LOG_COLLECTOR_PORT:=19997}"

if [ -z "${CRC_LOG_COLLECTOR:-}" ]; then
  DRIVER_IP="$(detect_driver_ip || true)"
  if [ -z "$DRIVER_IP" ]; then
    DRIVER_IP="127.0.0.1"
    warn "Could not detect a non-loopback IP automatically; using ${DRIVER_IP}."
    warn "If you run a multi-node cluster, export CRC_LOG_COLLECTOR='<driver_ip>:${CRC_LOG_COLLECTOR_PORT}' explicitly."
  fi
  export CRC_LOG_COLLECTOR="${DRIVER_IP}:${CRC_LOG_COLLECTOR_PORT}"
fi

log "CRC_LOG_COLLECTOR=${CRC_LOG_COLLECTOR}"

# =========================
# Verbosity toggles (CRC logger + Python warnings)
# =========================
# Aceita override via CLI: ex: CRC_LOG_LEVEL=INFO ./run.sh cfg.yaml
: "${CRC_LOG_LEVEL:=DEBUG}"                 # DEBUG/INFO/WARNING/ERROR/CRITICAL or numeric (10/20/30/40/50)
: "${CRC_LOG_INCLUDE_NONCRC_WARNINGS:=1}"   # 1 = allows third-party WARNING to pass through the filter
: "${CRC_LOG_FORCE_RECONFIG:=1}"            # force recreate handlers on this PID (useful for re-runs)
: "${PYTHONWARNINGS:=default}"              # ensures that warnings are not silenced by Python

export CRC_LOG_LEVEL CRC_LOG_INCLUDE_NONCRC_WARNINGS CRC_LOG_FORCE_RECONFIG PYTHONWARNINGS

log "CRC_LOG_LEVEL=${CRC_LOG_LEVEL}"
log "CRC_LOG_INCLUDE_NONCRC_WARNINGS=${CRC_LOG_INCLUDE_NONCRC_WARNINGS}"
log "CRC_LOG_FORCE_RECONFIG=${CRC_LOG_FORCE_RECONFIG}"
log "PYTHONWARNINGS=${PYTHONWARNINGS}"

# =========================
# Run the pipeline
# =========================
export CRC_LAUNCH_DIR="$PWD"

_enable_xtrace
PYTHONPATH="$PIPELINES_DIR:${PYTHONPATH:-}" \
CRC_LOG_COLLECTOR="$CRC_LOG_COLLECTOR" \
python "$PIPE_BASE/scripts/crd-run.py" "$CONFIG_PATH" --base_dir "$BASE_DIR_OVERRIDE"
_disable_xtrace

# =========================
# Epilogue
# =========================
log "Pipeline exited with code: 0"
log "✅ Success (run dir: ${BASE_DIR_OVERRIDE})"
exit 0

