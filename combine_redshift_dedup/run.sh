#!/bin/bash

# Exit on errors, unset vars are errors, and fail a pipeline if any command in it fails
set -Eeuo pipefail

# Make ERR propagate through functions and sourced scripts; improve xtrace prefix
set -o errtrace  # so ERR trap runs for functions/subshel ls/sourced files

# Pretty PS4 for `set -x` (shows timestamp, file, and line)
export PS4='+ [$(date "+%Y-%m-%d %H:%M:%S")] (${BASH_SOURCE##*/}:${LINENO}) '

# Utility: print a compact call stack (source:line in order)
_print_call_stack() {
  local i
  # BASH_SOURCE[0] is this file; BASH_LINENO[i-1] is the callsite line
  for ((i=0; i<${#BASH_SOURCE[@]}; i++)); do
    # Skip empty frames
    local src="${BASH_SOURCE[$i]}"
    local func="${FUNCNAME[$i]:-main}"
    local line="${BASH_LINENO[$((i-1))]:-?}"
    printf '   #%d %s:%s in %s()\n' "$i" "${src:-?}" "${line}" "${func}"
  done
}

# -------- Helpers --------

# Print an ISO-8601 timestamped line to stdout
log() {
  local ts
  ts="$(date "+%Y-%m-%d %H:%M:%S")"
  echo "[$ts] $*"
}

# Best-effort detection of the driver's primary IPv4 address.
# Priority:
#   1) hostname -I  (first non-loopback IPv4)
#   2) ip route get (extract 'src' field)
#   3) ifconfig/ip fallback heuristics
# Returns IP via stdout; empty string if not found.
detect_driver_ip() {
  # 1) hostname -I gives all addresses separated by spaces
  if command -v hostname >/dev/null 2>&1; then
    # shellcheck disable=SC2207
    local addrs=($(hostname -I 2>/dev/null || true))
    for a in "${addrs[@]:-}"; do
      # Pick first non-loopback IPv4
      if [[ "$a" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] && [[ ! "$a" =~ ^127\. ]]; then
        echo "$a"
        return 0
      fi
    done
  fi

  # 2) ip route get to a public address (forces route lookup with src)
  if command -v ip >/dev/null 2>&1; then
    # Try Cloudflare DNS; parse the 'src' token
    local r
    r="$(ip route get 1.1.1.1 2>/dev/null || true)"
    if [[ -n "$r" ]]; then
      # Prefer the 'src <ip>' token if present
      local src
      src="$(awk '/src/ {for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}' <<<"$r" || true)"
      if [[ "$src" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] && [[ ! "$src" =~ ^127\. ]]; then
        echo "$src"
        return 0
      fi
      # Otherwise, grab the last IPv4-looking field
      local last_ipv4
      last_ipv4="$(grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' <<<"$r" | tail -n1 || true)"
      if [[ "$last_ipv4" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] && [[ ! "$last_ipv4" =~ ^127\. ]]; then
        echo "$last_ipv4"
        return 0
      fi
    fi
  fi

  # 3) Fallbacks: try ifconfig (BSD/macOS) if available
  if command -v ifconfig >/dev/null 2>&1; then
    local cand
    cand="$(ifconfig 2>/dev/null | grep -Eo 'inet ([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $2}' | grep -v '^127\.' | head -n1 || true)"
    if [[ "$cand" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
      echo "$cand"
      return 0
    fi
  fi

  # Nothing worked
  echo ""
  return 1
}

# Pick next available process directory under a root (defaults to .)
pick_next_process_dir() {
  local root="${1:-.}"
  local i=1 name path
  while :; do
    printf -v name "process%03d" "$i"
    path="${root}/${name}"
    if [ ! -e "$path" ]; then
      echo "$path"
      return 0
    fi
    i=$((i+1))
  done
}

# -------- Error Trap --------
trap '{
  code=$?
  # Turn off xtrace for clean error logs
  { set +x; } 2>/dev/null
  # Snapshot failing command and location
  local_cmd=${BASH_COMMAND}
  local_src=${BASH_SOURCE[0]}
  local_line=${LINENO}

  log "[ERROR] Exit code: ${code}"
  log "[ERROR] While running: ${local_cmd}"
  log "[ERROR] At: ${local_src}:${local_line}"

  # If the error originated in a sourced file (e.g., install.sh), show the stack
  if [ ${#BASH_SOURCE[@]} -gt 1 ]; then
    log "[ERROR] Call stack (most recent first):"
    _print_call_stack | sed -e "s/^/[$(date "+%Y-%m-%d %H:%M:%S")] /"
  fi

  # Helpful runtime context
  log "[ERROR] Context:"
  log "        CONFIG_PATH='${CONFIG_PATH-}'"
  log "        BASE_DIR_OVERRIDE='${BASE_DIR_OVERRIDE-}'"
  log "        PIPELINES_DIR='${PIPELINES_DIR-}'"
  log "        INSTALL_PIPE='${INSTALL_PIPE-}'"
  log "        CRC_LOG_COLLECTOR='${CRC_LOG_COLLECTOR-}'"
  log "❌ Fail"
  exit $code
}' ERR

# -------- Args & basic checks --------

if [ $# -lt 1 ]; then
  echo "Usage: ./run.sh <config.yaml> [run_dir]"
  echo "If [run_dir] is not provided, the script will pick process001, process002, ..."
  exit 1
fi

CONFIG_PATH="$1"
# Optional 2nd arg. If empty, we auto-pick processNNN under current dir.
BASE_DIR_OVERRIDE="${2:-}"

if [ -z "${PIPELINES_DIR:-}" ] || [ ! -d "${PIPELINES_DIR:-/nonexistent}" ]; then
  echo "Error: PIPELINES_DIR not defined or not a directory."
  exit 1
fi

INSTALL_PIPE="$PIPELINES_DIR/combine_redshift_dedup/install.sh"
PIPE_BASE="$PIPELINES_DIR/combine_redshift_dedup"

if [ ! -f "$INSTALL_PIPE" ]; then
  echo "Error: Installation script not found at: $INSTALL_PIPE"
  exit 1
fi

# Auto-pick run directory if not supplied
if [ -z "$BASE_DIR_OVERRIDE" ]; then
  BASE_DIR_OVERRIDE="$(pick_next_process_dir ".")"
  log "No run directory provided. Using '${BASE_DIR_OVERRIDE}'."
fi

# Ensure the chosen run directory exists
mkdir -p "$BASE_DIR_OVERRIDE"

# -------- Logs wiring (shell level) --------

LOGS_DIR="${BASE_DIR_OVERRIDE}/process_info"
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/process.log"

# Duplicate stdout/stderr to both terminal and process.log
exec > >(tee -a "$LOG_FILE") 2>&1

# -------- Install pipeline deps --------

log "Installing pipeline..."
# shellcheck disable=SC1090
set -x
. "$INSTALL_PIPE"
{ set +x; } 2>/dev/null

# -------- CRC_LOG_COLLECTOR auto-detection --------
# If CRC_LOG_COLLECTOR is already set, keep it.
# Otherwise, build "<driver_ip>:<port>", where port defaults to 19997
# but can be overridden by CRC_LOG_COLLECTOR_PORT.

: "${CRC_LOG_COLLECTOR_PORT:=19997}"

if [ -z "${CRC_LOG_COLLECTOR:-}" ]; then
  DRIVER_IP="$(detect_driver_ip || true)"
  if [ -z "$DRIVER_IP" ]; then
    # Last resort: fall back to loopback (OK for single-node runs)
    DRIVER_IP="127.0.0.1"
    log "⚠️  Could not detect a non-loopback IP automatically; using ${DRIVER_IP}."
    log "   If you run a multi-node cluster, export CRC_LOG_COLLECTOR='<driver_ip>:${CRC_LOG_COLLECTOR_PORT}' explicitly."
  fi
  export CRC_LOG_COLLECTOR="${DRIVER_IP}:${CRC_LOG_COLLECTOR_PORT}"
fi

log "CRC_LOG_COLLECTOR=${CRC_LOG_COLLECTOR}"

# -------- Run the pipeline --------

set -x
PYTHONPATH="$PIPELINES_DIR:${PYTHONPATH:-}" \
CRC_LOG_COLLECTOR="$CRC_LOG_COLLECTOR" \
python "$PIPE_BASE/scripts/crd-run.py" "$CONFIG_PATH" --base_dir "$BASE_DIR_OVERRIDE"
set +x

# -------- Epilogue --------

log "Pipeline exited with code: 0"
log "✅ Success (run dir: ${BASE_DIR_OVERRIDE})"
exit 0
