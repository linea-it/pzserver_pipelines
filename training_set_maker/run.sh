#!/bin/bash

# ==============================
# Strict flags + errtrace + PS4
# ==============================
set -Eeuo pipefail
set -o errtrace  # propagate ERR into functions/subshells/sourced files
export PS4='+ [$(date "+%Y-%m-%d %H:%M:%S")] (${BASH_SOURCE##*/}:${LINENO}) '

# ---- HARD KILL any inherited xtrace right away ----
{ set +x; } 2>/dev/null
exec 9>/dev/null
# Route xtrace to /dev/null unless explicitly enabled
export BASH_XTRACEFD=9

# Enable xtrace ONLY when explicitly allowed and in interactive TTYs
_enable_xtrace() {
  if [ "${DEBUG:-0}" = "1" ] && [ "${ALLOW_XTRACE:-0}" = "1" ] && [ -t 2 ]; then
    export BASH_XTRACEFD=2
    set -x
  fi
}
_disable_xtrace() {
  { set +x; } 2>/dev/null
  export BASH_XTRACEFD=9
}

# ---------------- Helpers ----------------
_print_call_stack() {
  local i
  for ((i=0; i<${#BASH_SOURCE[@]}; i++)); do
    local src="${BASH_SOURCE[$i]}"
    local func="${FUNCNAME[$i]:-main}"
    local line="${BASH_LINENO[$((i-1))]:-?}"
    printf '   #%d %s:%s in %s()\n' "$i" "${src:-?}" "${line}" "${func}"
  done
}

log() {
  local ts; ts="$(date "+%Y-%m-%d %H:%M:%S")"
  echo "[$ts] $*"
}

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

# ---------------- Error Trap ----------------
trap '{
  code=$?
  { set +x; } 2>/dev/null
  local_cmd=${BASH_COMMAND}
  local_src=${BASH_SOURCE[0]}
  local_line=${LINENO}

  log "[ERROR] Exit code: ${code}"
  log "[ERROR] While running: ${local_cmd}"
  log "[ERROR] At: ${local_src}:${local_line}"

  if [ ${#BASH_SOURCE[@]} -gt 1 ]; then
    log "[ERROR] Call stack (most recent first):"
    _print_call_stack | sed -e "s/^/[$(date "+%Y-%m-%d %H:%M:%S")] /"
  fi

  log "[ERROR] Context:"
  log "        PIPELINES_DIR='${PIPELINES_DIR-}'"
  log "        INSTALL_PIPE='${INSTALL_PIPE-}'"
  log "        RUN_DIR='${RUN_DIR-}'"
  log "        LOG_FILE='${LOG_FILE-}'"
  log "❌ Fail"
  exit $code
}' ERR

# ---------------- Args ----------------
if [ $# -lt 1 ]; then
  echo "Usage: ./run.sh <config.yaml> [run_dir]"
  exit 1
fi

CONFIG_PATH="$1"
RUN_DIR="${2:-}"

# Auto-pick run directory if not supplied
if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$(pick_next_process_dir ".")"
  log "No run directory provided. Using '${RUN_DIR}'."
fi
mkdir -p "$RUN_DIR"

# ---------------- Basic checks ----------------
if [ -z "${PIPELINES_DIR:-}" ] || [ ! -d "${PIPELINES_DIR}" ]; then
  echo "Error: PIPELINES_DIR not defined."
  exit 1
fi

INSTALL_PIPE="$PIPELINES_DIR/training_set_maker/install.sh"
if [ ! -f "$INSTALL_PIPE" ]; then
  echo "Error: Installation script not found at: $INSTALL_PIPE"
  exit 1
fi

# ---------------- Logs wiring ----------------
LOGS_DIR="$RUN_DIR/process_info"
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/process.log"

# Replicate stdout/stderr: both terminal and process.log
exec > >(tee -a "$LOG_FILE") 2>&1

# ---------------- Install pipeline ----------------
log "Installing pipeline..."
_enable_xtrace
# shellcheck disable=SC1090
. "$INSTALL_PIPE"
_disable_xtrace

# ---------------- Run TSM ----------------
# NOTE: tsm-run expects positional args: config_path [cwd]
# We pass RUN_DIR as the optional working directory (cwd).
_enable_xtrace
tsm-run "$CONFIG_PATH" "$RUN_DIR"
_disable_xtrace

# ---------------- Epilogue ----------------
log "Pipeline exited with code: 0"
log "✅ Success (run dir: ${RUN_DIR})"
exit 0

