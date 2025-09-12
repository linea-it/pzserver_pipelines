#!/bin/bash

# ==============================
# Strict flags + errtrace + PS4
# ==============================
set -Eeuo pipefail
set -o errtrace  # propagate ERR into functions/subshells/sourced files

# Prefix for `set -x`: timestamp, file, and line
export PS4='+ [$(date "+%Y-%m-%d %H:%M:%S")] (${BASH_SOURCE##*/}:${LINENO}) '

# ---------------- Helpers ----------------

# Print a compact call stack (source:line in order)
_print_call_stack() {
  local i
  for ((i=0; i<${#BASH_SOURCE[@]}; i++)); do
    local src="${BASH_SOURCE[$i]}"
    local func="${FUNCNAME[$i]:-main}"
    local line="${BASH_LINENO[$((i-1))]:-?}"
    printf '   #%d %s:%s in %s()\n' "$i" "${src:-?}" "${line}" "${func}"
  done
}

# Logger with timestamp
log() {
  local ts
  ts="$(date "+%Y-%m-%d %H:%M:%S")"
  echo "[$ts] $*"
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
  log "❌ Fail"
  exit $code
}' ERR

# ---------------- Args ----------------
if [ $# -eq 0 ]; then
  echo "Error: No arguments provided."
  exit 1
fi

ARGS=$@

# ---------------- Basic checks ----------------
if [ -z "${PIPELINES_DIR:-}" ] || [ ! -d "${PIPELINES_DIR}" ]; then
    echo "Error: PIPELINES_DIR not defined."
    exit 1
fi

INSTALL_PIPE="$PIPELINES_DIR/training_set_maker/install.sh"
if [ ! -f "$INSTALL_PIPE" ]; then
    echo "Error: Installation script not found."
    exit 1
fi

# ---------------- Logs wiring ----------------
LOGS_DIR="./process_info"
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/process.log"

# Duplicate stdout/stderr to both terminal and process.log
exec > >(tee -a "$LOG_FILE") 2>&1

# ---------------- Install pipeline ----------------
log "Installing pipeline..."
# shellcheck disable=SC1090
set -x
. "$INSTALL_PIPE"
{ set +x; } 2>/dev/null

# ---------------- Run TSM ----------------
set -x
tsm-run $ARGS
{ set +x; } 2>/dev/null

# ---------------- Epilogue ----------------
log "Pipeline exited with code: 0"
log "✅ Success (logs: ${LOG_FILE})"
exit 0

