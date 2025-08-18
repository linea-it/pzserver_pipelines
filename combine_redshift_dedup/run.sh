#!/bin/bash

# Exit immediately on errors, treat unset variables as errors,
# and make pipelines fail if any command in them fails
set -Eeuo pipefail

# Trap errors and print a message before exiting
trap 'code=$?; echo "[ERROR] Command \"$BASH_COMMAND\" failed at line $LINENO (exit $code)"; exit 1' ERR

if [ $# -lt 2 ]; then
    echo "Usage: ./run.sh <config.yaml> <base_dir_override>"
    exit 1
fi

CONFIG_PATH="$1"
BASE_DIR_OVERRIDE="$2"

if [ -z "${PIPELINES_DIR:-}" ] || [ ! -d "$PIPELINES_DIR" ]; then
    echo "Error: PIPELINES_DIR not defined or not a directory."
    exit 1
fi

INSTALL_PIPE="$PIPELINES_DIR/combine_redshift_dedup/install.sh"
PIPE_BASE="$PIPELINES_DIR/combine_redshift_dedup"

# Create logs directory inside base_dir_override/process_info
LOGS_DIR="${BASE_DIR_OVERRIDE}/process_info"
mkdir -p "$LOGS_DIR"

# Fixed name for the log file
LOG_FILE="$LOGS_DIR/process.log"

if [ ! -f "$INSTALL_PIPE" ]; then
    echo "Error: Installation script not found at: $INSTALL_PIPE"
    exit 1
fi

# Redirect both stdout and stderr to the log file *and* the terminal
# without breaking error codes (avoids tee pipeline issue)
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Installing pipeline..."
# If install.sh fails, the ERR trap will trigger
. "$INSTALL_PIPE"

set -x

# Run Python script; if it fails, the ERR trap will trigger
PYTHONPATH="$PIPELINES_DIR:${PYTHONPATH:-}" \
python "$PIPE_BASE/scripts/crd-run.py" "$CONFIG_PATH" --base_dir "$BASE_DIR_OVERRIDE"
PIPE_EXIT=$?

set +x
echo "Pipeline exited with code: $PIPE_EXIT"
if [ $PIPE_EXIT -eq 0 ]; then
    echo "✅ Success"
else
    echo "❌ Fail"
fi

exit $PIPE_EXIT