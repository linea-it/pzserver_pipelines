#!/bin/bash

# Exit immediately on errors, treat unset variables as errors,
# and make pipelines fail if any command in them fails
set -Eeuo pipefail

# Trap errors: if any command fails, print timestamped error message + "❌ Fail" and exit 1
trap '{
    code=$?  # capture the real exit code immediately
    set +x   # then disable xtrace, without overwriting $code
    ts=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$ts] [ERROR] A command failed at line $LINENO (exit $code)"
    echo "[$ts] ❌ Fail"
    exit $code
}' ERR

# Require at least 2 arguments: config file and base directory override
if [ $# -lt 2 ]; then
    echo "Usage: ./run.sh <config.yaml> <base_dir_override>"
    exit 1
fi

CONFIG_PATH="$1"
BASE_DIR_OVERRIDE="$2"

# Check if PIPELINES_DIR environment variable is defined and points to a directory
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

# Verify installation script exists
if [ ! -f "$INSTALL_PIPE" ]; then
    echo "Error: Installation script not found at: $INSTALL_PIPE"
    exit 1
fi

# Redirect both stdout and stderr to the log file *and* the terminal
exec > >(tee -a "$LOG_FILE") 2>&1

# Timestamped log: start installation
ts=$(date "+%Y-%m-%d %H:%M:%S")
echo "[$ts] Installing pipeline..."
. "$INSTALL_PIPE"

# Enable command trace for debugging (commands will be printed with '+ ...')
set -x
PYTHONPATH="$PIPELINES_DIR:${PYTHONPATH:-}" \
python "$PIPE_BASE/scripts/crd-run.py" "$CONFIG_PATH" --base_dir "$BASE_DIR_OVERRIDE"
set +x

# If we reached this point, no error occurred (pipeline finished successfully)
ts=$(date "+%Y-%m-%d %H:%M:%S")
echo "[$ts] Pipeline exited with code: 0"
echo "[$ts] ✅ Success"
exit 0