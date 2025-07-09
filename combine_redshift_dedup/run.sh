#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: ./run.sh <config.yaml> <base_dir_override>"
    exit 1
fi

CONFIG_PATH="$1"
BASE_DIR_OVERRIDE="$2"

if [ ! -d "$PIPELINES_DIR" ]; then
    echo "Error: PIPELINES_DIR not defined."
    exit 1
fi

INSTALL_PIPE="$PIPELINES_DIR/combine_redshift_dedup/install.sh"
PIPE_BASE="$PIPELINES_DIR/combine_redshift_dedup"

# Creates logs directory inside base_dir_override/process_info
LOGS_DIR="${BASE_DIR_OVERRIDE}/process_info"
mkdir -p "$LOGS_DIR"

# Nome fixo do arquivo de log
LOG_FILE="$LOGS_DIR/process.log"

if [ ! -f "$INSTALL_PIPE" ]; then
    echo "Error: Installation script not found."
    exit 1
fi

{
    echo "Installing pipeline..."
    . "$INSTALL_PIPE"

    set -xe

    PYTHONPATH="$PIPELINES_DIR:$PYTHONPATH" \
    python "$PIPE_BASE/scripts/crd-run.py" "$CONFIG_PATH" --base_dir "$BASE_DIR_OVERRIDE"

    echo "Done."

} 2>&1 | tee -a "$LOG_FILE"
