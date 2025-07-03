#!/bin/bash

# Check if the argument was given
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided."
    exit 1
fi

ARGS=$@
shift $#

if [ ! -d "$PIPELINES_DIR" ]; then
    echo "Error: PIPELINES_DIR not defined."
    exit 1
fi

INSTALL_PIPE="$PIPELINES_DIR/combine_redshift_dedup/install.sh"
PIPE_BASE="$PIPELINES_DIR/combine_redshift_dedup"
LOGS_DIR="$PIPE_BASE/logs"
mkdir -p "$LOGS_DIR"

LOG_FILE="$LOGS_DIR/run_$(date +'%Y%m%d_%H%M%S').log"

if [ ! -f "$INSTALL_PIPE" ]; then
    echo "Error: Installation script not found."
    exit 1
fi

# Start logging everything
{
    echo "Installing pipeline..."
    . "$INSTALL_PIPE"

    set -xe

    PYTHONPATH="$PIPELINES_DIR:$PYTHONPATH" python "$PIPE_BASE/scripts/crd-run.py" $ARGS

    echo "Done."

} 2>&1 | tee -a "$LOG_FILE"
