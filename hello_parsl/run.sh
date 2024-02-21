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

INSTALL_PIPE="$PIPELINES_DIR/hello_parsl/install.sh"

if [ ! -f "$INSTALL_PIPE" ]; then
    echo "Error: Installation script not found."
    exit 1
fi

# Installing pipeline
echo "Installing pipeline..."
. "$INSTALL_PIPE"

set -xe

# Run the Python code with the given argument
run-hello-world $ARGS || { echo "Failed to run-hello-world"; exit 1; }

echo "Done."