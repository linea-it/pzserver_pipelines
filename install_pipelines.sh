#!/bin/bash

if [ ! -d "$PIPELINES_DIR" ]; then
    export PIPELINES_DIR=$APP_DIR
fi

echo "The directory containing the pipelines is: " "$PIPELINES_DIR"

source $(dirname $CONDA_EXE)/activate || { echo "Failed to activate Conda environment"; exit 1; }

for pipe in $( ls ${PIPELINES_DIR}/*/install.sh)
do
    echo "Installing: ${pipe}"
    . "$pipe"
done
