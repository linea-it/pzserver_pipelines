#!/bin/bash

source `dirname $CONDA_EXE`/activate || { echo "Failed to activate Conda environment"; exit 1; }

export APP_DIR="<APP_DIR>"
export PIPELINES_DIR="<PIPELINES_DIR>"
export DATASETS_DIR="<DATASETS_DIR>"

if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=$APP_DIR/src
else
    export PYTHONPATH=$PYTHONPATH:$APP_DIR/src
fi

conda activate pz_pipelines