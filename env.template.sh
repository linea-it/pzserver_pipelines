#!/bin/bash

export APP_DIR="<APP_DIR>"
export PIPELINES_DIR="<PIPELINES_DIR>"
export DATASETS_DIR="<DATASETS_DIR>/data-example"

if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=$APP_DIR/src
else
    export PYTHONPATH=$PYTHONPATH:$APP_DIR/src
fi

conda activate pz_pipelines