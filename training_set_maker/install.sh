#!/bin/bash --login

source `dirname $CONDA_EXE`/activate || { echo "Failed to activate Conda environment"; exit 1; }

if [ ! -d "$PIPELINES_DIR" ]; then
    echo "Error: PIPELINES_DIR not defined."
    exit 1
fi

PIPE_BASE="$PIPELINES_DIR/training_set_maker"
HASENV=`conda env list | grep pipe_tsm`

if [ -z "$HASENV" ]; then
    echo "Create virtual environment..."
    conda env create -f ${PIPE_BASE}/environment.yaml
    echo "Virtual environment created and packages installed."
# else
#     if [ "$CONDA_FORCE_UPDATE" == "yes" ]; then
#         echo "Virtual environment already exists. Updating..."
#         conda env update --file ${PIPE_BASE}/environment.yaml --prune
#     fi
fi

conda activate pipe_tsm

export PATH=$PATH:"$PIPE_BASE/scripts/"

if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$PIPE_BASE/packages/"
else
    export PYTHONPATH=$PYTHONPATH:"$PIPE_BASE/packages/"
fi

echo "Conda Environment: $CONDA_DEFAULT_ENV"