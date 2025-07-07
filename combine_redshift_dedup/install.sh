#!/bin/bash --login

source $(conda info --base)/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }
conda activate pipe_crd || { echo "Failed to activate pipe_crd"; exit 1; }

if [ ! -d "$PIPELINES_DIR" ]; then
    echo "Error: PIPELINES_DIR not defined."
    exit 1
fi

PIPE_BASE="$PIPELINES_DIR/combine_redshift_dedup"
HASENV=`conda env list | grep pipe_crd`

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

conda activate pipe_crd

# PATH export no longer needed since we're calling python explicitly
# export PATH=$PATH:"$PIPE_BASE/scripts/"

# Set PYTHONPATH so Python can find combine_redshift_dedup
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$PIPELINES_DIR"
else
    export PYTHONPATH="$PIPELINES_DIR:$PYTHONPATH"
fi

echo "Conda Environment: $CONDA_DEFAULT_ENV"

