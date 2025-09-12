#!/bin/bash --login

# Initialize conda (robust, like CRC)
source "$(conda info --base)"/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }

# Check PIPELINES_DIR
if [ ! -d "$PIPELINES_DIR" ]; then
    echo "Error: PIPELINES_DIR not defined."
    exit 1
fi

# Accept Anaconda TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

PIPE_BASE="$PIPELINES_DIR/training_set_maker"

# Robust check: does env 'pipe_tsm' exist?
if conda env list | awk '{print $1}' | grep -qx 'pipe_tsm'; then
    HASENV=1
else
    HASENV=0
fi

if [ "$HASENV" -eq 0 ]; then
    echo "Create virtual environment..."
    conda env create -f "${PIPE_BASE}/environment.yaml"
    echo "Virtual environment created and packages installed."
# else
#     if [ "$CONDA_FORCE_UPDATE" = "yes" ]; then
#         echo "Virtual environment already exists. Updating..."
#         conda env update --file "${PIPE_BASE}/environment.yaml" --prune
#     fi
fi

# Activate env
conda activate pipe_tsm || { echo "Failed to activate pipe_tsm"; exit 1; }

# Export PATH for TSM scripts
export PATH="$PATH:${PIPE_BASE}/scripts/"

# PYTHONPATH so Python can find training_set_maker packages
if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${PYTHONPATH}:${PIPE_BASE}/packages/"
else
    export PYTHONPATH="${PIPE_BASE}/packages/"
fi

echo "Conda Environment: $CONDA_DEFAULT_ENV"

