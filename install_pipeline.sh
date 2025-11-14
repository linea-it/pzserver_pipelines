#!/bin/bash

if [ ! -d "$PIPELINES_DIR" ]; then
    export PIPELINES_DIR=$APP_DIR
fi

echo "The directory containing the pipelines is: " "$PIPELINES_DIR"


# shellcheck disable=SC1091
# shellcheck disable=SC2086
source "$(dirname $CONDA_EXE)/activate" || { echo "Failed to activate Conda environment"; exit 1; }

# shellcheck disable=SC1091
# shellcheck disable=SC2086
. ${PIPELINES_DIR}/combine_redshift_dedup/install.sh || { echo "Failed to source combine_redshift_dedup/install.sh"; exit 1; }
