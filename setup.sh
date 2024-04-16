#!/bin/bash

export APP_DIR=`pwd`

if [ ! -d "$PIPELINES_DIR" ]; then
    export PIPELINES_DIR=$APP_DIR
fi

if [ ! -d "$DATASETS_DIR" ]; then
    export DATASETS_DIR=$APP_DIR/data-example
fi

echo "The directory containing the pipelines is: " $PIPELINES_DIR
read -p "Enter 'yes' to confirm the installation or 'no' to exit and reconfigure the variable PIPELINE_DIR " pinput

if [ "$pinput" != "yes" ]; then
	echo Exiting...
	exit 1
fi

echo "The directory containing the datasets is: " $DATASETS_DIR
read -p "Enter 'yes' to confirm the installation or 'no' to exit and reconfigure the variable DATASETS_DIR " dinput

if [ "$dinput" != "yes" ]; then
	echo Exiting...
	exit 1
fi

sed "s|<PIPELINES_DIR>|$PIPELINES_DIR|g" $PIPELINES_DIR/pipelines.template.yml > $PIPELINES_DIR/pipelines.yml
sed "s|<DATASETS_DIR>|$DATASETS_DIR|g" $DATASETS_DIR/datasets.template.yml > $DATASETS_DIR/datasets.yml

sed "s|<APP_DIR>|$APP_DIR|g" $APP_DIR/env.template.sh > $APP_DIR/env.sh
sed -i'' -e "s|<DATASETS_DIR>|$DATASETS_DIR|g" $APP_DIR/env.sh
sed -i'' -e "s|<PIPELINES_DIR>|$PIPELINES_DIR|g" $APP_DIR/env.sh

source `dirname $CONDA_EXE`/activate || { echo "Failed to activate Conda environment"; exit 1; }

HASENV=`conda env list | grep pz_pipelines`

if [ -z "$HASENV" ]; then
    echo "Create virtual environment..."
    conda env create -f environment.yml
    echo "Virtual environment created and packages installed."
else
    if [ "$CONDA_FORCE_UPDATE" == "yes" ]; then
        echo "Virtual environment already exists. Updating..."
        conda env update --file environment.yml --prune
    fi
fi

conda activate pz_pipelines
echo "Conda Environment: $CONDA_DEFAULT_ENV"
