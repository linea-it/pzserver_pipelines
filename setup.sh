#!/bin/bash

export APP_DIR=`pwd`

if [ -z "$PIPELINES_DIR" ]; then
    export PIPELINES_DIR=$APP_DIR
fi

if [ -z "$DATASETS_DIR" ]; then
    export DATASETS_DIR=$APP_DIR/data-example
fi

echo "The directory containing the pipelines is: " "$PIPELINES_DIR"
read -p "Enter 'yes' to confirm the installation or 'no' to exit and reconfigure the variable PIPELINES_DIR " pinput

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

pipe="${PIPELINES_DIR}/combine_redshift_dedup/config.template.yaml"
DIRCONF=$(dirname "$pipe")
echo "$DIRCONF"
sed "s|<DATASETS_DIR>|$DATASETS_DIR|g" "$pipe" > "$DIRCONF/config.yaml"

sed "s|<APP_DIR>|$APP_DIR|g" $APP_DIR/env.template.sh > $APP_DIR/env.sh
sed -i'' -e "s|<DATASETS_DIR>|$DATASETS_DIR|g" $APP_DIR/env.sh
sed -i'' -e "s|<PIPELINES_DIR>|$PIPELINES_DIR|g" $APP_DIR/env.sh
