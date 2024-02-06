# PZ Server pipelines

Repository to host the PZ Server's pipelines: 

### Combine Spec-z Catalogs (to do)
> Combine multiple spectroscopic redshift catalogs into a single sample with homogenized data formats and a unique system of quality flags translated from the survey's original files.  

### Training Set Maker (in progress)
> Create customized training and validation/test sets using a compilation of spectroscopic redshifts and LSST photometric data.

This repository currently contains a basic dataset, for testing purposes only. The ideal is to connect the pipelines to systems with access to a larger datasets.

## Install

The only requirement is to have miniconda or anaconda previously installed and . And run `setup.sh`:

```bash
git clone https://github.com/linea-it/pzserver_pipelines && cd pzserver_pipelines
./setup.sh
source env.sh
```

The `setup.sh` will suggest a directory where the pipelines and datasets are installed, type 'yes' to confirm or 'no' to configure the desired path in each case with the respective environment variables and then run again `setup.sh`.


## Run a pipeline

Currently the repository has two example pipelines: cross_lsdb and hello_parsl and to execute:

```bash
# execute cross_lsdb
cd $PIPELINES_DIR/cross_lsdb
bash run.sh config.yml
```

```bash
# execute hello_parsl
cd $PIPELINES_DIR/hello_parsl
bash run.sh config.yml
```

Software developed and delivered as part of the in-kind contribution program BRA-LIN, from LIneA to the Rubin Observatory's LSST. An overview of this and other contributions is available [here](https://linea-it.github.io/pz-lsst-inkind-doc/).  


