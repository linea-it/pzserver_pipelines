# PZ Server pipelines

Repository to host the PZ Server's pipelines: 

### Combine Spec-z Catalogs (to do)
> Combines multiple spectroscopic redshift catalogs into a single sample with homogenized data formats and a unique system of quality flags translated from the survey's original files.  

### Training Set Maker (in progress)
> Creates customized training and validation/test sets using a compilation of spectroscopic redshifts and LSST photometric data.


## Acknowledgements

Software developed and delivered as part of the in-kind contribution program BRA-LIN, from LIneA to the Rubin Observatory's LSST. An overview of this and other contributions is available [here](https://linea-it.github.io/pz-lsst-inkind-doc/). The pipelines take advantage of the software support layer developed by LINCC, available as Python libraries: [hipscat](https://github.com/astronomy-commons/hipscat), [hipscat-import](https://github.com/astronomy-commons/hipscat-import) and [lsdb](https://github.com/astronomy-commons/lsdb).    

## Test data

This repository currently contains a basic dataset, for testing purposes only. The ideal is to connect the pipelines to systems with access to a larger datasets.

## Install

The only requirement is to have miniconda or anaconda previously installed:

```bash
git clone https://github.com/linea-it/pzserver_pipelines && cd pzserver_pipelines
./setup.sh
source env.sh
```

The `setup.sh` will suggest a directory where the pipelines and datasets are installed, type 'yes' to confirm or 'no' to configure the desired path in each case with the respective environment variables and then run again `setup.sh`.


## Run a pipeline

Currently the repository has two example pipelines: cross_lsdb and hello_parsl. To execute them, simply:

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


