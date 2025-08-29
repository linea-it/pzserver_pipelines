# PZ Server pipelines

Repository to host the PZ Server's pipelines:

### Combine Redshift Catalogs 
> Combines multiple reference redshift catalogs into a single sample with homogenized data formats and a unique system of quality flags translated from the survey's original files.

### Training Set Maker 
> Creates customized training and validation/test sets using a compilation of spectroscopic redshifts and LSST photometric data.


## Acknowledgements

Software developed and delivered as part of the in-kind contribution program BRA-LIN, from LIneA to the Rubin Observatory's LSST. An overview of this and other contributions is available [here](https://linea-it.github.io/pz-lsst-inkind-doc/). The pipelines take advantage of the software support layer developed by LINCC, available as Python libraries: [hats](https://github.com/astronomy-commons/hats), [hats-import](https://github.com/astronomy-commons/hats-import) and [lsdb](https://github.com/astronomy-commons/lsdb).


## Tests 

### Test data

This repository currently contains a basic dataset, for testing purposes only. The ideal is to connect the pipelines to systems with access to a larger datasets.

### Install 

The only requirement is to have miniconda or anaconda previously installed:

```bash
git clone https://github.com/linea-it/pzserver_pipelines && cd pzserver_pipelines
./setup.sh
source env.sh
```

To install all pipelines at once: 

```
./install_pipelines.sh   
```

Alternatively, to install a single one: 

```
./<pipeline dir>/install.sh  
```

The `setup.sh` will suggest a directory where the pipelines and datasets are installed, type 'yes' to confirm or 'no' to configure the desired path in each case with the respective environment variables and then run again `setup.sh`.

The installation script creates new conda environments `pipe_crd` and `pipe_tsm`. 

## Run a pipeline

To execute, simply:

```bash
# execute combine redshift catalogs 
cd $PIPELINES_DIR/combine_redshift_dedup 
./run.sh config.yaml process001
```

```bash
# execute training set maker
cd $PIPELINES_DIR/training_set_maker
mkdir process001
./run.sh config.yaml process001
```


## Validation notebook 

To validate your test results, use the notebook `validation.ipynb`: 

```
conda install -c conda-forge jupyterlab ipykernel
python -m ipykernel install --user --name=pipe_crd
jupyter lab
```


