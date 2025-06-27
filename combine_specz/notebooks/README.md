# Conda Environment and Kernel Configuration for Open OnDemand (LIneA) – Pipeline Combine Spec-z

Author: Luigi Silva
Last reviewed: Jun. 18, 2025

## Accessing Open OnDemand
To access the Open OnDemand platform, follow these steps:

1. Go to: [ondemand.linea.org.br](https://ondemand.linea.org.br)
2. On the main page, click: **Clusters -> LIneA Shell Access**

---

## Setting Up the Environment via LIneA Shell Access

### 1. Download and install Miniconda in your `$SCRATCH` directory
```bash
cd $SCRATCH
curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -p $SCRATCH/miniconda
source miniconda/bin/activate
conda deactivate  # Deactivate base environment
```

### 2. Create the Conda environment
Before proceeding, it is recommended to configure Conda to use the following channels:
```bash
conda config --add channels conda-forge
conda config --add channels defaults
conda config --add channels anaconda
conda config --add channels pyviz
```
We also recommend installing and enabling the libmamba solver for faster environment resolution:
https://conda.github.io/conda-libmamba-solver/user-guide/

Two YAML files are available depending on your needs:
* `environment-short.yaml` → minimal environment with core libraries and fixed versions.
* `environment-complete.yaml` → full reproducible environment matching the development setup exactly.

To create the environment (example using the short version):

```bash
conda env create -p $SCRATCH/combine_specz_env -f environment-short.yaml
```

To activate the environment:

```bash
conda activate $SCRATCH/combine_specz_env
```

### 3. Set up the Jupyter kernel
With the environment activated:

```bash
JUPYTER_PATH=$SCRATCH/.local
export JUPYTER_PATH
python -m ipykernel install --prefix=$JUPYTER_PATH --name 'combine_specz_env'
```

This makes the kernel available in JupyterLab on Open OnDemand.

## Running the Pipeline Combine Spec-z
Once the environment and kernel are set up, follow these steps:

### 1. Generate Mock Spectroscopic Data
Open and run the notebook:
`generating-mock-specz-data.ipynb`
This will create three synthetic spectroscopic datasets under your `$SCRATCH` directory.

### 2. Configure `config.yaml`
Before running the main pipeline, edit the file config.yaml. Update all paths to reflect your directories in `$SCRATCH`.

Ensure that the paths to the three mock Parquet catalogs are correctly listed under the inputs section.

### 3. Execute the main pipeline notebook
Open the notebook:
`combine-specz-pipeline.ipynb`

At the top of the notebook, adjust:
* `path_to_yaml_file`: absolute path to your edited config.yaml
* any output or validation paths as needed in later cells

Then execute the notebook to run the full combine-and-deduplicate pipeline.

### Final Note
After completing the steps above, the combine_specz_env kernel will be available in JupyterLab. Simply select it in the interface and run your notebooks.

---

## References
* https://docs.linea.org.br/processamento/uso/openondemand.html