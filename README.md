# xwmt

**xWMT** is a Python package that provides a framework for calculating water mass tranformations in an xarray-based environment.

Quick Start Guide
-----------------

**Installing from scratch using `conda`**
This is the recommended mode of installation for developers.
```bash
git clone git@github.com:hdrake/xwmt.git
cd xwmt
conda env create -f ci/environment.yml
conda rename -n testing xwmt
conda activate xwmt
conda install -c conda-forge jupyterlab matplotlib
pip install -e .
```
You can verify that the package was properly installed by confirming it passes all of the tests with:
```bash
pytest
```
You can launch a Jupyterlab instance using this environment with:
```bash
python -m ipykernel install --user --name xwmt --display-name "xwmt"
jupyter-lab
```

**Minimal installation within an existing environment**
```bash
pip install git+https://github.com/hdrake/xwmt.git@main
```
