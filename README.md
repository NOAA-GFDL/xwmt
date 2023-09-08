# xwmt

**xWMT** is a Python package that provides a framework for calculating water mass tranformations in an xarray-based environment.

Quick Start Guide
-----------------

**Installing from scratch using `conda`**
```bash
git clone git@github.com:hdrake/xwmt.git
cd xwmt
conda env create -f ci/environment.yml
conda activate testing
pip install -e .
python -m ipykernel install --user --name xwmt --display-name "xwmt"
jupyter-lab
```

**Minimal installation within an existing environment**
```bash
pip install git+https://github.com/hdrake/xwmt.git@main
```
