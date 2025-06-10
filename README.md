# xwmt

**xWMT** is a Python package that provides a framework for calculating water mass tranformations in an xarray-based environment.

Quick Start Guide
-----------------

**Minimal installation within an existing environment**
```bash
pip install git+https://github.com/NOAA-GFDL/xwmt.git@main
```

**Installing from scratch using `conda`**

This is the recommended mode of installation for developers.
```bash
git clone git@github.com:NOAA-GFDL/xwmt.git
cd xwmt
conda env create -f docs/environment.yml
conda activate docs_env_xwmt
pip install -e .
```

You can verify that the package was properly installed by confirming it passes all of the tests with:
```bash
pytest -v
```

You can launch a Jupyterlab instance using this environment with:
```bash
python -m ipykernel install --user --name docs_env_xwmt --display-name "docs_env_xwmt"
jupyter-lab
```

