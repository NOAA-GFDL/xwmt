# xwmt example notebooks

The notebooks are meant to be read in the order below: each one assumes the
previous ones, and they get progressively more realistic (and more expensive).

If you are new to `xwmt`, start with **`quickstart.ipynb`** — it needs no data
and runs in under a minute.

| Notebook | What it covers | Data | Network |
|---|---|---|---|
| `quickstart.ipynb` | Your first transformation curve, on a synthetic basin | none | no |
| `closed_transformation_budget.ipynb` | Closed, comprehensive MOM6 budgets and the `xbudget` dictionary | 1.08 GB | first run |
| `north_atlantic_deep_water.ipynb` | Case study: where and how North Atlantic Deep Water forms | 1.08 GB (shared with above) | first run |
| `swmt_decomposition.ipynb` | Surface transformation when you only have surface fields | 1.20 GB | first run |
| `bring_your_own_model.ipynb` | Using `xwmt` with a model that isn't MOM6 | none | no |
| `swmt_from_cmip.ipynb` | Surface transformation from the CMIP6 archive | ~400 MB streamed | yes, every run |

## Data

The MOM6 example data come from [Zenodo record 15420739](https://zenodo.org/record/15420739)
and are downloaded on demand by `load_example_model_grid.py` into the repository's
`data/` directory. They are downloaded once and reused, and verified against the
checksums published in the Zenodo record. `data/` is gitignored.

The two MOM6 files are ~1 GB each, so the first run of a notebook that needs one
will take a while. `north_atlantic_deep_water.ipynb` and
`closed_transformation_budget.ipynb` share the same file, so running both costs
one download rather than two.

`swmt_from_cmip.ipynb` streams from the public Pangeo CMIP6 archive on Google
Cloud Storage rather than downloading a file, so it needs network access every
time it runs.

## Environment

```bash
conda env create -f ../docs/environment.yml
conda activate docs_env_xwmt
pip install -e ..
```

Note that `xwmt` requires `xgcm >= 0.10`, whose `Grid` takes `padding=` (older
versions took `boundary=`).

## A note on the synthetic notebooks

`quickstart.ipynb` and `bring_your_own_model.ipynb` build their own data via
`synthetic_ocean.py`. The basin is analytic, not model output — it exists so the
notebooks are instant and always reproducible. It is shaped to look like the real
thing (warm salty subtropics, cold strongly-cooled subpolar gyre) so the
transformation curves are physically recognizable, but do not read any science
into the numbers themselves.
