# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package does

`xwmt` computes **water mass transformations (WMT)** for ocean models in an xarray-based
environment. WMT is the diapycnal (cross-isosurface) rate at which water is converted from
one value of a tracer (`lambda`, e.g. temperature, salinity, or potential density) to another,
driven by tendency processes (surface forcing, mixing, etc.). The core operation is: take a
3D tracer-tendency field, then bin/integrate it across a target tracer coordinate ("transform
into lambda space") along the vertical, optionally integrating over the horizontal.

Conceptually this implements the Walin (1982) framework. See `docs/source/objective.rst` for
references. A companion package `xwmb` (github.com/hdrake/xwmb) handles the remaining water
mass budget terms; the budget-variable naming conventions come from the `xbudget` package.

Important domain caveat (from the docs): `xwmt` does **not** check conservation of heat/salt.
Garbage-in/garbage-out — input budgets must already be closed by the user.

## Commands

```bash
# Create the dev environment (conda recommended for developers)
conda env create -f docs/environment.yml      # full env (docs_env_xwmt)
# or the lighter CI env:
conda env update -f ci/environment.yml         # test_env_xwmt
pip install -e .

# Run the test suite (downloads ~Baltic test NetCDF from GFDL FTP on first run)
pytest -v

# Run a single test / module
pytest xwmt/tests/test_integrate_transformations.py -v
pytest xwmt/tests/test_integrate_transformations.py::test_functional_3d_theta -v

# Lint / format (configs: .pylintrc, black is in the env)
pylint xwmt
black xwmt
```

Note: `pytest` requires network access on first run — `conftest.py` downloads
`xwmb_test_data_Baltic_3d.20230830.nc` from the GFDL FTP server into the working directory.

## Architecture

Three source modules under `xwmt/`, plus tests:

- **`wm.py` — `WaterMass`**: the base class. Wraps an `xgcm.Grid` and handles all the
  *grid/thermodynamic* machinery that is independent of tendencies:
  - `__init__` deep-copies the input grid (never mutating the caller's) and delegates to
    `_build_vertical_metrics` / `_interpolate_thickness_to_interfaces` / `_compute_depth_coordinates`.
    It populates `self.Z_metrics` (cell-center and interface thicknesses — an **instance**
    attribute, not attached to the grid) and derives `z` (layer-center depth) and `z_interface`.
    There's a special path for purely 2D/surface data with no `Z` axis (synthesizes a single-layer Z).
  - `get_density()` derives `alpha`, `beta`, and density variables (`rho`, `sigma0`–`sigma4`)
    via the TEOS-10 `gsw` package, converting temperature/salinity to conservative/absolute
    forms as needed (controlled by `t_var`/`s_var`). It memoizes into `grid._ds` — re-calls are
    cheap and skip already-present variables.
  - Outcrop helpers (`get_outcrop_lev`, `sel_outcrop_lev`, `expand_surface_array_vertically`)
    locate/operate on the first "real" (thick enough) vertical level from the surface or floor —
    used to place surface fluxes into the correct outcropping layer.
  - `add_gridcoords()` and `_rebuild_grid()` (module-level) reconstruct an `xgcm.Grid` from an
    existing one, preserving coords/metrics/boundary (and optionally adding more). `_rebuild_grid`
    is the single source of truth for grid reconstruction, used by both `__init__` and `add_gridcoords`.

- **`compute.py`**: small functional helpers for converting interfacial fluxes (`Jlam`) or
  layer-integrated tendencies into `hlamdot` — the **vertically-extensive tracer tendency**
  (`h * lambda_dot`), the central quantity that gets transformed. These are pure functions:
  the cell-center thickness metric is passed in via `h=` (callers pass `self.Z_metrics["center"]`).

- **`wmt.py` — `WaterMassTransformations(WaterMass)`**: the main user-facing class. Adds the
  transformation logic on top of `WaterMass`. Driven by a `recipe` (a nested dict from
  the `xbudget` package, e.g. its `MOM6.yaml` preset) that maps tracers → lambda variable names
  and tendency "processes" → dataset variable names. The constructor argument was called
  `xbudget_dict` before xbudget 0.7.0 renamed the concept; that keyword (and the
  `.xbudget_dict` attribute) still work but emit a `FutureWarning`.

### Key data flow in `wmt.py`

The pipeline, from low to high level (each calls the one above it):

1. `process_names` / `datadict` — resolve a `(tracer, term)` pair to the actual dataset
   variables, and classify a tendency as either a `layer_integrated_tendency` (lives on
   Z-centers) or an `interfacial_flux` (lives on Z-interfaces).
2. `rho_tend` / `calc_hlamdot_and_lambda` — produce `hlamdot` and the scalar lambda field for a
   given `(lambda_name, term)`. For density lambdas, heat and salt contributions are computed
   **separately** (via `alpha`/`cp` and `beta`) and returned as a dict keyed `"heat"`/`"salt"`.
3. `transform_hlamdot_term` — the heart of the package: bins `hlamdot` into lambda space via the
   shared `_transform_one` kernel, using either `xhistogram` (area-integrated path) or
   `xgcm.transform` conservative remapping (column-wise path). The `method` (`"default"`/
   `"xhistogram"`/`"xgcm"`) selects which; `"default"` picks xhistogram when `integrate=True`,
   xgcm otherwise. The resolved method is kept **local to the call** (a prebinned target forces
   xgcm without corrupting `self.method`). Handles "prebinned" data where lambda is already a
   vertical coordinate (skips re-binning unless `rebin=True`). Density components are named via
   the `_COMPONENTS`/`_component_name` helpers (the `"_heat"`/`"_salt"` suffix convention).
   Finally, if `N_min` is set, `_mask_undersampled_bins` overwrites bins sampled by too few
   grid cells with `fill_value` (see below).
4. `transformations_from_hlamdot` — loops over process terms and merges results.
5. `map_transformations` (column-wise, `integrate=False`) and `integrate_transformations`
   (horizontally integrated, `integrate=True`) — the two top-level entry points. Both optionally
   sum heat+salt components (`sum_components`) and group process terms into kinematic vs material
   transformation categories (`group_processes`, via `_sum_components`/`_group_processes`).

### `N_min` masking of under-resolved bins

`count_cells_per_bin` histograms the *lambda field itself* (unweighted) into the target bins,
counting grid cells that are non-NaN, of nonzero thickness, and inside `mask`. `N_min` (settable
on the constructor and overridable per call on `transform_hlamdot_term`,
`transformations_from_hlamdot`, `map_transformations`, `integrate_transformations`) then
overwrites bins whose count is below it with `fill_value` (`0.0` by default; `np.nan` opts into
gaps). Points to keep in mind when editing:

- The count is **term-independent** by design, so the same bins are masked for every process and
  masked bins stay masked through `_sum_terms`/`_group_processes` (NaN propagates through the sums).
- The count is taken over `[*_horizontal_dims, _zc]` when `integrate=True` and over `[_zc]` alone
  when `integrate=False`, so `N_min` means something very different in the two cases.
- `_lambda_field` deliberately does *not* vertically expand a surface-only lambda: the expansion
  zero-fills non-outcropping layers, which is harmless for the transformation (their weights are
  zero) but would put those zeros in the census.
- The target coordinate is always `f"{lam_var}_l_target"`, matching `_transform_one` for both the
  prebinned and non-prebinned paths.
- `fill_value` defaults to `0.0`, **not** `np.nan`, and this is deliberate: results are almost
  always reduced further (`.mean("time")`, spatial means over `map_transformations` output) and
  xarray reductions skip NaN, so a NaN fill would silently average a partly-masked bin over only
  the samples that survived masking while still looking like a full mean. `0.0` also preserves the
  pre-existing value of empty bins. `test_default_fill_value_keeps_time_means_honest` pins this.
- `fill_value` uses `None` as its per-call "inherit the instance value" sentinel rather than
  defaulting to a literal at the call sites, because `np.nan != np.nan` makes an explicitly-passed
  `np.nan` indistinguishable from a default. `_validate_fill_value` therefore rejects `None`.
- Default is `N_min=None`, i.e. no masking and byte-identical results — keep it that way, since
  the functional golden-value tests assume it.

`lambda_name` can be a tracer key (`"heat"`, `"salt"`), a density name (`"sigma0"`…`"sigma4"`),
or any custom tracer in the budget dict. `term` is a process key; `available_processes()` lists
the ones actually present in the dataset.

### Sign conventions & units (easy to get wrong)

- Salt tendencies are multiplied by 1000 (kg→g) in `datadict`.
- `transformations_from_hlamdot` negates the transformed tendency (`-transformed_hlamdot`):
  transformation is defined as convergence into a layer.
- Density tendencies are scaled by `rho_ref` (Boussinesq assumption — see the inline
  "Is this correct for non-boussinesq case?" note in `calc_hlamdot_and_lambda`).
- Heat tendencies are divided by `cp` (specific heat) to get a temperature-like tendency.

## Tests

- `test_integrate_transformations.py` — "functional" regression tests that run the full pipeline
  on real MOM6 Baltic data and assert against hardcoded numerical answers (separate expected
  arrays for `xgcm` vs `xhistogram` methods). If you change the numerics, these golden values
  must be regenerated deliberately.
- `test_convergence_to_analytical.py` + `conftest.py::Helpers` — build idealized synthetic grids
  with analytically-known transformations and check that xwmt converges to the exact answer as
  resolution increases. This is where to validate any change to the core transformation math.
- `test_bugfixes.py` — fast unit/regression tests on a tiny synthetic grid (no data download)
  that pin previously-broken paths (input validation, constructor mask, prebinned method
  non-mutation, grid non-mutation).
- `test_bin_masking.py` — fast unit tests (no data download) for `count_cells_per_bin` and
  `N_min`, on tiny grids whose per-bin counts are known exactly. Covers both transformation
  backends, the prebinned/surface-only/multi-tile lambda paths, and laziness.
- Baltic test data is fetched by the `baltic_dataset_path` / `baltic_grid_and_budgets` session
  fixtures in `conftest.py` (HTTPS-first, checksum-pinned, skips cleanly when offline). `*.nc`
  is gitignored. The pinned `DATA_SHA256` must be updated if the dataset is intentionally changed.

## Versioning

**The git tag is the single source of truth.** `hatch-vcs`
(`[tool.hatch.version] source = "vcs"`) derives the version from the tag at build time
and writes it to `xwmt/_version.py`, which is **gitignored** — there is no version
string in the source tree. `xwmt/version.py` is a thin shim that imports from it, with
a `0.0.0+unknown` fallback for an un-built checkout.

Consequences worth remembering when editing:

- **Never add a version literal back to the tree**, and never "fix" a `0.0.0+unknown`
  by hardcoding one — it means the package was imported without being built or installed.
- **Any CI job that installs the package needs `fetch-depth: 0`.** A shallow clone cannot
  see the tag, so hatch-vcs silently resolves a `0.1.devN` version instead of failing. The
  `build` checkout in `ci.yml` and the one in `python-publish.yml` set it, and
  `.readthedocs.yaml` unshallows in `post_checkout` for the same reason. The same applies
  to a `pip install git+https://…` of a fork with no tags.
- `_version.py` **is** shipped inside the sdist, so building from the sdist (as
  conda-forge does) works with no git present. Do not add it to
  `[tool.hatch.build] exclude`.
- The conda-forge feedstock builds with `--no-build-isolation`, so its `host`
  requirements must list `hatch-vcs` next to `hatchling`.
- Releasing is just publishing a GitHub Release tagged `vX.Y.Z`; there is no bump commit.
  See "Releasing" in `README.md`.

## Conventions

- Everything stays **lazy/dask-friendly**: computations use `xr.apply_ufunc(..., dask="parallelized")`
  and avoid forcing computation. Preserve laziness when editing.
- Derived state (`p`, `sa`, `ct`, `alpha`, `beta`, density, `z`, `z_interface`, `{h}_i`) is
  accumulated onto the WaterMass's **own deep copy** `self.grid._ds`; the caller's grid is never
  mutated. Many methods are idempotent via `if "<var>" not in self.grid._ds` guards. (A future
  major version may move this derived state off `grid._ds` entirely — see the review notes.)
- `Z_metrics` (center/interface thickness) lives on the `WaterMass` **instance** as
  `self.Z_metrics`, not on the `xgcm.Grid`. Grid reconstruction goes through `_rebuild_grid`.
- Use the coordinate-name properties (`self._zc`, `self._zi`, `self._xc`, `self._yc`,
  `self._horizontal_dims`) instead of re-deriving `self.grid.axes['Z'].coords['center']` inline.
- **Multi-tile grids** (e.g. ECCOv4r4's 13-tile lat-lon-cap grid): when the input
  `xgcm.Grid` carries `face_connections`, `self._facedim` is the tile-dimension name
  (via xgcm's `_facedim`) and `self._horizontal_dims` prepends it, so horizontal
  integrations/binning broadcast across tiles automatically. `_rebuild_grid` must
  preserve `face_connections` or the face dim is silently dropped on the deep copy
  (issue #59). xwmt does no *horizontal* interpolation (all `grid.transform`/`interp`/
  `cumsum`/`diff` are along `Z`), so no face-connectivity math is needed — only the
  reduction dims. The heavy real-data check lives in `test_multitile_ecco.py`, gated
  off CI behind `XWMT_TEST_ECCO`; the fast synthetic 2-tile guard is in `test_bugfixes.py`.
