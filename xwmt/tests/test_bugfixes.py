"""
Regression tests for bugs fixed in the Tier-1 review pass. Each test pins a code
path that previously raised a NameError/TypeError, validated incorrectly, or was
silently ignored. These paths build a tiny single-column synthetic grid so they
run without the downloaded Baltic dataset.
"""
import warnings

import numpy as np
import pytest
import xarray as xr
import xgcm

import xwmt


def minimal_grid():
    """A tiny single-column grid with thickness, a tracer, an area metric, and a mask."""
    ds = xr.Dataset()
    ds = ds.assign_coords({
        'z_i': xr.DataArray(np.array([0., 1.]), dims=("z_i",)),
        'z_l': xr.DataArray(np.array([0.5]), dims=("z_l",)),
    })
    ds['dz'] = xr.DataArray(np.ones((1,)), dims=("z_l",))
    ds['temperature'] = xr.DataArray(np.array([2.0]), dims=("z_l",))
    ds['so'] = xr.DataArray(np.array([35.0]), dims=("z_l",))
    ds = ds.expand_dims(dim=('x', 'y')).assign_coords({
        'x': xr.DataArray([1.], dims=('x',)),
        'y': xr.DataArray([1.], dims=('y',)),
    })
    ds = ds.assign_coords({
        'rA': xr.DataArray([[1.]], dims=('x', 'y')),
        'lat': xr.DataArray([[1.]], dims=('x', 'y')),
    })
    ds['wet'] = xr.DataArray([[1.]], dims=('x', 'y'))
    coords = {'X': {'center': 'x'}, 'Y': {'center': 'y'}, 'Z': {'center': 'z_l', 'outer': 'z_i'}}
    metrics = {('X', 'Y'): ['rA']}
    return xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=False, autoparse_metadata=False)


FULL_BUDGET = {
    "mass": {"lambda": None, "thickness": "dz", "lhs": {}, "rhs": {}},
    "heat": {"lambda": "temperature", "lhs": {}, "rhs": {}},
    "salt": {"lambda": None, "lhs": {}, "rhs": {}},
}


@pytest.fixture
def grid():
    return minimal_grid()


def test_invalid_method_raises(grid):
    # B10: an unsupported `method` should fail fast with a clear ValueError.
    with pytest.raises(ValueError, match="method"):
        xwmt.WaterMassTransformations(grid, FULL_BUDGET, method="bogus")


def test_missing_thickness_raises(grid):
    # B2: `mass` present but no `thickness` must raise (previously fell through silently).
    budget = {"mass": {}, "heat": {"lambda": "temperature", "lhs": {}, "rhs": {}}}
    with pytest.raises(ValueError, match="thickness"):
        xwmt.WaterMassTransformations(grid, budget)


def test_zonal_mean_runs(grid):
    # B1: zonal_mean previously raised NameError (bare `grid` / `landmask_name`).
    wm = xwmt.WaterMass(grid, t_name="temperature", h_name="dz")
    result = wm.zonal_mean(wm.grid._ds["temperature"])
    assert np.isfinite(float(result.isel(y=0, z_l=0).values))


def test_get_density_invalid_name_raises(grid):
    # B5: a malformed density_name should raise instead of printing and continuing.
    wm = xwmt.WaterMass(grid, t_name="temperature", h_name="dz")
    with pytest.raises(ValueError, match="density_name"):
        wm.get_density("sigmaX")


def test_process_names_unknown_tracer(grid):
    # B8: process_names for an undefined tracer should return (None, None), not crash.
    wmt = xwmt.WaterMassTransformations(grid, FULL_BUDGET)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert wmt.process_names("not_a_tracer", "someterm") == (None, None)


def test_grid_not_mutated_by_construction(grid):
    # B1-adjacent / non-mutation improvement: constructing a WaterMass must not add
    # derived variables (z, z_interface, dz_i) to the caller's dataset.
    before = set(grid._ds.variables)
    xwmt.WaterMass(grid, t_name="temperature", h_name="dz")
    assert set(grid._ds.variables) == before
