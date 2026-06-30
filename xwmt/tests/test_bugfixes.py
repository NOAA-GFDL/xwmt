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
    ds = ds.assign_coords(
        {
            "z_i": xr.DataArray(np.array([0.0, 1.0]), dims=("z_i",)),
            "z_l": xr.DataArray(np.array([0.5]), dims=("z_l",)),
        }
    )
    ds["dz"] = xr.DataArray(np.ones((1,)), dims=("z_l",))
    ds["temperature"] = xr.DataArray(np.array([2.0]), dims=("z_l",))
    ds["so"] = xr.DataArray(np.array([35.0]), dims=("z_l",))
    ds = ds.expand_dims(dim=("x", "y")).assign_coords(
        {
            "x": xr.DataArray([1.0], dims=("x",)),
            "y": xr.DataArray([1.0], dims=("y",)),
        }
    )
    ds = ds.assign_coords(
        {
            "rA": xr.DataArray([[1.0]], dims=("x", "y")),
            "lat": xr.DataArray([[1.0]], dims=("x", "y")),
        }
    )
    ds["wet"] = xr.DataArray([[1.0]], dims=("x", "y"))
    coords = {
        "X": {"center": "x"},
        "Y": {"center": "y"},
        "Z": {"center": "z_l", "outer": "z_i"},
    }
    metrics = {("X", "Y"): ["rA"]}
    return xgcm.Grid(
        ds, coords=coords, metrics=metrics, periodic=False, autoparse_metadata=False
    )


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


def _heat_tendency_grid(zname="z"):
    """Tiny single-column grid carrying a heat tendency, for transformation tests.

    With ``zname="temperature"`` the vertical coordinate *is* the lambda variable
    (``temperature_l``/``temperature_i``), which exercises the "prebinned" code path.
    """
    Nz = 8
    edges = np.linspace(0.0, 1.0, Nz + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ci, cl = f"{zname}_i", f"{zname}_l"
    ds = xr.Dataset()
    ds = ds.assign_coords(
        {
            ci: xr.DataArray(edges, dims=(ci,)),
            cl: xr.DataArray(centers, dims=(cl,)),
        }
    )
    ds = ds.assign_coords({"dz": xr.DataArray(np.diff(edges), dims=(cl,))})
    ds["temperature"] = xr.DataArray(centers, dims=(cl,))
    ds["heat_tendency"] = xr.DataArray(np.ones(Nz), dims=(cl,)) * ds.dz
    ds = ds.expand_dims(("x", "y")).assign_coords(
        {"x": xr.DataArray([1.0], dims=("x",)), "y": xr.DataArray([1.0], dims=("y",))}
    )
    ds = ds.assign_coords({"rA": xr.DataArray([[1.0]], dims=("x", "y"))})
    coords = {
        "X": {"center": "x"},
        "Y": {"center": "y"},
        "Z": {"center": cl, "outer": ci},
    }
    grid = xgcm.Grid(
        ds,
        coords=coords,
        metrics={("X", "Y"): ["rA"]},
        periodic=False,
        autoparse_metadata=False,
    )
    budget = {
        "mass": {"lambda": None, "thickness": "dz", "lhs": {}, "rhs": {}},
        "heat": {
            "lambda": "temperature",
            "lhs": {"tendency": "heat_tendency"},
            "rhs": {},
        },
        "salt": {"lambda": None, "lhs": {}, "rhs": {}},
    }
    return grid, budget, edges


def test_constructor_mask_is_applied():
    # B6: a mask passed to the constructor was previously stored nowhere and ignored.
    grid, budget, edges = _heat_tendency_grid()
    all_false = xr.DataArray([[False]], dims=("x", "y"))
    masked = xwmt.WaterMassTransformations(
        grid, budget, mask=all_false, cp=1.0, rho_ref=1.0, method="xhistogram"
    )
    full = xwmt.WaterMassTransformations(
        grid, budget, cp=1.0, rho_ref=1.0, method="xhistogram"
    )
    t_masked = masked.integrate_transformations(
        "heat", bins=edges, sum_components=False
    )
    t_full = full.integrate_transformations("heat", bins=edges, sum_components=False)
    # An all-False constructor mask zeros the weights -> transformation must vanish,
    # and must differ from the (non-trivial) unmasked result.
    assert np.allclose(np.nan_to_num(t_masked["tendency"].values), 0.0)
    assert np.any(np.abs(np.nan_to_num(t_full["tendency"].values)) > 0.0)


def test_method_not_mutated_by_prebinned_transform():
    # B3: a prebinned target used to set `self.method = "xgcm"`, silently corrupting the
    # method for all later calls. It must now be call-local; self.method is unchanged.
    grid, budget, edges = _heat_tendency_grid(zname="temperature")  # prebinned coords
    wmt = xwmt.WaterMassTransformations(
        grid, budget, cp=1.0, rho_ref=1.0, method="xhistogram"
    )
    wmt.integrate_transformations("heat", bins=edges, sum_components=False)
    assert wmt.method == "xhistogram"
