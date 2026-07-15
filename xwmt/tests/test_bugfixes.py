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
    """A tiny single-column grid with thickness, a tracer, and an area metric."""
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
    coords = {
        "X": {"center": "x"},
        "Y": {"center": "y"},
        "Z": {"center": "z_l", "outer": "z_i"},
    }
    metrics = {("X", "Y"): ["rA"]}
    return xgcm.Grid(
        ds, coords=coords, metrics=metrics, padding="fill", autoparse_metadata=False
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


def _surface_grid():
    """A tiny 2D surface grid (no Z axis), as used by the surface-WMT example notebooks."""
    ny, nx = 4, 5
    ds = xr.Dataset()
    ds = ds.assign_coords(
        {"x": np.arange(nx, dtype=float), "y": np.arange(ny, dtype=float)}
    )
    ds["tos"] = xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))
    ds["sos"] = xr.DataArray(35.0 * np.ones((ny, nx)), dims=("y", "x"))
    ds = ds.assign_coords(
        {"areacello": xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))}
    )
    return xgcm.Grid(
        ds,
        coords={"X": {"center": "x"}, "Y": {"center": "y"}},
        metrics={("X", "Y"): "areacello"},
        padding="fill",
        autoparse_metadata=False,
    )


def test_missing_mass_raises(grid):
    # B2: the budget must contain a `mass` entry.
    budget = {"heat": {"lambda": "temperature", "lhs": {}, "rhs": {}}}
    with pytest.raises(ValueError, match="mass"):
        xwmt.WaterMassTransformations(grid, budget)


def test_surface_wmt_without_thickness():
    # B2 regression: `thickness` is optional. A surface budget (`{"mass": {}}`) on a
    # 2D grid with no Z axis must construct, not raise (surface WMT needs no thickness).
    budget = {
        "mass": {},
        "heat": {"surface_lambda": "tos"},
        "salt": {"surface_lambda": "sos"},
    }
    wmt = xwmt.WaterMassTransformations(_surface_grid(), budget)
    assert wmt.tracer_dict["heat"] == "tos"


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
        padding="fill",
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


def _ranged_tracer_grid():
    """A single column whose tracer spans a real range, so bins are meaningful."""
    nz = 8
    ds = xr.Dataset()
    ds = ds.assign_coords(
        {
            "z_i": xr.DataArray(np.linspace(0.0, 1.0, nz + 1), dims=("z_i",)),
            "z_l": xr.DataArray(
                0.5
                * (
                    np.linspace(0.0, 1.0, nz + 1)[:-1]
                    + np.linspace(0.0, 1.0, nz + 1)[1:]
                ),
                dims=("z_l",),
            ),
        }
    )
    ds["dz"] = xr.DataArray(np.full((nz,), 1.0 / nz), dims=("z_l",))
    ds["temperature"] = xr.DataArray(np.linspace(0.0, 10.0, nz), dims=("z_l",))
    ds["so"] = xr.DataArray(np.full((nz,), 35.0), dims=("z_l",))
    ds = ds.expand_dims(dim=("x", "y")).assign_coords(
        {"x": xr.DataArray([1.0], dims=("x",)), "y": xr.DataArray([1.0], dims=("y",))}
    )
    ds = ds.assign_coords(
        {
            "rA": xr.DataArray([[1.0]], dims=("x", "y")),
            "lat": xr.DataArray([[1.0]], dims=("x", "y")),
        }
    )
    grid = xgcm.Grid(
        ds,
        coords={
            "X": {"center": "x"},
            "Y": {"center": "y"},
            "Z": {"center": "z_l", "outer": "z_i"},
        },
        metrics={("X", "Y"): ["rA"]},
        padding="fill",
        autoparse_metadata=False,
    )
    return grid


def test_infer_bins_accepts_dataarray():
    # `infer_bins` took the min/max of `da` as 0-d DataArrays and handed them to
    # `np.linspace`, which rejects them -- so the documented input (a DataArray)
    # always raised and only `da.values` worked.
    wm = xwmt.WaterMass(_ranged_tracer_grid(), t_name="temperature", h_name="dz")
    bins = wm.infer_bins(wm.grid._ds.temperature, nbins=11)
    assert bins.shape == (11,)
    assert np.isclose(bins[0], 0.0) and np.isclose(bins[-1], 10.0)


def test_infer_bins_percentiles_is_reachable():
    # The `percentiles` branch was unreachable on every input: a DataArray hit the
    # `np.linspace` bug above, and an ndarray has no `.quantile`.
    wm = xwmt.WaterMass(_ranged_tracer_grid(), t_name="temperature", h_name="dz")
    bins = wm.infer_bins(wm.grid._ds.temperature, percentiles=[0.25, 0.75], nbins=5)
    assert bins.shape == (5,)
    # Clipped to the interquartile range, so strictly inside the full 0-10 span.
    assert bins[0] > 0.0 and bins[-1] < 10.0


def test_integrate_transformations_default_bins():
    # `bins=None` is the documented default of both top-level entry points, and it
    # routes through `infer_bins` -- so the 0-d DataArray bug above meant the default
    # call signature raised for every user. Pin the default path itself, not just the
    # helper.
    grid, budget, _ = _heat_tendency_grid()
    wmt = xwmt.WaterMassTransformations(grid, budget, cp=1.0, rho_ref=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformations = wmt.integrate_transformations("heat", sum_components=False)
    assert "tendency" in transformations
