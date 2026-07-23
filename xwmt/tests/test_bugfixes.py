"""
Regression tests for bugs fixed in the Tier-1 review pass. Each test pins a code
path that previously raised a NameError/TypeError, validated incorrectly, or was
silently ignored. These paths build a tiny single-column synthetic grid so they
run without the downloaded Baltic dataset.
"""

import copy
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


# -- `xbudget_dict` -> `recipe` rename: the deprecation shim -----------------


def test_recipe_positional_still_works(grid):
    """The historical positional call is unaffected by the rename."""
    wmt = xwmt.WaterMassTransformations(grid, FULL_BUDGET)
    assert wmt.recipe["heat"]["lambda"] == "temperature"


def test_deprecated_xbudget_dict_kwarg_warns(grid):
    with pytest.warns(FutureWarning, match="xbudget_dict"):
        wmt = xwmt.WaterMassTransformations(grid, xbudget_dict=FULL_BUDGET)
    assert wmt.recipe["heat"]["lambda"] == "temperature"


def test_recipe_and_xbudget_dict_together_raises(grid):
    with pytest.raises(TypeError, match="both"):
        xwmt.WaterMassTransformations(grid, FULL_BUDGET, xbudget_dict=FULL_BUDGET)


def test_recipe_missing_raises(grid):
    with pytest.raises(TypeError, match="recipe"):
        xwmt.WaterMassTransformations(grid)


def test_deprecated_xbudget_dict_property_warns(grid):
    wmt = xwmt.WaterMassTransformations(grid, FULL_BUDGET)
    with pytest.warns(FutureWarning, match="xbudget_dict"):
        assert wmt.xbudget_dict is wmt.recipe


def test_deprecated_xbudget_dict_property_is_settable(grid):
    """It was a plain attribute before, so assignment must keep working."""
    wmt = xwmt.WaterMassTransformations(grid, FULL_BUDGET)
    with pytest.warns(FutureWarning, match="xbudget_dict"):
        wmt.xbudget_dict = {"mass": {}}
    assert wmt.recipe == {"mass": {}}


def test_internals_do_not_use_the_deprecated_property(grid):
    """Normal use must not trip xwmt's own deprecation warning.

    Internals read `self.recipe`; if one reverts to `self.xbudget_dict` the
    library would warn at users who did nothing wrong.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        wmt = xwmt.WaterMassTransformations(grid, FULL_BUDGET)
        wmt.lambdas()
        wmt.available_processes()
    assert not [w for w in rec if "xbudget_dict" in str(w.message)]


# -- budget metadata: absent key vs. present-but-None ------------------------


def test_budget_metadata_presence_vs_none():
    """The `_UNSET` sentinel must distinguish an absent key from a `None` value.

    This is the entire reason the helper exists. `lambda: None` and
    `thickness: None` are legitimate declarations that must resolve to `None`,
    while an *absent* key must fall through -- to `surface_lambda`, or to
    `h_name`'s default. Collapsing the two (e.g. a regression to `.get(key)`)
    would silently reject the idealized/surface budgets.
    """
    read = xwmt.wmt.WaterMassTransformations._budget_metadata
    UNSET = xwmt.wmt.WaterMassTransformations._UNSET
    LAMBDAS = ("lambda", "surface_lambda")

    # present-but-None resolves to None, not the sentinel
    assert read({"heat": {"lambda": None}}, "heat", LAMBDAS) is None
    assert read({"mass": {"thickness": None}}, "mass", ("thickness",)) is None
    # absent resolves to the sentinel
    assert read({"heat": {}}, "heat", LAMBDAS) is UNSET
    assert read({"mass": {}}, "mass", ("thickness",)) is UNSET
    # keys are tried in order: the first *present* one wins, even if it is None
    assert (
        read({"heat": {"lambda": None, "surface_lambda": "tos"}}, "heat", LAMBDAS)
        is None
    )
    # ...and it falls through when the first key is absent
    assert read({"heat": {"surface_lambda": "tos"}}, "heat", LAMBDAS) == "tos"


def test_tracer_with_explicit_null_lambda_is_accepted(grid):
    """`salt: {lambda: None}` (as the idealized/surface budgets declare) resolves."""
    wmt = xwmt.WaterMassTransformations(grid, FULL_BUDGET)
    assert wmt.tracer_dict["salt"] is None


def test_surface_lambda_used_when_lambda_absent(grid):
    budget = copy.deepcopy(FULL_BUDGET)
    del budget["heat"]["lambda"]
    budget["heat"]["surface_lambda"] = "temperature"
    wmt = xwmt.WaterMassTransformations(grid, budget)
    assert wmt.tracer_dict["heat"] == "temperature"


def test_tracer_missing_both_lambda_keys_raises(grid):
    budget = copy.deepcopy(FULL_BUDGET)
    del budget["heat"]["lambda"]
    with pytest.raises(ValueError, match="lambda"):
        xwmt.WaterMassTransformations(grid, budget)


def test_mass_thickness_sets_h_name(grid):
    """The mass budget's `thickness` is what WaterMass builds its metrics from."""
    wmt = xwmt.WaterMassTransformations(grid, FULL_BUDGET)
    assert wmt.h_name == "dz"


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


def _multitile_heat_grid(nz=6, ny=3, nx=4):
    """A 2-tile (LLC-like) grid carrying a heat tendency and `face_connections`,
    together with the matching pair of single-tile grids, for the issue-#59
    "broadcast across tiles" tests.

    The two tiles are glued along Y via `face_connections`, giving the grid a
    `face` dimension in addition to X/Y — exactly the topology (in miniature)
    that xwmt must reduce over when integrating on a multi-tile grid such as
    ECCOv4r4's 13-tile lat-lon-cap grid.
    """
    z_i = np.linspace(0.0, 1.0, nz + 1)
    z_l = 0.5 * (z_i[:-1] + z_i[1:])
    f, z, y, x = np.meshgrid(
        np.arange(2), z_l, np.arange(ny), np.arange(nx), indexing="ij"
    )
    # Temperature in [0, 1) with a distinct structure on each tile, so binning is
    # non-trivial and the two tiles populate different bins.
    temperature = (z + 0.13 * y + 0.07 * x + 0.29 * f) % 1.0
    # Extensive (layer-integrated) heat tendency, varying in space and by tile.
    dz = np.broadcast_to(np.diff(z_i)[None, :, None, None], temperature.shape)
    heat_tendency = (np.cos(2.0 * np.pi * z) + 0.5 * f + 0.1 * x) * dz

    def _ds(face=None):
        """Full 2-tile dataset (face=None) or the single tile `face`."""
        pick = (lambda a: a) if face is None else (lambda a: a[face])
        spatial = ("face", "z_l", "y", "x") if face is None else ("z_l", "y", "x")
        area = ("face", "y", "x") if face is None else ("y", "x")
        ds = xr.Dataset()
        coords = {
            "x": np.arange(nx, dtype=float),
            "y": np.arange(ny, dtype=float),
            "z_l": z_l,
            "z_i": z_i,
        }
        if face is None:
            coords["face"] = np.arange(2)
        ds = ds.assign_coords(coords)
        ds["temperature"] = (spatial, pick(temperature))
        ds["heat_tendency"] = (spatial, pick(heat_tendency))
        ds["dz"] = (spatial, pick(dz))
        ds["rA"] = (area, np.ones((2, ny, nx) if face is None else (ny, nx)))
        return ds

    coords = {
        "X": {"center": "x"},
        "Y": {"center": "y"},
        "Z": {"center": "z_l", "outer": "z_i"},
    }
    face_connections = {
        "face": {
            0: {"Y": (None, (1, "Y", False))},
            1: {"Y": ((0, "Y", False), None)},
        }
    }

    def _grid(ds, multitile):
        kw = {"face_connections": face_connections} if multitile else {}
        return xgcm.Grid(
            ds,
            coords=coords,
            metrics={("X", "Y"): ["rA"]},
            padding="fill",
            autoparse_metadata=False,
            **kw,
        )

    full = _grid(_ds(), multitile=True)
    tiles = [_grid(_ds(n), multitile=False) for n in range(2)]
    budget = {
        "mass": {"lambda": None, "thickness": "dz", "lhs": {}, "rhs": {}},
        "heat": {
            "lambda": "temperature",
            "lhs": {"tendency": "heat_tendency"},
            "rhs": {},
        },
        "salt": {"lambda": None, "lhs": {}, "rhs": {}},
    }
    bins = np.linspace(0.0, 1.0, 9)
    return full, tiles, budget, bins


def test_multitile_facedim_preserved():
    # issue #59: the face/tile dim (xgcm `_facedim`) must survive WaterMass's
    # deep-copy of the grid via `_rebuild_grid` and appear in the horizontal
    # reduction dims. Before the fix, `_rebuild_grid` dropped `face_connections`,
    # so `_facedim` was silently lost on the copy.
    full, _tiles, budget, _bins = _multitile_heat_grid()
    wmt = xwmt.WaterMassTransformations(full, budget, cp=1.0, rho_ref=1.0)
    assert wmt._facedim == "face"
    assert wmt._horizontal_dims == ["face", "x", "y"]


@pytest.mark.parametrize("method", ["xhistogram", "xgcm"])
def test_multitile_broadcast_equivalence(method):
    # issue #59: integrating transformations over a multi-tile grid must equal the
    # sum of the per-tile single-tile integrations, since binning `hlamdot` into
    # lambda space and summing over the horizontal are both linear in the tiles.
    # This pins that the `face` dimension is correctly included in the reduction.
    full, tiles, budget, bins = _multitile_heat_grid()

    def integrate(grid):
        wmt = xwmt.WaterMassTransformations(
            grid, budget, cp=1.0, rho_ref=1.0, method=method
        )
        return wmt.integrate_transformations("heat", bins=bins, sum_components=False)[
            "tendency"
        ]

    full_transformation = integrate(full)
    tiles_transformation = sum(integrate(g) for g in tiles)

    # The transformation is non-trivial (otherwise the equality would be vacuous).
    assert np.any(np.abs(np.nan_to_num(full_transformation.values)) > 0.0)
    assert np.allclose(
        np.nan_to_num(full_transformation.values),
        np.nan_to_num(tiles_transformation.values),
        rtol=1e-10,
        atol=1e-12,
    )
