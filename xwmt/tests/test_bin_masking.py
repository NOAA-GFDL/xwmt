"""
Tests for `N_min` masking of under-resolved lambda bins.

Transformation rates binned into the tails of the water mass distribution are
often built from a handful of grid cells, and plot as noise that looks like
signal. `N_min` masks bins sampled by fewer than that many grid cells; these
tests pin the count itself (`count_cells_per_bin`) and the masking it drives.

The grids here are tiny and synthetic, so the expected counts are known exactly
and no test data needs downloading.
"""

import numpy as np
import pytest
import xarray as xr
import xgcm

import xwmt

# Cell temperatures for the 3x1 column x 4 level test grid, chosen so the counts
# per unit-width bin are known exactly: 1 cell in [0, 1), 3 in [1, 2), 8 in [2, 3).
TEMPERATURES = np.array(
    [
        [0.5, 1.5, 1.5],
        [1.5, 2.5, 2.5],
        [2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5],
    ]
)
BINS = np.array([0.0, 1.0, 2.0, 3.0])
EXPECTED_COUNTS = np.array([1, 3, 8])


def _tendency_grid(zname="z", temperatures=TEMPERATURES):
    """
    A tiny (nz, ny=1, nx) grid carrying a heat tendency and a controlled temperature
    distribution.

    With ``zname="temperature"`` the vertical coordinate *is* the lambda variable
    (``temperature_l``/``temperature_i``), exercising the "prebinned" code path.
    """
    nz, nx = temperatures.shape
    edges = np.linspace(0.0, 1.0, nz + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ci, cl = f"{zname}_i", f"{zname}_l"

    ds = xr.Dataset()
    ds = ds.assign_coords(
        {
            ci: xr.DataArray(edges, dims=(ci,)),
            cl: xr.DataArray(centers, dims=(cl,)),
            "x": xr.DataArray(np.arange(nx, dtype=float), dims=("x",)),
            "y": xr.DataArray(np.array([0.0]), dims=("y",)),
        }
    )
    ds["dz"] = xr.DataArray(
        np.tile(np.diff(edges)[:, None, None], (1, 1, nx)), dims=(cl, "y", "x")
    )
    ds["temperature"] = xr.DataArray(temperatures[:, None, :], dims=(cl, "y", "x"))
    # A uniform, strictly positive tendency, so every populated bin has a nonzero
    # transformation rate before masking.
    ds["heat_tendency"] = (
        xr.DataArray(np.ones((nz, 1, nx)), dims=(cl, "y", "x")) * ds.dz
    )
    ds = ds.assign_coords({"rA": xr.DataArray(np.ones((1, nx)), dims=("y", "x"))})

    grid = xgcm.Grid(
        ds,
        coords={
            "X": {"center": "x"},
            "Y": {"center": "y"},
            "Z": {"center": cl, "outer": ci},
        },
        metrics={("X", "Y"): ["rA"]},
        padding="fill",
        autoparse_metadata=False,
    )
    recipe = {
        "mass": {"lambda": None, "thickness": "dz", "lhs": {}, "rhs": {}},
        "heat": {
            "lambda": "temperature",
            "lhs": {"tendency": "heat_tendency"},
            "rhs": {},
        },
        "salt": {"lambda": None, "lhs": {}, "rhs": {}},
    }
    return grid, recipe


def _wmt(**kwargs):
    grid, recipe = _tendency_grid()
    return xwmt.WaterMassTransformations(grid, recipe, cp=1.0, rho_ref=1.0, **kwargs)


def test_count_cells_per_bin_integrated():
    # The domain-wide census is just the histogram of lambda over all grid cells.
    wmt = _wmt()
    counts = wmt.count_cells_per_bin("heat", bins=BINS)
    assert counts.dims == ("temperature_l_target",)
    assert np.array_equal(counts.values, EXPECTED_COUNTS)


def test_count_cells_per_bin_column_wise():
    # Column-wise, each of the 3 columns is counted separately (4 cells per column).
    wmt = _wmt()
    counts = wmt.count_cells_per_bin("heat", bins=BINS, integrate=False)
    assert set(counts.dims) == {"x", "y", "temperature_l_target"}
    assert np.array_equal(
        counts.transpose("temperature_l_target", "y", "x").values.squeeze(),
        np.array([[1, 0, 0], [1, 1, 1], [2, 3, 3]]),
    )
    # Every cell is counted exactly once.
    assert counts.sum().item() == TEMPERATURES.size


def test_count_cells_per_bin_respects_mask():
    # A region mask excludes those columns from the census entirely.
    grid, recipe = _tendency_grid()
    keep_first_column = xr.DataArray([[True, False, False]], dims=("y", "x"))
    wmt = xwmt.WaterMassTransformations(grid, recipe, cp=1.0, rho_ref=1.0)
    counts = wmt.count_cells_per_bin("heat", bins=BINS, mask=keep_first_column)
    assert np.array_equal(counts.values, np.array([1, 1, 2]))


def test_count_cells_per_bin_ignores_vanished_layers():
    # Layers of zero thickness are not real grid cells and must not be counted.
    grid, recipe = _tendency_grid()
    grid._ds["dz"] = xr.where(
        grid._ds.z_l == grid._ds.z_l[0], 0.0, grid._ds.dz
    ).transpose(*grid._ds.dz.dims)
    wmt = xwmt.WaterMassTransformations(grid, recipe, cp=1.0, rho_ref=1.0)
    counts = wmt.count_cells_per_bin("heat", bins=BINS)
    # The vanished top layer held 1 cell in bin 0 and 2 cells in bin 1.
    assert np.array_equal(counts.values, np.array([0, 1, 8]))


@pytest.mark.parametrize("method", ["xhistogram", "xgcm"])
def test_N_min_masks_undersampled_bins(method):
    # The bin sampled by a single cell is masked at N_min=2; better-sampled bins
    # keep exactly the values they had without masking. This must hold for both
    # transformation backends, whose target coordinates have to line up.
    wmt = _wmt(method=method)
    full = wmt.integrate_transformations("heat", bins=BINS, sum_components=False)
    masked = wmt.integrate_transformations(
        "heat", bins=BINS, sum_components=False, N_min=2
    )

    assert not np.any(np.isnan(full["tendency"].values))
    assert np.array_equal(np.isnan(masked["tendency"].values), EXPECTED_COUNTS < 2)
    keep = ~np.isnan(masked["tendency"].values)
    assert np.allclose(masked["tendency"].values[keep], full["tendency"].values[keep])
    # Masking is only worth having if it removed a nonzero rate.
    assert np.any(np.abs(full["tendency"].values[~keep]) > 0.0)


def test_N_min_none_is_a_noop():
    # The default must leave results bit-for-bit unchanged.
    wmt = _wmt()
    default = wmt.integrate_transformations("heat", bins=BINS, sum_components=False)
    explicit = wmt.integrate_transformations(
        "heat", bins=BINS, sum_components=False, N_min=None
    )
    assert np.array_equal(default["tendency"].values, explicit["tendency"].values)
    assert not np.any(np.isnan(default["tendency"].values))


def test_N_min_zero_masks_nothing():
    wmt = _wmt()
    masked = wmt.integrate_transformations(
        "heat", bins=BINS, sum_components=False, N_min=0
    )
    assert not np.any(np.isnan(masked["tendency"].values))


def test_N_min_from_constructor_and_per_call_override():
    # The constructor value is the default; a per-call N_min wins over it.
    wmt = _wmt(N_min=2)
    assert wmt.N_min == 2
    from_constructor = wmt.integrate_transformations(
        "heat", bins=BINS, sum_components=False
    )
    assert np.array_equal(
        np.isnan(from_constructor["tendency"].values), EXPECTED_COUNTS < 2
    )

    override = wmt.integrate_transformations(
        "heat", bins=BINS, sum_components=False, N_min=4
    )
    assert np.array_equal(np.isnan(override["tendency"].values), EXPECTED_COUNTS < 4)


def test_N_min_masks_column_wise_transformations():
    # In `map_transformations` the count is per column, so N_min is bounded by the
    # number of vertical levels rather than by the size of the domain.
    wmt = _wmt()
    masked = wmt.map_transformations(
        "heat", bins=BINS, sum_components=False, N_min=2
    ).transpose("temperature_l_target", "y", "x")
    expected_counts = np.array([[1, 0, 0], [1, 1, 1], [2, 3, 3]])
    assert np.array_equal(
        np.isnan(masked["tendency"].values.squeeze()), expected_counts < 2
    )


def test_N_min_propagates_into_summed_and_grouped_terms():
    # Masked bins must stay masked once terms are summed into groups, rather than
    # quietly reappearing as a partial sum.
    wmt = _wmt()
    masked = wmt.integrate_transformations(
        "heat", bins=BINS, sum_components=True, group_processes=True, N_min=2
    )
    assert np.array_equal(
        np.isnan(masked["kinematic_transformation"].values), EXPECTED_COUNTS < 2
    )


def test_N_min_prebinned_lambda():
    # When lambda is already the vertical coordinate, the census still has to
    # broadcast the 1D coordinate across the horizontal and land on the same
    # target coordinate the transformation uses.
    grid, recipe = _tendency_grid(zname="temperature")
    wmt = xwmt.WaterMassTransformations(grid, recipe, cp=1.0, rho_ref=1.0)
    bins = np.linspace(0.0, 1.0, 5)  # the native layer interfaces

    counts = wmt.count_cells_per_bin("heat", bins=bins)
    assert counts.dims == ("temperature_l_target",)
    # One layer per bin, times 3 columns.
    assert np.array_equal(counts.values, np.array([3, 3, 3, 3]))

    masked = wmt.integrate_transformations(
        "heat", bins=bins, sum_components=False, N_min=4
    )
    assert np.all(np.isnan(masked["tendency"].values))
    kept = wmt.integrate_transformations(
        "heat", bins=bins, sum_components=False, N_min=3
    )
    assert not np.any(np.isnan(kept["tendency"].values))


def _surface_grid(ny=4, nx=5):
    """A tiny 2D surface grid (no Z axis), as used for surface WMT."""
    ds = xr.Dataset()
    ds = ds.assign_coords(
        {"x": np.arange(nx, dtype=float), "y": np.arange(ny, dtype=float)}
    )
    # One cell per bin edge step: `ny * nx` cells spread evenly over 10 unit bins.
    ds["tos"] = xr.DataArray(
        np.arange(ny * nx, dtype=float).reshape(ny, nx) / (ny * nx) * 10.0,
        dims=("y", "x"),
    )
    ds["sos"] = xr.DataArray(35.0 * np.ones((ny, nx)), dims=("y", "x"))
    ds["hfds"] = xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))
    ds = ds.assign_coords(
        {"areacello": xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))}
    )
    grid = xgcm.Grid(
        ds,
        coords={"X": {"center": "x"}, "Y": {"center": "y"}},
        metrics={("X", "Y"): "areacello"},
        padding="fill",
        autoparse_metadata=False,
    )
    recipe = {
        "mass": {},
        "heat": {"surface_lambda": "tos", "rhs": {"surface_flux": "hfds"}},
        "salt": {"surface_lambda": "sos"},
    }
    return grid, recipe


def test_N_min_surface_only_lambda():
    # A surface (2D) lambda has no vertical dimension: the census is over surface
    # cells only. It must *not* count the zero-filled layers that
    # `expand_surface_array_vertically` synthesizes for the transformation itself.
    grid, recipe = _surface_grid()
    wmt = xwmt.WaterMassTransformations(grid, recipe, cp=1.0, rho_ref=1.0)
    bins = np.arange(0.0, 11.0, 1.0)

    counts = wmt.count_cells_per_bin("heat", bins=bins)
    assert np.array_equal(counts.values, np.full(10, 2))

    full = wmt.integrate_transformations("heat", bins=bins, sum_components=False)
    assert not np.any(np.isnan(full["surface_flux"].values))
    masked = wmt.integrate_transformations(
        "heat", bins=bins, sum_components=False, N_min=3
    )
    assert np.all(np.isnan(masked["surface_flux"].values))
    kept = wmt.integrate_transformations(
        "heat", bins=bins, sum_components=False, N_min=2
    )
    assert np.allclose(kept["surface_flux"].values, full["surface_flux"].values)


def test_N_min_counts_across_tiles_of_a_multitile_grid():
    # On a multi-tile (LLC/ECCO-like) grid the census must reduce over the face
    # dimension too, so that the whole domain -- not one tile -- decides whether a
    # bin is under-resolved (cf. the issue-#59 broadcast tests).
    from xwmt.tests.test_bugfixes import _multitile_heat_grid

    full, tiles, recipe, bins = _multitile_heat_grid()
    counts_full = xwmt.WaterMassTransformations(
        full, recipe, cp=1.0, rho_ref=1.0
    ).count_cells_per_bin("heat", bins=bins)
    counts_tiles = sum(
        xwmt.WaterMassTransformations(g, recipe, cp=1.0, rho_ref=1.0)
        .count_cells_per_bin("heat", bins=bins)
        .values
        for g in tiles
    )
    assert counts_full.dims == ("temperature_l_target",)
    assert np.array_equal(counts_full.values, counts_tiles)


def test_N_min_validation():
    grid, recipe = _tendency_grid()
    with pytest.raises(ValueError, match="non-negative"):
        xwmt.WaterMassTransformations(grid, recipe, N_min=-1)
    with pytest.raises(TypeError, match="integer"):
        xwmt.WaterMassTransformations(grid, recipe, N_min=2.5)
    with pytest.raises(TypeError, match="integer"):
        xwmt.WaterMassTransformations(grid, recipe, N_min="10")

    wmt = xwmt.WaterMassTransformations(grid, recipe, cp=1.0, rho_ref=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        wmt.integrate_transformations("heat", bins=BINS, N_min=-3)


def test_N_min_stays_lazy():
    # Masking must not trigger computation of a dask-backed dataset.
    grid, recipe = _tendency_grid()
    grid._ds = grid._ds.chunk({"x": 1})
    wmt = xwmt.WaterMassTransformations(
        xgcm.Grid(
            grid._ds,
            coords={
                "X": {"center": "x"},
                "Y": {"center": "y"},
                "Z": {"center": "z_l", "outer": "z_i"},
            },
            metrics={("X", "Y"): ["rA"]},
            padding="fill",
            autoparse_metadata=False,
        ),
        recipe,
        cp=1.0,
        rho_ref=1.0,
    )
    masked = wmt.integrate_transformations(
        "heat", bins=BINS, sum_components=False, N_min=2
    )
    assert masked["tendency"].chunks is not None
    assert np.array_equal(
        np.isnan(masked["tendency"].compute().values), EXPECTED_COUNTS < 2
    )
