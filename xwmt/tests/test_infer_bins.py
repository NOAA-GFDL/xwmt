"""
Tests for `WaterMass.infer_bins`, the fallback used when a caller does not pass
`bins=` explicitly.

Two things are pinned here. First, that the fallback works at all: it used to
hand `np.linspace` a pair of 0-d DataArrays, which numpy tried to wrap its 1-D
output back into, so *every* call without explicit bins raised `ValueError:
dimensions () must have the same length as the number of data dimensions`. The
rest of the suite missed it because every functional test passes `bins=`.

Second, that `percentiles` and `spacing` stay distinct. `percentiles` bounds the
range the bins span; `spacing` distributes the edges within it. Only the latter
makes the bins quantile-based, which is easy to misread from the name of the
former.

Runs on the tiny synthetic grids from `test_bugfixes`, so no download is needed.
"""

import warnings

import numpy as np
import pytest
import xarray as xr

import xwmt

from xwmt.tests.test_bugfixes import _heat_tendency_grid


@pytest.fixture
def wmt_and_lambda():
    grid, budget, _ = _heat_tendency_grid()
    wmt = xwmt.WaterMassTransformations(
        grid, budget, cp=1.0, rho_ref=1.0, method="xhistogram"
    )
    return wmt, grid._ds["temperature"]


@pytest.fixture
def skewed(wmt_and_lambda):
    """A lambda field whose values bunch up at one end.

    Uniform data is the one case where linear and quantile spacing agree, so it
    cannot distinguish them.
    """
    wmt, da = wmt_and_lambda
    return wmt, xr.DataArray(
        np.geomspace(1.0, 100.0, da.size).reshape(da.shape),
        dims=da.dims,
        coords=da.coords,
        name="skewed",
    )


# --- the fallback path works at all -------------------------------------------


def test_transformations_without_explicit_bins(wmt_and_lambda):
    wmt, _ = wmt_and_lambda
    out = wmt.integrate_transformations("heat")
    assert "tendency" in out
    assert np.isfinite(out["tendency"].values).any()


def test_infer_bins_returns_plain_float_edges(wmt_and_lambda):
    wmt, da = wmt_and_lambda
    bins = wmt.infer_bins(da, nbins=7)
    assert isinstance(bins, np.ndarray)
    assert not isinstance(bins, xr.DataArray)
    assert bins.dtype.kind == "f"
    assert bins.shape == (7,)


@pytest.mark.parametrize("spacing", ["linear", "quantiles"])
def test_edges_increase_and_span_the_data(wmt_and_lambda, spacing):
    wmt, da = wmt_and_lambda
    bins = wmt.infer_bins(da, nbins=9, spacing=spacing)
    assert np.all(np.diff(bins) > 0)
    assert bins[0] == pytest.approx(float(da.min()))
    assert bins[-1] == pytest.approx(float(da.max()))


@pytest.mark.parametrize("spacing", ["linear", "quantiles"])
def test_works_on_dask_backed_data(wmt_and_lambda, spacing):
    # Bin edges must be concrete numbers, so this is allowed to compute; it just
    # must not fail on a lazy input.
    wmt, da = wmt_and_lambda
    eager = wmt.infer_bins(da, nbins=6, spacing=spacing)
    lazy = wmt.infer_bins(da.chunk({"x": 1}), nbins=6, spacing=spacing)
    np.testing.assert_allclose(eager, lazy)


# --- `percentiles` bounds the range -------------------------------------------


def test_percentiles_trim_the_range(wmt_and_lambda):
    wmt, da = wmt_and_lambda
    full = wmt.infer_bins(da, nbins=5)
    trimmed = wmt.infer_bins(da, percentiles=(0.1, 0.9), nbins=5)
    assert trimmed[0] > full[0]
    assert trimmed[-1] < full[-1]


def test_percentiles_accepts_a_list_for_backwards_compatibility(wmt_and_lambda):
    wmt, da = wmt_and_lambda
    np.testing.assert_allclose(
        wmt.infer_bins(da, percentiles=[0.1, 0.9], nbins=5),
        wmt.infer_bins(da, percentiles=(0.1, 0.9), nbins=5),
    )


def test_percentiles_alone_do_not_make_spacing_quantile_based(skewed):
    # The heart of the naming confusion: trimming the range leaves the edges
    # inside it equally spaced.
    wmt, da = skewed
    bins = wmt.infer_bins(da, percentiles=(0.05, 0.95), nbins=8)
    widths = np.diff(bins)
    np.testing.assert_allclose(widths, widths[0])


# --- `spacing` distributes the edges ------------------------------------------


def test_linear_spacing_gives_equal_widths(skewed):
    wmt, da = skewed
    widths = np.diff(wmt.infer_bins(da, nbins=11))
    np.testing.assert_allclose(widths, widths[0])


def test_quantile_spacing_gives_equal_populations(skewed):
    wmt, da = skewed
    values = da.values.ravel()
    linear = np.histogram(values, bins=wmt.infer_bins(da, nbins=11))[0]
    quantile = np.histogram(
        values, bins=wmt.infer_bins(da, nbins=11, spacing="quantiles")
    )[0]
    # Equal population is the point; on a skewed field linear is far from it.
    assert quantile.max() - quantile.min() <= 1
    assert linear.max() - linear.min() > 1
    assert linear.sum() == quantile.sum() == values.size


def test_quantile_spacing_has_unequal_widths(skewed):
    wmt, da = skewed
    widths = np.diff(wmt.infer_bins(da, nbins=11, spacing="quantiles"))
    assert widths.max() > 2 * widths.min()


def test_spacings_agree_on_uniform_data(wmt_and_lambda):
    # `temperature` here is `np.linspace`, so the quantiles of a uniform grid of
    # probabilities are themselves uniformly spaced.
    wmt, da = wmt_and_lambda
    np.testing.assert_allclose(
        wmt.infer_bins(da, nbins=9),
        wmt.infer_bins(da, nbins=9, spacing="quantiles"),
        atol=1e-12,
    )


def test_percentiles_and_spacing_compose(skewed):
    """`spacing="quantiles"` places edges at quantiles *of the trimmed range*."""
    wmt, da = skewed
    lower, upper = 0.1, 0.9
    bins = wmt.infer_bins(da, percentiles=(lower, upper), nbins=6, spacing="quantiles")
    expected = np.asarray(da.quantile(np.linspace(lower, upper, 6), dim=da.dims))
    np.testing.assert_allclose(bins, expected)


def test_linear_is_the_default(skewed):
    wmt, da = skewed
    np.testing.assert_allclose(
        wmt.infer_bins(da, nbins=7), wmt.infer_bins(da, nbins=7, spacing="linear")
    )


# --- guard rails ---------------------------------------------------------------


def test_unknown_spacing_raises(wmt_and_lambda):
    wmt, da = wmt_and_lambda
    with pytest.raises(ValueError, match="spacing"):
        wmt.infer_bins(da, spacing="quantile")  # singular: a plausible typo


@pytest.mark.parametrize(
    "percentiles", [(0.5,), (0.1, 0.5, 0.9), (0.9, 0.1), (-0.1, 0.9), (0.1, 1.5)]
)
def test_invalid_percentiles_raise(wmt_and_lambda, percentiles):
    wmt, da = wmt_and_lambda
    with pytest.raises(ValueError, match="percentiles"):
        wmt.infer_bins(da, percentiles=percentiles)


def test_quantile_spacing_warns_when_bins_would_be_degenerate(wmt_and_lambda):
    """A field with a plateau cannot be split into equally populated bins.

    Repeated edges mean zero-width bins, which divide by zero downstream, so this
    warns rather than returning bins that quietly produce infinities.
    """
    wmt, da = wmt_and_lambda
    plateau = xr.zeros_like(da)
    plateau[..., -1] = 1.0
    with pytest.warns(UserWarning, match="repeated bin edges"):
        wmt.infer_bins(plateau, nbins=10, spacing="quantiles")


def test_linear_spacing_does_not_warn_on_a_plateau(wmt_and_lambda):
    # The same field is fine linearly: the edges depend only on the endpoints.
    wmt, da = wmt_and_lambda
    plateau = xr.zeros_like(da)
    plateau[..., -1] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bins = wmt.infer_bins(plateau, nbins=10)
    assert np.all(np.diff(bins) > 0)
