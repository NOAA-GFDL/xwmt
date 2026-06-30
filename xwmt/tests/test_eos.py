"""Tests for the generalized, `xeos`-backed equation-of-state handling.

These cover the `xwmt.eos` helpers directly and their integration through
`WaterMass.get_density`, including:

- the EOS spec resolver (`resolve_eos`),
- temperature/salinity kind conversion (`convert_ts`),
- that the default "teos10" EOS reproduces `gsw` bit-for-bit (so the refactor is
  numerically inert for the historical default),
- that every `xeos` EOS can be selected and yields physically plausible output,
- selecting an EOS by `xeos.EquationOfState` instance (e.g. `xeos.from_model`),
- the deprecated `teos10` keyword, and
- the `eos=None` "bring your own alpha/beta/density" contract.
"""

import warnings

import numpy as np
import pytest
import xarray as xr
import xgcm
import gsw
import xeos

import xwmt
from xwmt.eos import list_eos, resolve_eos, convert_ts


def _grid(t=(18.0, 12.0, 5.0), s=(35.0, 35.2, 34.9), lat=30.0, lon=0.0):
    """A tiny single-column 3-layer grid with temperature, salinity, thickness."""
    zi = np.array([0.0, 10.0, 30.0, 60.0])
    zl = 0.5 * (zi[:-1] + zi[1:])
    ds = xr.Dataset(
        {
            "thetao": ("z_l", np.array(t)),
            "so": ("z_l", np.array(s)),
            "thkcello": ("z_l", np.diff(zi)),
        },
        coords={
            "z_l": ("z_l", zl),
            "z_i": ("z_i", zi),
            "lon": lon,
            "lat": lat,
        },
    )
    ds = ds.expand_dims(("x", "y")).assign_coords(x=[1.0], y=[1.0])
    ds["rA"] = (("x", "y"), [[1.0]])
    coords = {
        "X": {"center": "x"},
        "Y": {"center": "y"},
        "Z": {"center": "z_l", "outer": "z_i"},
    }
    return xgcm.Grid(
        ds,
        coords=coords,
        metrics={("X", "Y"): ["rA"]},
        periodic=False,
        autoparse_metadata=False,
    )


def test_list_eos_matches_xeos():
    assert list_eos() == xeos.list_eos()
    assert "teos10" in list_eos()
    assert "wright97-full" in list_eos()


def test_resolve_eos_variants():
    # string id -> EquationOfState
    eos = resolve_eos("teos10")
    assert isinstance(eos, xeos.EquationOfState)
    assert eos.id == "teos10"
    # instance passes through unchanged
    assert resolve_eos(eos) is eos
    # None passes through
    assert resolve_eos(None) is None
    # unknown id raises (from xeos)
    with pytest.raises(Exception):
        resolve_eos("not-a-real-eos")
    # wrong type raises TypeError
    with pytest.raises(TypeError):
        resolve_eos(42)


def test_teos10_matches_gsw_bit_for_bit():
    """The default EOS must reproduce the historical gsw results exactly."""
    grid = _grid()
    SA = grid._ds.so.squeeze().values
    CT = grid._ds.thetao.squeeze().values

    wm = xwmt.WaterMass(grid, eos="teos10")
    rho = wm.get_density("rho").squeeze().values
    p = wm.grid._ds.p.squeeze().values
    np.testing.assert_array_equal(rho, gsw.rho(SA, CT, p))

    wm2 = xwmt.WaterMass(grid, eos="teos10")
    s2 = wm2.get_density("sigma2").squeeze().values
    np.testing.assert_array_equal(s2, gsw.sigma2(SA, CT))


@pytest.mark.parametrize("eos", sorted(list_eos()))
def test_every_eos_gives_plausible_density(eos):
    """Every registered EOS is selectable and returns sensible seawater values."""
    grid = _grid()
    wm = xwmt.WaterMass(grid, eos=eos)
    rho = wm.get_density("rho").squeeze().values
    s2 = wm.get_density("sigma2").squeeze().values
    alpha = wm.grid._ds.alpha.squeeze().values
    beta = wm.grid._ds.beta.squeeze().values
    # Idealized/linear EOS (e.g. the Roquet idealized forms, mpas-linear) are tuned
    # to different reference states, so only assert broad physical sanity here; the
    # tight realistic range is checked in test_teos10_matches_gsw_bit_for_bit.
    assert np.all(np.isfinite(rho)) and np.all((900.0 < rho) & (rho < 1100.0))
    assert np.all(np.isfinite(s2))
    assert np.all(np.isfinite(alpha)) and np.all(np.isfinite(beta))
    # Denser-with-saltier holds for every physical seawater EOS here.
    assert np.all(beta > 0.0)


def test_eos_by_instance_from_model():
    """An EOS can be supplied as a prebuilt xeos.EquationOfState (e.g. from_model)."""
    grid = _grid()
    eos = xeos.from_model("MOM6", "WRIGHT_FULL")
    wm = xwmt.WaterMass(grid, eos=eos)
    rho = wm.get_density("rho").squeeze().values
    # Identical to selecting the resolved canonical id directly.
    wm_id = xwmt.WaterMass(_grid(), eos="wright97-full")
    np.testing.assert_array_equal(rho, wm_id.get_density("rho").squeeze().values)


def test_convert_ts_identity_for_matching_kinds():
    """Conservative/absolute input to a TEOS-10 EOS needs no conversion."""
    grid = _grid()
    t, s = grid._ds.thetao, grid._ds.so
    p = xr.zeros_like(t)
    eos = resolve_eos("teos10")
    temp, salt = convert_ts(t, s, eos, "conservative", "absolute", p)
    assert temp is t
    assert salt is s


def test_convert_ts_converts_for_potential_practical_eos():
    """A potential/practical EOS triggers real conversions that change the values."""
    grid = _grid()
    t, s = grid._ds.thetao, grid._ds.so
    p = xr.full_like(t, 100.0)
    lon = xr.full_like(t, 0.0)
    lat = xr.full_like(t, 30.0)
    eos = resolve_eos("jmd95")  # wants potential temperature + practical salinity
    temp, salt = convert_ts(t, s, eos, "conservative", "absolute", p, lon=lon, lat=lat)
    # Conversions actually ran (values differ from the absolute/conservative input).
    assert not np.allclose(temp.values, t.values)
    assert not np.allclose(salt.values, s.values)
    # And match gsw's own conversions.
    np.testing.assert_allclose(
        salt.values, gsw.SP_from_SA(s.values, p.values, lon.values, lat.values)
    )


def test_convert_ts_bad_kinds_raise():
    eos = resolve_eos("teos10")
    p = xr.DataArray(np.zeros(3))
    t = xr.DataArray(np.zeros(3))
    with pytest.raises(ValueError):
        convert_ts(t, t, eos, "bogus", "absolute", p)
    with pytest.raises(ValueError):
        convert_ts(t, t, eos, "conservative", "bogus", p)


def test_deprecated_teos10_true_maps_to_teos10():
    grid = _grid()
    with pytest.warns(DeprecationWarning):
        wm = xwmt.WaterMass(grid, teos10=True)
    assert wm.eos is not None and wm.eos.id == "teos10"


def test_deprecated_teos10_false_maps_to_none():
    grid = _grid()
    with pytest.warns(DeprecationWarning):
        wm = xwmt.WaterMass(grid, teos10=False)
    assert wm.eos is None


def test_explicit_eos_overrides_deprecated_teos10():
    grid = _grid()
    with pytest.warns(DeprecationWarning):
        wm = xwmt.WaterMass(grid, eos="wright97-full", teos10=True)
    assert wm.eos.id == "wright97-full"


def test_eos_none_requires_user_alpha_beta_density():
    grid = _grid()
    wm = xwmt.WaterMass(grid, eos=None)
    with pytest.raises(ValueError, match="alpha"):
        wm.get_density("sigma0")


def test_eos_none_with_supplied_fields_returns_them():
    grid = _grid()
    # Pre-supply alpha, beta and a density field.
    grid._ds["alpha"] = xr.full_like(grid._ds.thetao, 2.0e-4)
    grid._ds["beta"] = xr.full_like(grid._ds.thetao, 7.5e-4)
    grid._ds["sigma0"] = xr.full_like(grid._ds.thetao, 25.0)
    wm = xwmt.WaterMass(grid, eos=None)
    out = wm.get_density("sigma0")
    np.testing.assert_array_equal(out.values, grid._ds.sigma0.values)


def test_user_supplied_alpha_beta_are_not_overwritten():
    grid = _grid()
    grid._ds["alpha"] = xr.full_like(grid._ds.thetao, 1.234e-4)
    grid._ds["beta"] = xr.full_like(grid._ds.thetao, 5.678e-4)
    wm = xwmt.WaterMass(grid, eos="teos10")
    wm.get_density("rho")
    assert np.all(wm.grid._ds.alpha.values == 1.234e-4)
    assert np.all(wm.grid._ds.beta.values == 5.678e-4)
