"""
Tests for the CF-style metadata xwmt attaches to its output.

Two invariants, and both matter: the correct attributes are present, and none of
the input diagnostics' attributes survive onto anything derived from them. The
second is the easier one to lose silently -- xarray propagates attributes through
arithmetic, so a rate in kg s-1 will happily describe itself as the "W m-2" of
the tendency it came from unless something clears it.

Everything here runs on the tiny synthetic grids from `test_bugfixes`, so no
dataset download is needed.
"""

import numpy as np
import pytest
import xarray as xr
import xgcm

import xwmt
from xwmt import attrs as xattrs

from xwmt.tests.test_bugfixes import (
    _heat_tendency_grid,
    _multitile_heat_grid,
    _surface_grid,
    minimal_grid,
)

#: Units of the input tendencies in the MOM6/xbudget conventions. None of these
#: may appear on any output variable.
INPUT_UNITS = {"W m-2", "kg m-2 s-1", "W", "degC", "psu", "m"}


def _wmt_and_bins(**kwargs):
    grid, budget, edges = _heat_tendency_grid()
    # Label the lambda field the way a model would, so we can check that the
    # target bin coordinate inherits its metadata.
    grid._ds["temperature"].attrs.update(
        {"units": "degC", "long_name": "potential temperature"}
    )
    # Extensive, per grid cell: the xbudget conventions multiply cell area into
    # every tendency before xwmt sees it, so this is W and not W m-2. (The
    # fixture builds `ones * dz` on a grid with `rA = 1.0`.)
    grid._ds["heat_tendency"].attrs.update({"units": "W"})
    wmt = xwmt.WaterMassTransformations(
        grid, budget, cp=1.0, rho_ref=1.0, method="xhistogram", **kwargs
    )
    return wmt, edges


@pytest.fixture
def heat_transformations():
    wmt, edges = _wmt_and_bins()
    return wmt, wmt.integrate_transformations("heat", bins=edges, sum_components=False)


def test_transformation_rates_are_in_kg_per_second(heat_transformations):
    _, transformations = heat_transformations
    rates = [v for v in transformations.data_vars if not v.endswith("_bnds")]
    assert rates, "expected at least one transformation rate variable"
    for name in rates:
        assert transformations[name].attrs["units"] == "kg s-1"


def test_no_input_attributes_leak_into_output(heat_transformations):
    """No output variable may describe itself with an input tendency's units."""
    _, transformations = heat_transformations
    for name, da in transformations.data_vars.items():
        if name.endswith("_bnds"):
            continue
        assert da.attrs.get("units") not in INPUT_UNITS - {"kg s-1"}
        # Sampling metadata describes the input diagnostic, not the rate.
        assert "cell_measures" not in da.attrs
        assert "time_avg_info" not in da.attrs


def test_transformation_rates_are_described(heat_transformations):
    _, transformations = heat_transformations
    da = transformations["tendency"]
    assert da.attrs["long_name"].startswith("Water mass transformation rate")
    assert "heat" in da.attrs["long_name"]
    assert da.attrs["xwmt_term"] == "tendency"
    assert da.attrs["xwmt_lambda"] == "heat"
    assert da.attrs["xwmt_lambda_variable"] == "temperature"
    assert da.attrs["xwmt_method"] == "xhistogram"
    assert da.attrs["xwmt_version"] == xwmt.__version__
    assert da.attrs["xwmt_source_variables"] == "heat_tendency"
    # `tendency` is declared on the left-hand side of the synthetic heat budget.
    assert da.attrs["xwmt_budget_side"] == "lhs"


def test_sign_convention_is_documented(heat_transformations):
    """Transformation is convergence into a layer, i.e. toward lower lambda."""
    _, transformations = heat_transformations
    comment = transformations["tendency"].attrs["comment"]
    assert "Positive values denote net transformation toward lower heat" in comment


def test_cell_methods_distinguish_integrated_from_mapped():
    wmt, edges = _wmt_and_bins()
    integrated = wmt.integrate_transformations("heat", bins=edges, sum_components=False)
    mapped = wmt.map_transformations("heat", bins=edges, sum_components=False)

    # Integrated sums over the horizontal as well as the vertical...
    assert integrated["tendency"].attrs["cell_methods"] == "x: sum y: sum z_l: sum"
    # ...mapped only over the vertical.
    assert mapped["tendency"].attrs["cell_methods"] == "z_l: sum"

    # Both are totals in kg s-1; the mapped one says explicitly that its values
    # are per grid cell rather than per unit area, since the horizontal area
    # factor is carried by the input budget terms.
    assert mapped["tendency"].attrs["units"] == "kg s-1"
    assert "per grid cell" in mapped["tendency"].attrs["comment"]
    assert "per grid cell" not in integrated["tendency"].attrs["comment"]


def test_cell_methods_include_face_dimension_on_multitile_grid():
    """A multi-tile grid also reduces over `face`, which cell_methods must record."""
    multi, _tiles, budget, bins = _multitile_heat_grid()
    wmt = xwmt.WaterMassTransformations(
        multi, budget, cp=1.0, rho_ref=1.0, method="xhistogram"
    )
    transformations = wmt.integrate_transformations(
        "heat", bins=bins, sum_components=False
    )
    assert "face: sum" in transformations["tendency"].attrs["cell_methods"]


def test_lambda_coordinate_is_described_with_cf_bounds(heat_transformations):
    _, transformations = heat_transformations
    coord = transformations["temperature_l_target"]
    # Units and name are inherited from the lambda field itself, so a model's own
    # metadata carries through rather than being guessed at.
    assert coord.attrs["units"] == "degC"
    assert "potential temperature" in coord.attrs["long_name"]
    assert "bin centers" in coord.attrs["long_name"]

    bounds_name = coord.attrs["bounds"]
    assert bounds_name == "temperature_l_target_bnds"
    bounds = transformations[bounds_name]
    assert bounds.dims == ("temperature_l_target", "bnds")

    # The bounds must reproduce exactly the bin edges that were requested, which
    # is what lets a reader recover the bin widths the rates were divided by.
    _, edges = _wmt_and_bins()
    np.testing.assert_allclose(bounds.values[:, 0], edges[:-1])
    np.testing.assert_allclose(bounds.values[:, 1], edges[1:])
    # The private carrier attribute must not survive into the output.
    assert "xwmt_bin_edges" not in coord.attrs


def test_global_attributes_describe_the_calculation():
    wmt, edges = _wmt_and_bins()
    integrated = wmt.integrate_transformations("heat", bins=edges, sum_components=False)
    mapped = wmt.map_transformations("heat", bins=edges, sum_components=False)

    assert integrated.attrs["xwmt_version"] == xwmt.__version__
    assert integrated.attrs["xwmt_lambda"] == "heat"
    assert integrated.attrs["xwmt_integrate"] == "true"
    assert mapped.attrs["xwmt_integrate"] == "false"
    # The conservation disclaimer travels with the data, not just the docs.
    assert "conserved" in integrated.attrs["comment"]
    assert "10.1029/2024MS004383" in integrated.attrs["references"]


def test_summed_terms_are_described():
    """`sum(das)` drops attributes, so the grouped variables need their own."""
    wmt, edges = _wmt_and_bins()
    grouped = wmt.integrate_transformations(
        "heat", bins=edges, sum_components=False, group_processes=True
    )
    kinematic = grouped["kinematic_transformation"]
    assert kinematic.attrs["units"] == "kg s-1"
    assert "left-hand-side" in kinematic.attrs["long_name"]
    assert kinematic.attrs["xwmt_budget_side"] == "lhs"
    assert kinematic.attrs["xwmt_summed_terms"] == "tendency"
    # A sum is not attributable to one source tendency; its provenance is the
    # list of terms that went into it.
    assert "xwmt_source_variables" not in kinematic.attrs


def test_output_survives_a_netcdf_round_trip(tmp_path):
    """Every attribute must be netCDF-serializable, and the CF bounds must decode."""
    wmt, edges = _wmt_and_bins()
    transformations = wmt.integrate_transformations(
        "heat", bins=edges, sum_components=False, group_processes=True
    )
    path = tmp_path / "wmt.nc"
    transformations.to_netcdf(path)
    with xr.open_dataset(path) as roundtripped:
        assert roundtripped["tendency"].attrs["units"] == "kg s-1"
        assert roundtripped.attrs["xwmt_lambda"] == "heat"
        coord = roundtripped["temperature_l_target"]
        assert coord.attrs["bounds"] == "temperature_l_target_bnds"
        assert "temperature_l_target_bnds" in roundtripped.variables


def test_surface_only_output_is_described():
    """The 2D/surface path synthesizes a Z axis; its output still needs metadata."""
    budget = {
        "mass": {},
        "heat": {"surface_lambda": "tos", "lhs": {}, "rhs": {}},
        "salt": {"surface_lambda": "sos", "lhs": {}, "rhs": {}},
    }
    wmt = xwmt.WaterMassTransformations(_surface_grid(), budget)
    h_i = wmt.grid._ds[f"{wmt.h_name}_i"]
    assert "placeholder" in h_i.attrs["long_name"]


def _generic_tracer_grid(tracer_units="mol s-1", lambda_units="mol m-3"):
    """A budget that is neither heat, salt, nor mass.

    Exercises the generic-tracer branch of `calc_hlamdot_and_lambda`, which
    applies no conversion at all -- no `cp`, no salt-to-grams rescaling, no
    reference density. Whatever units come out are the input tendency's divided
    by the lambda's.
    """
    grid, _, edges = _heat_tendency_grid()
    ds = grid._ds
    ds["dissic"] = ds["temperature"] * 0.0 + np.linspace(
        0.0, 1.0, ds.sizes["z_l"]
    ).reshape(ds["temperature"].shape)
    ds["dissic"].attrs.update(
        {"units": lambda_units, "long_name": "dissolved inorganic carbon"}
    )
    ds["dic_tendency"] = ds["heat_tendency"].copy()
    ds["dic_tendency"].attrs.update({"units": tracer_units})
    budget = {
        "mass": {"lambda": None, "thickness": "dz", "lhs": {}, "rhs": {}},
        "dic": {"lambda": "dissic", "lhs": {"tendency": "dic_tendency"}, "rhs": {}},
    }
    return grid, budget, edges


def test_generic_tracer_rate_is_a_volume_not_a_mass_rate():
    """A tracer budget with no density factor anywhere is a volume tendency.

    The generic-tracer path multiplies by nothing, so a `mol s-1` tendency
    binned in a `mol m-3` concentration is `m3 s-1`. Asserting `kg s-1` here --
    as a hardcoded constant did -- is simply false.
    """
    grid, budget, edges = _generic_tracer_grid()
    wmt = xwmt.WaterMassTransformations(grid, budget, method="xhistogram")
    transformations = wmt.integrate_transformations("dic", bins=edges)
    assert transformations["tendency"].attrs["units"] == "m3 s-1"


def test_area_intensive_tendency_is_reported_as_area_intensive():
    """xwmt never multiplies by cell area, so a per-m2 input stays per-m2.

    The xbudget conventions deliver area-extensive tendencies, but nothing
    enforces that for a hand-built recipe. Asserting `kg s-1` regardless -- as a
    hardcoded constant did -- mislabels this case exactly as badly as it
    mislabelled the generic tracer.
    """
    grid, budget, edges = _heat_tendency_grid()
    grid._ds["temperature"].attrs.update({"units": "degC"})
    grid._ds["heat_tendency"].attrs.update({"units": "W m-2"})
    wmt = xwmt.WaterMassTransformations(
        grid, budget, cp=1.0, rho_ref=1.0, method="xhistogram"
    )
    transformations = wmt.integrate_transformations("heat", bins=edges)
    assert transformations["tendency"].attrs["units"] == "m-2 kg s-1"


def _density_grid():
    """A minimal grid with both heat and salt tendencies, for a density lambda."""
    grid, budget, edges = _heat_tendency_grid()
    ds = grid._ds
    ds["temperature"].attrs.update({"units": "degC"})
    ds["heat_tendency"].attrs.update({"units": "W"})
    ds["so"] = ds["temperature"] * 0.0 + 35.0
    ds["so"].attrs.update({"units": "psu"})
    ds["salt_tendency"] = ds["heat_tendency"].copy()
    ds["salt_tendency"].attrs.update({"units": "kg s-1"})
    # `get_density` needs a position to convert between practical and absolute
    # salinity (`lon` as well as `lat`, for the practical-salinity backends).
    grid._ds = ds.assign_coords(
        {
            "lat": xr.DataArray([[45.0]], dims=("x", "y")),
            "lon": xr.DataArray([[-30.0]], dims=("x", "y")),
        }
    )
    budget["salt"] = {
        "lambda": "so",
        "lhs": {"tendency": "salt_tendency"},
        "rhs": {},
    }
    return grid, budget, edges


def test_density_components_and_their_sum_are_all_mass_rates():
    """rho_ref cancels against the density lambda's own kg m-3.

    That cancellation is why the Boussinesq scaling is dimensionally invisible,
    and it is what makes the heat and salt components of a density
    transformation commensurable enough to add.
    """
    grid, budget, _ = _density_grid()
    wmt = xwmt.WaterMassTransformations(grid, budget, method="xhistogram")
    transformations = wmt.integrate_transformations(
        "sigma0", bins=np.linspace(0.0, 40.0, 9), sum_components=True
    )
    rates = [v for v in transformations.data_vars if not v.endswith("_bnds")]
    assert "tendency_heat" in rates and "tendency_salt" in rates and "tendency" in rates
    for name in rates:
        assert transformations[name].attrs["units"] == "kg s-1", name


@pytest.mark.parametrize("eos", ["teos10", "jmd95"])
def test_derived_units_do_not_depend_on_the_equation_of_state(eos):
    """`alpha`/`beta` units are derived, never read off the arrays xeos returns.

    xeos labels `beta` "1" for its practical-salinity backends and "kg g-1" for
    its absolute-salinity ones while returning values of the same magnitude, so
    reading the label would make identical arithmetic emit units differing by a
    factor of 1000 depending on which EOS the caller picked.
    """
    grid, budget, _ = _density_grid()
    wmt = xwmt.WaterMassTransformations(grid, budget, eos=eos, method="xhistogram")
    transformations = wmt.integrate_transformations(
        "sigma0", bins=np.linspace(0.0, 40.0, 9), sum_components=True
    )
    assert transformations["tendency_salt"].attrs["units"] == "kg s-1"
    assert transformations["tendency_heat"].attrs["units"] == "kg s-1"


def test_generic_tracer_over_a_mass_fraction_is_a_mass_rate():
    """A generic tracer can legitimately be a mass rate -- it depends on the lambda."""
    grid, budget, edges = _generic_tracer_grid(lambda_units="mol kg-1")
    wmt = xwmt.WaterMassTransformations(grid, budget, method="xhistogram")
    transformations = wmt.integrate_transformations("dic", bins=edges)
    assert transformations["tendency"].attrs["units"] == "kg s-1"


def test_units_fall_back_to_the_recipes_declaration():
    """An unlabelled tendency still resolves if the recipe declares the budget's units.

    That is the recipe author asserting the convention, not xwmt guessing, so it
    is recorded as a different `xwmt_units_source`.
    """
    grid, budget, edges = _heat_tendency_grid()
    grid._ds["temperature"].attrs.update({"units": "degC"})
    budget["heat"]["units"] = "W"
    wmt = xwmt.WaterMassTransformations(
        grid, budget, cp=1.0, rho_ref=1.0, method="xhistogram"
    )
    da = wmt.integrate_transformations("heat", bins=edges)["tendency"]
    assert da.attrs["units"] == "kg s-1"
    assert da.attrs["xwmt_units_source"] == "recipe"


def test_a_labelled_tendency_beats_the_recipes_declaration():
    """The data wins over the declaration: it is the more specific statement."""
    grid, budget, edges = _heat_tendency_grid()
    grid._ds["temperature"].attrs.update({"units": "degC"})
    grid._ds["heat_tendency"].attrs.update({"units": "W m-2"})
    budget["heat"]["units"] = "W"
    wmt = xwmt.WaterMassTransformations(
        grid, budget, cp=1.0, rho_ref=1.0, method="xhistogram"
    )
    da = wmt.integrate_transformations("heat", bins=edges)["tendency"]
    assert da.attrs["units"] == "m-2 kg s-1"
    assert da.attrs["xwmt_units_source"] == "tendency"


def test_undeterminable_units_are_omitted_not_guessed():
    """No `units` key at all, and one warning saying why -- never a wrong string."""
    grid, budget, edges = _heat_tendency_grid()
    grid._ds["temperature"].attrs.update({"units": "degC"})
    # No units on `heat_tendency`, and no budget-level declaration.
    wmt = xwmt.WaterMassTransformations(
        grid, budget, cp=1.0, rho_ref=1.0, method="xhistogram"
    )
    with pytest.warns(UserWarning, match="Could not determine the units"):
        da = wmt.integrate_transformations("heat", bins=edges)["tendency"]
    assert "units" not in da.attrs
    assert da.attrs["xwmt_units_source"] == "unknown"
    # Everything else is still described.
    assert da.attrs["long_name"].startswith("Water mass transformation rate")


def test_summing_terms_with_different_units_declines_to_label_the_sum():
    """Adding W to W m-2 is a real physics warning, so it earns one.

    The terms are still summed -- the numbers are what they have always been --
    but the result may not claim either summand's units.
    """
    grid, budget, edges = _heat_tendency_grid()
    grid._ds["temperature"].attrs.update({"units": "degC"})
    grid._ds["extensive"] = grid._ds["heat_tendency"].copy()
    grid._ds["extensive"].attrs.update({"units": "W"})
    grid._ds["intensive"] = grid._ds["heat_tendency"].copy()
    grid._ds["intensive"].attrs.update({"units": "W m-2"})
    # Both on the same side of the budget, so `group_processes` sums them
    # together rather than into separate groups.
    budget["heat"]["lhs"] = {}
    budget["heat"]["rhs"] = {"extensive": "extensive", "intensive": "intensive"}
    wmt = xwmt.WaterMassTransformations(
        grid, budget, cp=1.0, rho_ref=1.0, method="xhistogram"
    )
    with pytest.warns(UserWarning, match="different units"):
        transformations = wmt.integrate_transformations(
            "heat", bins=edges, group_processes=True
        )
    # The contributing terms keep their own, correct, units...
    assert transformations["extensive"].attrs["units"] == "kg s-1"
    assert transformations["intensive"].attrs["units"] == "m-2 kg s-1"
    # ...but their sum may not claim either.
    grouped = transformations["material_transformation"]
    assert "units" not in grouped.attrs
    assert "xwmt_units_source" not in grouped.attrs
    assert grouped.attrs["xwmt_summed_terms"] == "extensive, intensive"


def test_real_budget_output_is_clean_and_carries_xbudget_provenance(
    baltic_grid_and_budgets,
):
    """
    End-to-end on real MOM6 output, where the attributes to leak are real.

    The tendencies xbudget builds are in W and kg s-1 and carry MOM6's own
    `units`/`cell_methods`, none of which may reach the transformation rates. It
    also pins the other direction: xbudget's `provenance` *does* reach the output,
    which is what lets a user see that e.g. the "negative rhs" term is
    -1 x cp x tos x boundary_forcing_h_tendency x rho x area.
    """
    grid, budgets = baltic_grid_and_budgets
    wmt = xwmt.WaterMassTransformations(grid, budgets)
    transformations = wmt.integrate_transformations(
        "sigma2", bins=np.linspace(15.0, 19.0, 5), group_processes=True
    )

    rates = [v for v in transformations.data_vars if not v.endswith("_bnds")]
    for name in rates:
        attrs = transformations[name].attrs
        assert attrs["units"] == "kg s-1"
        assert attrs["long_name"].startswith("Water mass transformation rate")
        for leaked in ("cell_measures", "time_avg_info", "standard_name"):
            assert leaked not in attrs, f"{name} leaked {leaked}"

    negative_rhs = transformations["surface_ocean_flux_advective_negative_rhs_salt"]
    assert (
        negative_rhs.attrs["xwmt_source_variables"]
        == "salt_rhs_surface_ocean_flux_advective_negative_rhs"
    )
    assert negative_rhs.attrs["xbudget_path"].startswith("salt/rhs/")
    assert negative_rhs.attrs["xwmt_budget_side"] == "rhs"
    # Flattened from xbudget's mixed str/float provenance list.
    assert "boundary_forcing_h_tendency" in negative_rhs.attrs["provenance"]
    assert isinstance(negative_rhs.attrs["provenance"], str)


def test_real_budget_output_round_trips_to_netcdf(baltic_grid_and_budgets, tmp_path):
    """Mixed-type xbudget provenance is not netCDF-serializable unless flattened."""
    grid, budgets = baltic_grid_and_budgets
    wmt = xwmt.WaterMassTransformations(grid, budgets)
    transformations = wmt.integrate_transformations(
        "sigma2", bins=np.linspace(15.0, 19.0, 5)
    )
    path = tmp_path / "baltic_wmt.nc"
    transformations.to_netcdf(path)
    with xr.open_dataset(path) as roundtripped:
        assert roundtripped["diffusion"].attrs["units"] == "kg s-1"
        assert roundtripped["sigma2_l_target"].attrs["units"] == "kg m-3"


# ---------------------------------------------------------------------------
# WaterMass-derived fields
# ---------------------------------------------------------------------------


@pytest.fixture
def water_mass():
    grid = minimal_grid()
    # Tag the thickness the way a MOM6 diagnostic is tagged, so we can prove those
    # attributes do not ride along onto the fields derived from it.
    grid._ds["dz"].attrs.update(
        {
            "long_name": "Cell Thickness",
            "units": "m",
            "standard_name": "cell_thickness",
            "cell_methods": "area:mean zl:sum yh:mean xh:mean time: mean",
            "cell_measures": "area: area_t",
            "time_avg_info": "average_T1,average_T2,average_DT",
        }
    )
    wm = xwmt.WaterMass(grid, t_name="temperature", s_name="so", h_name="dz")
    wm.get_density("sigma2")
    return wm


def test_pressure_does_not_inherit_thickness_metadata(water_mass):
    """`p` is in dbar, derived from a thickness in metres via a height."""
    p = water_mass.grid._ds["p"]
    assert p.attrs["units"] == "dbar"
    assert p.attrs["standard_name"] == "sea_water_pressure"
    assert p.attrs["long_name"] == "sea water pressure"
    # Pressure increases downward, so the `positive: "up"` of the height field it
    # is derived from must not carry over either.
    assert "positive" not in p.attrs
    for leaked in ("cell_methods", "cell_measures", "time_avg_info"):
        assert leaked not in p.attrs


def test_height_coordinates_are_described(water_mass):
    ds = water_mass.grid._ds
    for name in ("z", "z_interface"):
        assert ds[name].attrs["units"] == "m"
        # These run from 0 at the free surface to negative values at depth.
        assert ds[name].attrs["positive"] == "up"
        assert ds[name].min() <= 0.0
        assert "standard_name" not in ds[name].attrs


def test_potential_density_anomaly_is_described(water_mass):
    sigma2 = water_mass.grid._ds["sigma2"]
    assert sigma2.attrs["units"] == "kg m-3"
    assert "2000 dbar" in sigma2.attrs["long_name"]


def test_eos_metadata_is_not_overwritten(water_mass):
    """xeos describes alpha/beta better than xwmt could; only leakage is cleared."""
    ds = water_mass.grid._ds
    for name in ("alpha", "beta"):
        assert ds[name].attrs["long_name"] == (
            "thermal expansion coefficient"
            if name == "alpha"
            else "haline contraction coefficient"
        )
        for leaked in ("cell_methods", "cell_measures", "time_avg_info"):
            assert leaked not in ds[name].attrs


def test_user_supplied_alpha_beta_are_left_alone():
    """A caller-supplied coefficient carries its author's metadata; xwmt must not strip it."""
    grid = minimal_grid()
    for name, units in (("alpha", "K-1"), ("beta", "kg g-1")):
        grid._ds[name] = xr.DataArray(
            np.full((1, 1, 1), 1e-4), dims=("x", "y", "z_l")
        ).assign_attrs({"units": units, "cell_methods": "time: mean", "mine": "yes"})
    wm = xwmt.WaterMass(grid, t_name="temperature", s_name="so", h_name="dz")
    wm.get_density("sigma2")
    assert wm.grid._ds["alpha"].attrs == {
        "units": "K-1",
        "cell_methods": "time: mean",
        "mine": "yes",
    }


def test_interface_thickness_keeps_units_but_not_vertical_cell_method(water_mass):
    h_i = water_mass.grid._ds["dz_i"]
    assert h_i.attrs["units"] == "m"
    assert h_i.attrs["standard_name"] == "cell_thickness"
    assert "interfaces" in h_i.attrs["long_name"]
    # The field moved from layer centers to interfaces, so the source's vertical
    # cell method no longer describes it.
    assert "cell_methods" not in h_i.attrs


# ---------------------------------------------------------------------------
# attrs helpers
# ---------------------------------------------------------------------------


def test_set_default_attrs_never_overwrites():
    da = xr.DataArray([1.0], attrs={"units": "already set"})
    xattrs.set_default_attrs(da, {"units": "kg s-1", "long_name": "new"})
    assert da.attrs["units"] == "already set"
    assert da.attrs["long_name"] == "new"


def test_set_default_attrs_skips_none():
    da = xr.DataArray([1.0])
    xattrs.set_default_attrs(da, {"standard_name": None})
    assert "standard_name" not in da.attrs


def test_netcdf_safe_flattens_mixed_provenance():
    """xbudget provenance mixes variable names with numeric constants."""
    value = [-1.0, 3992.0, "tos", "boundary_forcing_h_tendency", 1035.0, "areacello"]
    assert xattrs.netcdf_safe(value) == (
        "-1.0, 3992.0, tos, boundary_forcing_h_tendency, 1035.0, areacello"
    )
    # Homogeneously numeric sequences stay numeric, since those serialize fine.
    assert xattrs.netcdf_safe([1.0, 2.0]) == [1.0, 2.0]
    # Nested lists are flattened rather than stringified verbatim.
    assert xattrs.netcdf_safe([["a", "b"], "c"]) == "a, b, c"


def test_lambda_coord_units_fall_back_for_derived_densities():
    """Densities are derived by xwmt, so there is no source variable to inherit from."""
    assert xattrs.lambda_coord_attrs("sigma2")["units"] == "kg m-3"
    assert xattrs.lambda_coord_attrs("rho")["units"] == "kg m-3"
    # An unknown tracer with no source metadata gets no invented units.
    assert "units" not in xattrs.lambda_coord_attrs("some_tracer")


def test_strip_inherited_attrs_can_keep_identity():
    da = xr.DataArray(
        [1.0],
        attrs={"units": "degC-1", "cell_methods": "time: mean", "long_name": "alpha"},
    )
    xattrs.strip_inherited_attrs(da, identity=False)
    assert da.attrs == {"units": "degC-1", "long_name": "alpha"}
    xattrs.strip_inherited_attrs(da)
    assert da.attrs == {}
