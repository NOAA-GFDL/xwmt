"""Tests for `xwmt.units`, the unit algebra behind the output `units` attribute.

Fast and synthetic -- nothing here touches a dataset. The end-to-end checks that
the algebra is wired into the right six places live in `test_attrs.py`.
"""

import pytest

from cf_units import Unit

from xwmt import units as u


class TestParse:
    def test_round_trips_the_strings_xwmt_emits(self):
        for spec in ["kg s-1", "m3 s-1", "W", "W m-2", "kg m-3", "degC", "1"]:
            assert u.format_units(u.parse(spec)) == spec

    def test_psu_is_dimensionless_with_a_scale_of_one_thousandth(self):
        # The whole salt branch rests on this: PSS-78 carries no dimension but
        # is scaled so that 35 psu is 35 g of salt per kg of seawater.
        parsed = u.parse("psu")
        assert parsed is not None
        assert parsed == Unit("g kg-1")
        assert parsed.is_dimensionless()

    @pytest.mark.parametrize(
        "spelling", ["psu", "PSU", "pss-78", "PSS-78", "practical_salinity_units"]
    )
    def test_psu_spellings_all_alias(self, spelling):
        assert u.parse(spelling) == Unit("0.001")

    def test_cf_units_alone_cannot_parse_psu(self):
        # Documents why the alias table has to exist at all.
        with pytest.raises(ValueError):
            Unit("psu")

    @pytest.mark.parametrize("spec", [None, "", "   ", "not a unit", "unknown"])
    def test_unknown_inputs_give_none_rather_than_raising(self, spec):
        assert u.parse(spec) is None

    def test_dimensionless_is_not_unknown(self):
        # "1" is a real answer; None means "could not tell".
        assert u.parse("1") == Unit("1")

    def test_accepts_a_unit_instance(self):
        assert u.parse(Unit("kg s-1")) == Unit("kg s-1")


class TestAlgebra:
    def test_unknown_propagates_through_every_operation(self):
        known = u.parse("kg s-1")
        assert u.multiply(known, None) is None
        assert u.divide(None, known) is None
        assert u.divide(known, None) is None
        assert u.reciprocal(None) is None
        assert u.scale(None, 1000.0) is None

    def test_scaling_values_divides_the_units(self):
        # Multiplying an array's values by 1000 restates kg as g.
        assert u.scale(u.parse("kg s-1"), 1000.0) == Unit("g s-1")

    def test_reciprocal_of_salinity_is_the_haline_contraction_coefficient(self):
        assert u.reciprocal(u.parse("psu")) == Unit("kg g-1")

    def test_reciprocal_of_temperature_is_the_thermal_expansion_coefficient(self):
        assert u.reciprocal(u.parse("degC")) == Unit("K-1")

    def test_format_uses_spaces_not_the_cf_units_dot(self):
        assert u.format_units(Unit("kg s-1")) == "kg s-1"
        assert "." not in u.format_units(u.divide(u.parse("W"), u.parse("J kg-1")))

    def test_same_units_ignores_unknowns(self):
        assert u.same_units(["kg s-1", None, "kg s-1"])
        assert u.same_units([None, None])
        assert not u.same_units(["kg s-1", "m3 s-1"])

    def test_same_units_distinguishes_scale_not_just_dimension(self):
        # g s-1 and kg s-1 are convertible but not equal; a sum of the two
        # must not claim either label.
        assert not u.same_units(["kg s-1", "g s-1"])


class TestPipelineFormulas:
    """The three formulas the six dimensional sites in `wmt.py` reduce to.

    These pin the algebra independently of the plumbing, so a failure here is
    unambiguously a units bug rather than a wiring bug.
    """

    def _heat(self, tendency, lam="degC"):
        # mirrors wmt.py `calc_hlamdot_and_lambda` heat branch (/ cp)
        # then wmt.py `_transform_one` (/ bin width, in the lambda's units)
        per_cp = u.divide(u.parse(tendency), u.parse(u.CP_UNITS))
        return u.format_units(u.divide(per_cp, u.parse(lam)))

    def _salt(self, tendency, lam="psu"):
        # mirrors wmt.py `datadict` (values x1000) then / bin width
        grams = u.scale(u.parse(tendency), 1000.0)
        return u.format_units(u.divide(grams, u.parse(lam)))

    def _generic(self, tendency, lam):
        # mirrors wmt.py `calc_hlamdot_and_lambda` generic branch (no
        # conversion at all) then / bin width
        return u.format_units(u.divide(u.parse(tendency), u.parse(lam)))

    def test_conventional_heat_budget_is_a_mass_rate(self):
        assert self._heat("W") == "kg s-1"

    def test_conventional_salt_budget_is_a_mass_rate(self):
        assert self._salt("kg s-1") == "kg s-1"

    def test_generic_tracer_over_a_concentration_is_a_volume_rate(self):
        # The bug this module exists to fix: no density factor anywhere on the
        # generic path, so the answer is a volume tendency.
        assert self._generic("mol s-1", "mol m-3") == "m3 s-1"

    def test_generic_tracer_over_a_mass_fraction_is_a_mass_rate(self):
        # A generic tracer legitimately can be a mass rate -- it depends on
        # what the lambda is per.
        assert self._generic("mol s-1", "mol kg-1") == "kg s-1"

    def test_density_components_both_reduce_to_the_scalar_cases(self):
        # rho_ref cancels against the density lambda's own kg m-3, which is
        # why the Boussinesq scaling is dimensionally invisible.
        rho_ref, sigma = u.parse(u.RHO_REF_UNITS), u.parse("kg m-3")
        alpha = u.reciprocal(u.parse("degC"))
        beta = u.reciprocal(u.parse("psu"))

        heat = u.divide(u.parse("W"), u.parse(u.CP_UNITS))
        heat = u.divide(u.multiply(heat, alpha, rho_ref), sigma)

        salt = u.scale(u.parse("kg s-1"), 1000.0)
        salt = u.divide(u.multiply(salt, beta, rho_ref), sigma)

        assert u.format_units(heat) == "kg s-1"
        assert u.format_units(salt) == "kg s-1"

    def test_an_area_intensive_tendency_is_reported_as_such(self):
        # xwmt never multiplies by cell area; if the caller hands it a per-m2
        # tendency the result really is per-m2, and now says so.
        assert self._heat("W m-2") == "m-2 kg s-1"

    def test_salinity_labelled_dimensionless_keeps_its_scale(self):
        # Some files label `so` as "1" rather than "psu". The x1000 then has
        # nothing to cancel, and the honest answer says so rather than
        # silently claiming kg s-1.
        assert u.parse(self._salt("kg s-1", lam="1")) == Unit("g s-1")

    def test_cmip6_salinity_spelling_agrees_with_psu(self):
        # CMIP6 labels `so` as "0.001", which needs no alias and must land in
        # the same place as MOM6's "psu".
        assert self._salt("kg s-1", lam="0.001") == self._salt("kg s-1", lam="psu")


def test_every_emitted_string_is_udunits_parseable():
    """CF requires UDUNITS-2-parseable units; check the ones xwmt can emit.

    `cf_units` is a hard dependency via xbudget, but the importorskip matches
    the convention in the sibling `xeos` package and keeps this honest if that
    ever changes.
    """
    cf_units = pytest.importorskip("cf_units")
    emitted = [
        u.format_units(u.parse(s))
        for s in ["kg s-1", "m3 s-1", "kg m-2 s-1", "g s-1", "psu", "degC", "1"]
    ]
    for spec in emitted:
        cf_units.Unit(spec)  # raises ValueError if UDUNITS-2 cannot parse it
