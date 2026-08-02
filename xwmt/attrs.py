"""CF-style metadata for the variables `xwmt` produces.

This module is the single source of truth for the attributes attached to xwmt
output: the water mass transformation rates returned by
:class:`~xwmt.wmt.WaterMassTransformations`, the lambda bin coordinate they are
indexed by, and the intermediate fields :class:`~xwmt.wm.WaterMass` derives onto
its own copy of the dataset.

Every one of those is computed from model diagnostics, so none of them may keep
the attributes of the fields it was computed from: a transformation rate in
kg s-1 derived from a tendency in W must not describe itself as W. The
two guards for that are :func:`strip_inherited_attrs`, which clears what
arithmetic dragged along, and :func:`set_default_attrs`, which fills in xwmt's
own description without displacing anything an upstream package set deliberately.

Conventions followed here:

- Units strings are UDUNITS-2 parseable, as the CF conventions require: ``"kg
  s-1"``, never ``"kg/s"``.
- ``standard_name`` is only set where a genuine CF standard name exists. There
  is no CF standard name for a water mass transformation rate, so transformation
  variables get ``long_name``/``units``/``comment`` and no invented standard name.
- Per-process prose stays *out* of xwmt. The `long_name` of a transformation term
  is generated mechanically from the term key, and whatever descriptive metadata
  the upstream `xbudget` package stamped onto the source tendency (`provenance`,
  `xbudget_path`, `xbudget_op`) is propagated verbatim. Any metadata xbudget
  gains later therefore flows through for free, and xwmt does not end up owning
  a duplicate glossary of model-specific term names.
- Attributes set by upstream packages are never overwritten; see
  :func:`set_default_attrs`.

What units a transformation is in
---------------------------------
Derived, not assumed. :meth:`~xwmt.wmt.WaterMassTransformations._transformation_units`
walks the same dimensional operations the numerics do -- there are only six, and
they are tabulated in `CLAUDE.md` -- using :mod:`xwmt.units`. The three shapes
it reduces to are::

    heat     U(tendency) / U(cp) / U(theta)         W / (J kg-1 degC-1) / degC  = kg s-1
    salt     U(tendency) / 1000 / U(S)              kg s-1 / 1000 / 0.001       = kg s-1
    generic  U(tendency) / U(lambda)                mol s-1 / (mol m-3)         = m3 s-1

A density lambda reduces to the heat and salt cases: the `rho_ref` factor
cancels against the density's own ``kg m-3``, which is exactly why the
Boussinesq scaling is dimensionally invisible, and `alpha`/`beta` cancel the
tracer's units because they are its reciprocal.

With the xbudget conventions -- cell area already multiplied in, so heat
tendencies arrive in W and salt tendencies in kg s-1 -- every rate is a **mass**
tendency in ``kg s-1``. A tracer budget that never multiplies by a density is a
**volume** tendency in ``m3 s-1`` instead. Those are the only two answers a
water mass transformation can have; anything else means an input was not what
the conventions assume, and now says so rather than being relabelled ``kg s-1``.

Two conventions are worth stating because they are what make the algebra close:

- **Practical salinity is dimensionless with a scale of 1e-3**, the same unit as
  ``g kg-1``. That is what the ``* 1000.0`` in `datadict` pairs with. See the
  :mod:`xwmt.units` docstring, which also explains why this differs from the
  alias `xbudget.units` uses.
- **`cp` and `rho_ref` units are fixed by xwmt's own API contract**, not read
  from a recipe's `constants` table -- xwmt uses the *values* its caller passed,
  so taking units from a table it did not take the value from would be a
  category error.

When the units of an input cannot be determined, the `units` key is **omitted**
rather than guessed, and `xwmt_units_source` records which authority answered
(``"tendency"``, ``"recipe"`` or ``"unknown"``). A rate labelled with the wrong
units is worse than one labelled with none.

None of this varies between the two entry points: `integrate_transformations`
additionally sums over the horizontal, so its result is a domain total, while
`map_transformations` leaves the horizontal dimensions in place and so returns a
total *per grid cell* rather than a flux density per square metre. That is a
difference in what is summed, not in dimension -- xwmt never multiplies by cell
area -- so both carry the same `units`, and the mapped variant distinguishes
itself through `cell_methods` and its `comment`.
"""

import numbers

from xwmt.version import __version__

__all__ = [
    "set_default_attrs",
    "strip_inherited_attrs",
    "netcdf_safe",
    "prettify",
    "transformation_attrs",
    "summed_term_attrs",
    "lambda_coord_attrs",
    "bin_bounds_attrs",
    "dataset_attrs",
    "derived_field_attrs",
    "pressure_attrs",
]


#: Attributes copied straight through from the source xbudget tendency variable,
#: so that upstream provenance metadata reaches the output without xwmt having to
#: understand it.
_XBUDGET_PASSTHROUGH = ("provenance", "xbudget_path", "xbudget_op")

#: Fallback units for lambdas that xwmt derives itself and so have no source
#: variable to inherit `units` from.
_LAMBDA_UNITS = {
    "rho": "kg m-3",
    "sigma0": "kg m-3",
    "sigma1": "kg m-3",
    "sigma2": "kg m-3",
    "sigma3": "kg m-3",
    "sigma4": "kg m-3",
}

_REFERENCES = (
    "Walin, G. (1982), doi:10.1111/j.2153-3490.1982.tb01806.x; "
    "Drake, H. F., et al. (2025), doi:10.1029/2024MS004383"
)

_DISCLAIMER = (
    "xwmt does not check that mass or tracers are conserved. It is the user's "
    "responsibility to ensure that the input budgets are closed; improperly "
    "conserved fields yield incorrect transformation rates."
)


def set_default_attrs(da, attrs):
    """Attach `attrs` to `da`, without overwriting anything already set.

    Upstream packages annotate their own output -- `xeos`, for instance, stamps
    `units`/`long_name`/`standard_name` onto the `alpha`, `beta` and density
    fields it returns. Those are more authoritative than anything xwmt could
    infer, so only genuinely missing keys are filled in. Keys whose value is
    None are skipped, which lets callers pass optional attributes
    unconditionally.

    Returns `da` for convenient chaining.
    """
    for key, value in attrs.items():
        if value is None or key in da.attrs:
            continue
        da.attrs[key] = netcdf_safe(value)
    return da


#: Attributes an equation-of-state backend sets on the coefficients it returns.
#: `xeos` gives `alpha`/`beta` a `units` and a `long_name` and no `standard_name`
#: (the CF table has no name for a haline contraction coefficient), so those two
#: are the only keys on a coefficient that can have come from xeos rather than
#: from the temperature it was computed from.
EOS_COEFFICIENT_ATTRS = ("units", "long_name")

#: The same for a density, which `xeos` additionally gives a real
#: `standard_name` -- and does so by overwriting whatever the input contributed,
#: so keeping it is safe here in a way it is not for the coefficients.
EOS_DENSITY_ATTRS = ("units", "long_name", "standard_name")


def strip_inherited_attrs(da, keep=()):
    """Drop every attribute `da` inherited from the fields it was derived from.

    An allowlist, not a blocklist, and deliberately so. The previous shape of
    this function named the attributes to *remove*, which meant every attribute
    nobody had thought of survived onto a field it did not describe: a layer
    thickness's `valid_range: [0, 10000]` (metres) ended up on a pressure in
    dbar and on a density in kg m-3, and an input temperature's `interp_method`
    and `grid_mapping` ended up on everything derived from it. A derived field is
    a *different quantity*, so nothing carries over unless something states that
    it should.

    `keep` names the attributes that may survive because an upstream package set
    them deliberately -- see :data:`EOS_COEFFICIENT_ATTRS` and
    :data:`EOS_DENSITY_ATTRS`. Note it cannot be `("units", "long_name",
    "standard_name")` for `alpha`/`beta`: xeos sets no `standard_name` on those,
    so anything found there leaked in, and a thermal expansion coefficient in
    K-1 announcing itself as `sea_water_potential_temperature` is exactly the
    kind of claim a CF-aware reader acts on.

    Returns `da` for convenient chaining.
    """
    keep = set(keep)
    for key in [k for k in da.attrs if k not in keep]:
        del da.attrs[key]
    return da


def netcdf_safe(value):
    """Coerce `value` into something the netCDF attribute model can hold.

    netCDF attributes are scalars, strings, or homogeneously-typed numeric
    sequences. xbudget's `provenance`/`xbudget_path` are lists that freely mix
    variable names with the numeric constants used to build a term, e.g.
    ``[-1.0, 3992.0, 'tos', 'boundary_forcing_h_tendency', 1035.0,
    'areacello']``. Writing that to netCDF raises, so mixed and nested
    sequences are flattened to a comma-joined string. Homogeneous numeric
    sequences are left alone, since those serialize fine and are more useful
    as numbers.
    """
    if isinstance(value, (str, numbers.Number)):
        return value
    if isinstance(value, (list, tuple)):
        flat = list(_flatten(value))
        if flat and all(
            isinstance(v, numbers.Number) and not isinstance(v, bool) for v in flat
        ):
            return flat
        return ", ".join(str(v) for v in flat)
    return str(value)


def _flatten(seq):
    for item in seq:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


def prettify(term):
    """Render a snake_case term key as human-readable prose.

    ``"surface_ocean_flux_advective_negative_rhs"`` ->
    ``"surface ocean flux advective negative rhs"``.
    """
    return str(term).replace("_", " ")


def collect_source_attrs(source_arrays):
    """Merge the xbudget passthrough attributes of one or more source tendencies.

    `source_arrays` maps a source variable name to the corresponding
    `xr.DataArray` (or None, for a component that this budget does not provide).
    A density transformation is built from both a heat and a salt tendency, so
    each passthrough attribute can have two values; they are joined rather than
    one silently winning.
    """
    names = [name for name, da in source_arrays.items() if da is not None]
    attrs = {"xwmt_source_variables": ", ".join(names) if names else None}
    for key in _XBUDGET_PASSTHROUGH:
        values = [
            str(netcdf_safe(da.attrs[key]))
            for da in source_arrays.values()
            if da is not None and key in da.attrs
        ]
        # Preserve order while dropping duplicates (heat and salt tendencies
        # often share e.g. the same `xbudget_op`).
        seen = {}
        for value in values:
            seen.setdefault(value, None)
        attrs[key] = "; ".join(seen) if seen else None
    if isinstance(attrs.get("xbudget_path"), str):
        # xbudget_path arrives as ["heat", "rhs", "diffusion"], flattened by
        # netcdf_safe to "heat, rhs, diffusion". A slash-separated path reads
        # better and is unambiguous when two budgets are joined by "; ".
        attrs["xbudget_path"] = attrs["xbudget_path"].replace(", ", "/")
    return attrs


def budget_side(source_attrs, recipe, term):
    """Which side of the tracer budget `term` sits on: ``"lhs"``, ``"rhs"``, or None.

    This is the distinction between *kinematic* transformation (left-hand side,
    the resolved tendency and advective terms) and *material* transformation
    (right-hand side, the process terms), so it is worth recording per variable.
    Preferred source is xbudget's own `xbudget_path`; the recipe is consulted as
    a fallback for older xbudget versions that do not stamp it.
    """
    path = source_attrs.get("xbudget_path")
    if isinstance(path, str):
        for side in ("lhs", "rhs"):
            if f"/{side}/" in path:
                return side
    for budget in recipe.values():
        if not isinstance(budget, dict):
            continue
        for side in ("lhs", "rhs"):
            if isinstance(budget.get(side), dict) and term in budget[side]:
                return side
    return None


def _transformation_comment(lam_name, integrate):
    comment = (
        f"Water mass transformation rate in the Walin (1982) framework: the "
        f"extensive tracer tendency binned in {lam_name} along the vertical and "
        f"divided by the bin width. Positive values denote net transformation "
        f"toward lower {lam_name}."
    )
    if not integrate:
        comment += (
            " Values are totals per grid cell, not per unit area: the horizontal "
            "area factor is carried by the input budget terms."
        )
    return comment


def _cell_methods(horizontal_dims, zdim, integrate):
    dims = ([*horizontal_dims] if integrate else []) + [zdim]
    return " ".join(f"{dim}: sum" for dim in dims if dim is not None)


def _long_name(lam_name, description, component=None):
    name = f"Water mass transformation rate across {lam_name} surfaces {description}"
    if component in ("heat", "salt"):
        name += f" ({component} component)"
    return name


def transformation_attrs(
    lam_name,
    lam_var,
    term,
    component,
    method,
    integrate,
    horizontal_dims,
    zdim,
    *,
    source_attrs=None,
    side=None,
    units=None,
    units_source=None,
):
    """Attributes for a single transformation-rate variable.

    Parameters
    ----------
    lam_name : str
        The lambda the transformation is across, as the user named it (e.g.
        "sigma2", "heat").
    lam_var : str
        The dataset variable holding that lambda (e.g. "sigma2", "thetao").
    term : str
        Process term key from the xbudget recipe (e.g. "diffusion").
    component : str or None
        "heat"/"salt" for the two halves of a density transformation, "total"
        once they have been summed, None for a non-density lambda.
    method : str
        The transformation backend actually used for this call ("xhistogram" or
        "xgcm"), which is not necessarily `self.method` -- a prebinned target
        forces "xgcm".
    integrate : bool
        Whether the horizontal dimensions were summed over.
    horizontal_dims, zdim : list of str, str
        Dimension names, used to build a CF `cell_methods` string.
    source_attrs : dict, optional
        Output of :func:`collect_source_attrs` for the underlying tendencies.
    side : str, optional
        "lhs"/"rhs", from :func:`budget_side`.
    units : str, optional
        Units of the rate, derived by
        :meth:`~xwmt.wmt.WaterMassTransformations._transformation_units`. None
        when they could not be determined, in which case the key is omitted
        rather than guessed.
    units_source : str, optional
        Which authority supplied `units` -- "tendency", "recipe" or "unknown".
    """
    source_attrs = source_attrs or {}
    attrs = {
        "units": units,
        "xwmt_units_source": units_source,
        "long_name": _long_name(lam_name, f"due to {prettify(term)}", component),
        "cell_methods": _cell_methods(horizontal_dims, zdim, integrate),
        "comment": _transformation_comment(lam_name, integrate),
        "xwmt_lambda": lam_name,
        "xwmt_lambda_variable": lam_var,
        "xwmt_term": term,
        "xwmt_component": component,
        "xwmt_budget_side": side,
        "xwmt_method": method,
        "xwmt_version": __version__,
        **source_attrs,
    }
    return {k: v for k, v in attrs.items() if v is not None}


#: Prose for the two process groupings produced by `group_processes=True`.
_GROUP_DESCRIPTIONS = {
    "kinematic_transformation": (
        "summed over all left-hand-side (kinematic) budget terms",
        "lhs",
    ),
    "material_transformation": (
        "summed over all right-hand-side (material) budget terms",
        "rhs",
    ),
}


def summed_term_attrs(base_attrs, newterm, summed_terms, component=None):
    """Attributes for a variable built by summing other transformation variables.

    Covers both groupings xwmt performs: heat + salt components of one process,
    and all processes on one side of the budget. `base_attrs` are the attributes
    of one contributing variable, which supply the parts that are shared
    (lambda, units, cell_methods, comment).
    """
    attrs = {
        key: value
        for key, value in base_attrs.items()
        # Provenance of a contributing term does not describe the sum; the
        # `xwmt_summed_terms` list below is the sum's provenance.
        if key not in _XBUDGET_PASSTHROUGH + ("xwmt_source_variables", "xwmt_term")
    }
    lam_name = base_attrs.get("xwmt_lambda", "lambda")

    stem = newterm
    if component in ("heat", "salt"):
        stem = newterm[: -len(f"_{component}")]
    if stem in _GROUP_DESCRIPTIONS:
        description, side = _GROUP_DESCRIPTIONS[stem]
        attrs["xwmt_budget_side"] = side
    else:
        description = f"due to {prettify(stem)}"
        attrs["xwmt_term"] = stem

    attrs["long_name"] = _long_name(lam_name, description, component)
    attrs["xwmt_component"] = component
    attrs["xwmt_summed_terms"] = ", ".join(summed_terms)
    return {k: v for k, v in attrs.items() if v is not None}


def lambda_coord_attrs(lam_name, source=None, bounds=None):
    """Attributes for the ``{lambda}_l_target`` bin-centre coordinate.

    `units`/`long_name`/`standard_name` are inherited from the lambda field
    itself when it carries them (a model's `thetao` knows it is in degC), with
    :data:`_LAMBDA_UNITS` covering the densities xwmt derives internally.
    """
    inherited = dict(source.attrs) if source is not None else {}
    attrs = {
        "units": inherited.get("units", _LAMBDA_UNITS.get(lam_name)),
        "long_name": f"{inherited.get('long_name', lam_name)} bin centers",
        "standard_name": inherited.get("standard_name"),
        "bounds": bounds,
        "comment": (
            f"Centers of the target {lam_name} bins that the transformation rates "
            f"were binned into. Bin edges are given by the `bounds` variable."
        ),
    }
    return {k: v for k, v in attrs.items() if v is not None}


def bin_bounds_attrs(lam_name, source=None):
    """Attributes for the CF `bounds` variable holding the lambda bin edges."""
    inherited = dict(source.attrs) if source is not None else {}
    attrs = {
        "units": inherited.get("units", _LAMBDA_UNITS.get(lam_name)),
        "long_name": f"{inherited.get('long_name', lam_name)} bin edges",
    }
    return {k: v for k, v in attrs.items() if v is not None}


def dataset_attrs(lam_name, method, integrate):
    """Global attributes for a returned transformation Dataset."""
    return {
        "title": f"Water mass transformation rates across {lam_name} surfaces",
        "comment": _DISCLAIMER,
        "references": _REFERENCES,
        "source": f"xwmt {__version__} (https://github.com/NOAA-GFDL/xwmt)",
        "xwmt_version": __version__,
        "xwmt_lambda": lam_name,
        "xwmt_method": method,
        # netCDF has no boolean attribute type.
        "xwmt_integrate": "true" if integrate else "false",
    }


#: Metadata for the intermediate fields `WaterMass` derives onto its own copy of
#: the dataset. `alpha`, `beta` and the density variables are deliberately absent
#: for the in-situ case: `xeos` already annotates those, and `set_default_attrs`
#: leaves them alone. `sigmaX` *is* listed because the ``density - 1000``
#: subtraction that turns density into a potential-density anomaly drops the
#: attributes xeos set (and would leave a now-wrong "in-situ density" long_name).
_DERIVED_FIELD_ATTRS = {
    # No `comment` naming how the pressure was obtained: that now depends on the
    # `gravity` argument, so `pressure_attrs` fills it in per call rather than
    # this table asserting one of the two paths unconditionally.
    "p": {
        "units": "dbar",
        "long_name": "sea water pressure",
        "standard_name": "sea_water_pressure",
    },
    # `z` and `z_interface` are heights, not depths: they are built by
    # accumulating layer thickness downward with a negative sign, so they run
    # from 0 at the free surface to large negative values at depth. Hence
    # `positive: "up"` -- which is also the convention `gsw.p_from_z` expects,
    # and how `p` is derived from them.
    "z": {
        "units": "m",
        "long_name": "height of layer centers relative to the geoid",
        "positive": "up",
        "comment": "Negative below the sea surface.",
    },
    "z_interface": {
        "units": "m",
        "long_name": "height of layer interfaces relative to the geoid",
        "positive": "up",
        "comment": "Negative below the sea surface.",
    },
}


def pressure_attrs(gravity):
    """Attributes for the sea pressure `WaterMass` derives from depth.

    `gravity` is `WaterMass.gravity`: a number for the Boussinesq hydrostatic
    pressure, or ``"gsw"`` for `gsw.p_from_z`'s latitude-dependent gravity. Which
    of the two ran is worth recording, because they do not agree -- a constant
    9.81 differs from gsw's local gravity by a few parts in a thousand, which is
    a real (if small) difference in the pressure the equation of state is
    evaluated at.
    """
    attrs = derived_field_attrs("p")
    if gravity == "gsw":
        attrs["comment"] = (
            "Derived from height via gsw.p_from_z, whose gravity varies with latitude."
        )
    else:
        attrs["comment"] = (
            f"Boussinesq hydrostatic pressure from height, with a constant "
            f"gravitational acceleration of {gravity} m s-2."
        )
    return attrs


def derived_field_attrs(name):
    """Attributes for a `WaterMass`-derived field, or ``{}`` if none are defined."""
    if name in _DERIVED_FIELD_ATTRS:
        return dict(_DERIVED_FIELD_ATTRS[name])
    if name.startswith("sigma") and name[len("sigma") :].isdigit():
        ref_dbar = int(name[len("sigma") :]) * 1000
        return {
            "units": "kg m-3",
            "long_name": (
                f"potential density anomaly referenced to {ref_dbar} dbar "
                f"(density minus 1000 kg m-3)"
            ),
        }
    return {}


def synthetic_thickness_attrs(interface):
    """Attributes for the placeholder thickness of a synthesized single-layer Z axis.

    Purely 2D/surface datasets have no vertical axis, so `WaterMass` invents a
    single unit-thickness layer to transform in. Labelling it dimensionless makes
    clear that it is a placeholder rather than a physical layer thickness.
    """
    position = "interfaces" if interface else "centers"
    return {
        "units": "1",
        "long_name": (
            f"placeholder unit thickness at layer {position} for the synthesized "
            f"single-layer vertical axis of surface-only data"
        ),
    }


def thickness_interface_attrs(source=None):
    """Attributes for the layer thickness interpolated onto vertical interfaces.

    `units` and `standard_name` carry over from the layer thickness itself -- it is
    still a thickness, on the same horizontal grid. `cell_methods` deliberately does
    not: the field has moved from layer centers to interfaces, so the source's
    vertical cell method no longer describes it.
    """
    inherited = dict(source.attrs) if source is not None else {}
    attrs = {
        "units": inherited.get("units", "m"),
        "long_name": "cell thickness at vertical interfaces",
        "standard_name": inherited.get("standard_name"),
    }
    return {k: v for k, v in attrs.items() if v is not None}
