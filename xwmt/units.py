"""Unit algebra for the quantities `xwmt` derives.

A thin layer over `cf_units` (which arrives with `xbudget >= 0.8.0`), not an
algebra of its own. It exists to hold three things xwmt cannot get from
cf-units directly: the practical-salinity alias, the spelling convention for
emitted strings, and the units of the two scalar constants xwmt's own API
defines.

Every function here is pure and total -- it never raises and never warns, it
just answers "what are the units?". Unknown (``None``) propagates: any operation
with an unknown operand is unknown, never a fabricated dimension. Whether an
unknown is *tolerable* is policy, and lives in `xwmt.wmt` rather than here. This
mirrors the contract in `xbudget.units`, deliberately, so that reasoning
transfers between the two packages.

Practical salinity
------------------
UDUNITS has no ``psu``, so it has to be aliased. xwmt aliases it to ``"0.001"``:
PSS-78 carries no dimension, but it is *scaled* -- the scale was designed so
that a practical salinity of 35 is numerically 35 grams of salt per kilogram of
seawater. cf-units agrees that this is the same unit as grams per kilogram::

    Unit("0.001") == Unit("g kg-1")   # True

That scale is exactly what the ``* 1000.0`` in `WaterMassTransformations.datadict`
pairs with: it restates a salt tendency in grams so that dividing by a salinity
bin width leaves a seawater mass rate. Read `psu` as dimensionless-with-scale-1
instead and that factor of 1000 has nothing to cancel against, and the salt
transformation rate comes out labelled a thousand times smaller than it is.

Note that `xbudget.units` aliases ``psu`` to ``"1"``. That is not a
disagreement about the physics: xbudget only asks whether two terms are
*convertible*, and says so explicitly -- "the residual 1e-3 from a
`unit_conversion` factor is a scale, not a dimension, and does not affect
convertibility". xwmt divides by that scale, so it has to keep it.

The coefficients
----------------
`alpha` and `beta` units are **derived** (`U(alpha) = U(theta)^-1`,
`U(beta) = U(S)^-1`), never read from the arrays `xeos` returns. `xeos` labels
`beta` as ``"1"`` for its practical-salinity backends and ``"kg g-1"`` for its
absolute-salinity ones while returning values of the same magnitude (~7.2e-4 to
~7.4e-4 across all of them), so reading the label would make xwmt's output
depend on which equation of state the caller picked. Deriving structurally also
covers a caller who supplies `alpha`/`beta` themselves.
"""

import re

from cf_units import Unit

#: A ``.`` separating two factors, as opposed to the decimal point of a leading
#: numeric scale. A separator always has a non-digit on at least one side
#: (``kg.s-1``, ``m2.s-1``); a decimal point always has digits on both.
_SEPARATOR_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")

__all__ = [
    "CP_UNITS",
    "RHO_REF_UNITS",
    "parse",
    "format_units",
    "multiply",
    "divide",
    "reciprocal",
    "scale",
    "same_units",
]

#: Units of the `cp` constructor argument of :class:`~xwmt.wm.WaterMass`. Fixed
#: by xwmt's own API contract -- `cp` is documented as a specific heat capacity
#: and the arithmetic at `rho_tend`/`calc_hlamdot_and_lambda` is only correct
#: for one choice of units -- so it is stated here rather than read from a
#: recipe's `constants` table, whose *value* xwmt does not use.
CP_UNITS = "J kg-1 degC-1"

#: Units of the `rho_ref` constructor argument. See :data:`CP_UNITS`.
RHO_REF_UNITS = "kg m-3"

#: UDUNITS has no practical-salinity unit. PSS-78 is dimensionless but scaled:
#: see the module docstring for why this is 1e-3 and not 1.
_ALIASES = {
    "psu": "0.001",
    "PSU": "0.001",
    "practical_salinity_unit": "0.001",
    "practical_salinity_units": "0.001",
    "pss-78": "0.001",
    "PSS-78": "0.001",
}


def parse(spec):
    """Parse a units string (or `Unit`) into a `Unit`, or `None` if unknown.

    `None` is returned for `None`, a blank string, an unparseable string, or
    cf-units' `unknown`/`no-unit` sentinels -- anything that is not a real
    dimension. That is deliberately distinct from dimensionless (`Unit("1")`),
    which is a perfectly good answer.
    """
    if spec is None:
        return None
    if isinstance(spec, Unit):
        return spec if _is_real(spec) else None
    text = str(spec).strip()
    if not text:
        return None
    text = _ALIASES.get(text, text)
    try:
        unit = Unit(text)
    except ValueError:
        return None
    return unit if _is_real(unit) else None


def _is_real(unit):
    """True unless `unit` is cf-units' `unknown`/`no-unit` sentinel."""
    return not (unit.is_unknown() or unit.is_no_unit())


def format_units(unit):
    """Render `unit` in the spelling xwmt emits, or `None` if unknown.

    cf-units separates factors with ``.`` (``"kg.s-1"``); CF and the rest of
    this package use a space (``"kg s-1"``). Both are UDUNITS-2 parseable, but
    only one matches what xwmt writes everywhere else, so every string that
    leaves this module goes through here.

    Only *separator* dots are rewritten. A dimensionless-but-scaled unit renders
    with a leading numeric factor whose decimal point must survive -- practical
    salinity is ``"0.001"``, and ``"0 001"`` is not the same number.
    """
    if unit is None:
        return None
    return _SEPARATOR_DOT.sub(" ", str(unit))


def multiply(*units):
    """Units of a product; `None` if any operand is unknown."""
    units = list(units)
    if any(u is None for u in units):
        return None
    if not units:
        return Unit("1")
    # Seed with the first real unit rather than Unit("1") so that an offset
    # operand (degC) is never the right-hand side of `1 * degC`.
    result = units[0]
    for unit in units[1:]:
        result = result * unit
    return result


def divide(numerator, denominator):
    """Units of a quotient; `None` if either operand is unknown."""
    if numerator is None or denominator is None:
        return None
    return numerator / denominator


def reciprocal(unit):
    """Units of ``1 / x``; `None` if unknown.

    This is how `alpha` and `beta` get their units -- see the module docstring.
    """
    if unit is None:
        return None
    return Unit("1") / unit


def scale(unit, factor):
    """Units of an array whose *values* were multiplied by `factor`.

    The quantity is unchanged, so scaling the values by `factor` divides the
    units by it: an array in ``kg s-1`` multiplied by 1000 is an array in
    ``g s-1``. Getting this direction backwards is the easiest mistake in the
    whole pipeline, which is why it is a named function rather than an inline
    ``* factor``.
    """
    if unit is None:
        return None
    return unit / factor


def same_units(specs):
    """Whether every *known* spelling in `specs` denotes the same unit.

    Used to decide whether a sum of transformation terms may keep its summands'
    units. Unknown operands do not make a sum incompatible -- they make its
    result unknown -- so they are ignored here.
    """
    known = [u for u in (parse(s) for s in specs) if u is not None]
    return all(u == known[0] for u in known[1:])
