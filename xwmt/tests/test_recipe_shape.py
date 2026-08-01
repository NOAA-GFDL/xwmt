"""
Regression tests for two ways the *shape* of the recipe leaked into xwmt.

- xbudget 0.8.0 added a reserved top-level ``constants`` table to every shipped
  recipe. It sits beside the budgets but is not one, and xwmt read every
  top-level key as a tracer -- so a 0.8.0 preset passed straight to
  `WaterMassTransformations` raised `ValueError: recipe must contain element
  ["constants"]["lambda"]`. Only the raw-recipe path was affected;
  `BudgetQuery.aggregate()` never emits the key, which is why the rest of the
  suite kept passing.

- `available_processes()` deduplicated through a `set`, so the order it returned
  varied with the interpreter's hash seed. That order decides the order terms are
  merged, hence the order of the output's variables and -- before the Dataset's
  attributes were cleared -- which single term's `long_name`, `xwmt_term`,
  `provenance` and `xbudget_path` xarray promoted to *global* attributes. Three
  identical runs could describe the same Dataset three different ways.

The recipes here are hand-built rather than loaded from xbudget, so these tests
mean the same thing on xbudget 0.7.0 (which has no ``constants`` table) and on
0.8.0 (which puts one in every preset). They run on the tiny synthetic grids from
`test_bugfixes`, so no dataset download is needed.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import xwmt
from xwmt import attrs as xattrs

from xwmt.tests.test_bugfixes import _heat_tendency_grid, minimal_grid

#: The reserved table as xbudget 0.8.0 actually writes it: named scalars, each a
#: `{value, units}` mapping. Its *body* is what a budget-walker would trip on, so
#: the tests use the real shape rather than an empty dict.
CONSTANTS_BLOCK = {
    "seawater_density": {"value": 1035.0, "units": "kg m-3"},
    "specific_heat_capacity": {"value": 3992.0, "units": "J kg-1 degC-1"},
}


def _multi_process_grid():
    """A heat budget with several process terms, in a deliberate order.

    `processes` is the order `available_processes()` must report: the budget's
    `lhs` term first, then its `rhs` terms as written. The `rhs` order is
    intentionally neither alphabetical nor sorted, so the test cannot pass on a
    `set` by luck.
    """
    grid, budget, edges = _heat_tendency_grid()
    # Label the lambda the way a model would, so the target bin coordinate has
    # inherited metadata worth checking survives the merge.
    grid._ds["temperature"].attrs.update(
        {"units": "degC", "long_name": "potential temperature"}
    )
    rhs = ["diffusion", "advection", "frazil_ice", "boundary_forcing"]
    for name in rhs:
        grid._ds[f"heat_{name}"] = grid._ds["heat_tendency"].copy()
    budget["heat"]["rhs"] = {name: f"heat_{name}" for name in rhs}
    processes = list(budget["heat"]["lhs"]) + rhs
    return grid, budget, edges, processes


def _with_constants(budget):
    """The same budget with the reserved table in front, as 0.8.0 presets have it."""
    return {"constants": CONSTANTS_BLOCK, **budget}


# --- the reserved `constants` table -------------------------------------------


def test_constants_table_is_not_treated_as_a_tracer():
    grid, budget, _, _ = _multi_process_grid()
    wmt = xwmt.WaterMassTransformations(grid, _with_constants(budget))
    assert "constants" not in wmt.tracer_dict
    assert set(wmt.tracer_dict) == {"heat", "salt"}


def test_constants_table_does_not_become_a_budget():
    grid, budget, _, _ = _multi_process_grid()
    wmt = xwmt.WaterMassTransformations(grid, _with_constants(budget))
    # A budget gets a `processes_<name>_dict`; the reserved table must not.
    assert not hasattr(wmt, "processes_constants_dict")
    assert "constants" not in wmt.available_processes()
    assert "constants" not in wmt.available_processes(available=False)


def test_constants_table_is_preserved_on_the_recipe():
    # Skipping it while walking budgets must not mean dropping it: the recipe
    # attribute should still describe what the caller passed in.
    grid, budget, _, _ = _multi_process_grid()
    wmt = xwmt.WaterMassTransformations(grid, _with_constants(budget))
    assert wmt.recipe["constants"] == CONSTANTS_BLOCK


def test_constants_table_does_not_change_results():
    """The reserved table is inert: identical output with and without it."""
    grid_a, budget_a, edges, _ = _multi_process_grid()
    grid_b, budget_b, _, _ = _multi_process_grid()
    kwargs = dict(cp=1.0, rho_ref=1.0, method="xhistogram")
    plain = xwmt.WaterMassTransformations(grid_a, budget_a, **kwargs)
    withc = xwmt.WaterMassTransformations(grid_b, _with_constants(budget_b), **kwargs)
    out_plain = plain.integrate_transformations("heat", bins=edges)
    out_withc = withc.integrate_transformations("heat", bins=edges)
    assert list(out_plain.data_vars) == list(out_withc.data_vars)
    for name in out_plain.data_vars:
        np.testing.assert_array_equal(out_plain[name].values, out_withc[name].values)


def test_budget_without_a_lambda_still_raises():
    # The fix must not turn a genuinely malformed budget into a silent skip.
    grid = minimal_grid()
    bad = {"mass": {"thickness": "dz"}, "heat": {}}
    with pytest.raises(ValueError, match="lambda"):
        xwmt.WaterMassTransformations(grid, bad)


# --- deterministic process order ----------------------------------------------


def test_available_processes_follows_recipe_order():
    grid, budget, _, processes = _multi_process_grid()
    wmt = xwmt.WaterMassTransformations(grid, budget)
    assert wmt.available_processes() == processes


def test_available_processes_returns_a_list_when_not_filtering():
    # Documented return type is a list; it used to be a `set` in this branch,
    # which is exactly how the ordering became hash-dependent.
    grid, budget, _, processes = _multi_process_grid()
    wmt = xwmt.WaterMassTransformations(grid, budget)
    assert wmt.available_processes(available=False) == processes


#: Run in a subprocess so the interpreter's hash seed can actually be varied --
#: PYTHONHASHSEED is read once at startup, so this cannot be done in-process.
_ORDER_PROBE = (
    "from xwmt.tests.test_recipe_shape import _multi_process_grid; "
    "import xwmt; "
    "g, b, _, _ = _multi_process_grid(); "
    "print(','.join(xwmt.WaterMassTransformations(g, b).available_processes()))"
)


def _process_order_under(seed):
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    out = subprocess.run(
        [sys.executable, "-c", _ORDER_PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_process_order_is_independent_of_the_hash_seed():
    # With a `set`, these seeds disagree; the recipe order must survive both.
    orders = {seed: _process_order_under(seed) for seed in (0, 1, 2, 3)}
    assert len(set(orders.values())) == 1, orders
    assert orders[0] == "tendency,diffusion,advection,frazil_ice,boundary_forcing"


# --- global attributes describe the Dataset, not one arbitrary term -----------


@pytest.fixture
def multi_process_output():
    grid, budget, edges, _ = _multi_process_grid()
    wmt = xwmt.WaterMassTransformations(
        grid, budget, cp=1.0, rho_ref=1.0, method="xhistogram"
    )
    return wmt.integrate_transformations("heat", bins=edges, sum_components=False)


def test_global_attrs_are_exactly_the_dataset_level_ones(multi_process_output):
    expected = set(xattrs.dataset_attrs("heat", "xhistogram", True))
    assert set(multi_process_output.attrs) == expected


@pytest.mark.parametrize(
    "leaked",
    [
        "xwmt_term",
        "xwmt_source_variables",
        "xwmt_budget_side",
        "xwmt_lambda_variable",
        "xbudget_path",
        "xbudget_op",
        "provenance",
        "long_name",
        "cell_methods",
        "units",
    ],
)
def test_no_term_specific_attr_reaches_the_global_attrs(multi_process_output, leaked):
    # Each of these describes one process term. On a Dataset holding several,
    # a global copy taken from whichever term merged first is simply wrong.
    assert leaked not in multi_process_output.attrs


def test_terms_still_carry_their_own_attrs(multi_process_output):
    # The counterpart to the test above: clearing the Dataset's attributes must
    # not clear the variables'.
    rates = [v for v in multi_process_output.data_vars if not v.endswith("_bnds")]
    assert rates
    for name in rates:
        assert multi_process_output[name].attrs["xwmt_term"] == name
        assert multi_process_output[name].attrs["units"] == "kg s-1"


def test_merging_terms_preserves_coordinate_attrs(multi_process_output):
    """Guards the tempting one-line fix.

    Clearing the leaked attributes with `xr.merge(..., combine_attrs="drop")`
    also strips every *variable and coordinate* attribute, including the private
    bin edges the CF `bounds` variable is built from -- so the bounds silently
    disappear. Only the Dataset's own attributes may be cleared.
    """
    coord = multi_process_output["temperature_l_target"]
    assert coord.attrs.get("units") == "degC"
    assert coord.attrs.get("bounds") == "temperature_l_target_bnds"
    assert "temperature_l_target_bnds" in multi_process_output


def test_output_is_reproducible_across_hash_seeds(multi_process_output):
    # Variable order is what the merge order becomes visible as.
    assert list(multi_process_output.data_vars) == [
        "tendency",
        "diffusion",
        "advection",
        "frazil_ice",
        "boundary_forcing",
        "temperature_l_target_bnds",
    ]
