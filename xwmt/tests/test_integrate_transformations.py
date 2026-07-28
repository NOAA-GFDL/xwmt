import pytest
import numpy as np
import xwmt

## Default parameters except that we group all processes together
kwargs = {"group_processes": True}


# heat
def test_functional_3d_theta(baltic_grid_and_budgets):
    grid, simple_budgets = baltic_grid_and_budgets
    answer_dict = {
        "xgcm": np.array([5.89988332e08, 1.71073947e08, 1.60436210e09, 6.14312533e08]),
        "xhistogram": np.array(
            [6.11109007e08, 6.16747397e08, 1.16849868e09, 4.80601100e08]
        ),
    }
    for method in ["xgcm", "xhistogram"]:
        wmt = xwmt.WaterMassTransformations(grid, simple_budgets, method=method)
        total_wmt = wmt.integrate_transformations(
            "heat", bins=np.linspace(0.0, 4.0, 5), **kwargs
        )["material_transformation"]
        assert np.all(np.isclose(total_wmt.values, answer_dict[method]))


# salt
def test_functional_3d_salt(baltic_grid_and_budgets):
    grid, simple_budgets = baltic_grid_and_budgets
    answer_dict = {
        "xgcm": np.array(
            [-6.92695360e07, -2.92969936e08, -5.65359683e07, 1.31220262e08]
        ),
        "xhistogram": np.array(
            [-7.09139741e07, -2.94351853e08, -2.40161246e07, 6.65517384e07]
        ),
    }
    for method in ["xgcm", "xhistogram"]:
        wmt = xwmt.WaterMassTransformations(grid, simple_budgets, method=method)
        total_wmt = wmt.integrate_transformations(
            "salt", bins=np.linspace(5.0, 9.0, 5), **kwargs
        )["material_transformation"]
        assert np.all(np.isclose(total_wmt.values, answer_dict[method]))


# sigma2
def test_functional_3d_sigma2(baltic_grid_and_budgets):
    """Density transformation under the default constant `gravity=9.81`.

    These values differ from the `gravity="gsw"` ones below by <5e-4 relative --
    the whole of that difference is the depth-to-pressure conversion, and
    `test_functional_3d_sigma2_gsw_gravity` pins the previous behaviour exactly.
    """
    grid, simple_budgets = baltic_grid_and_budgets
    answer_dict = {
        "xgcm": np.array([-3.89187983e08, 1.11433652e08, 3.97748322e07, 7.12432497e06]),
        "xhistogram": np.array(
            [-3.77089597e08, 9.13415550e07, 4.22420504e07, 6.72720441e06]
        ),
    }
    for method in ["xgcm", "xhistogram"]:
        wmt = xwmt.WaterMassTransformations(grid, simple_budgets, method=method)
        total_wmt = wmt.integrate_transformations(
            "sigma2", bins=np.linspace(15.0, 19.0, 5), **kwargs
        )["material_transformation"]
        assert np.all(np.isclose(total_wmt.values, answer_dict[method]))


# sigma2, with the latitude-dependent gravity that used to be the only option
def test_functional_3d_sigma2_gsw_gravity(baltic_grid_and_budgets):
    """`gravity="gsw"` must still reproduce the pre-`gravity` golden values.

    Density is the only quantity the depth-to-pressure conversion touches, so
    this is where a regression in it would show up.
    """
    grid, simple_budgets = baltic_grid_and_budgets
    answer_dict = {
        "xgcm": np.array([-3.89013506e08, 1.11459836e08, 3.97737451e07, 7.12295765e06]),
        "xhistogram": np.array(
            [-3.76907156e08, 9.13575231e07, 4.22416664e07, 6.72588395e06]
        ),
    }
    for method in ["xgcm", "xhistogram"]:
        wmt = xwmt.WaterMassTransformations(
            grid, simple_budgets, method=method, gravity="gsw"
        )
        total_wmt = wmt.integrate_transformations(
            "sigma2", bins=np.linspace(15.0, 19.0, 5), **kwargs
        )["material_transformation"]
        assert np.all(np.isclose(total_wmt.values, answer_dict[method]))
