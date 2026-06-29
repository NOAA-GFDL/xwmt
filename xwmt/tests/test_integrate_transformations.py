import pytest
import numpy as np
import xwmt

## Default parameters except that we group all processes together
kwargs = {'group_processes': True}

# heat
def test_functional_3d_theta(baltic_grid_and_budgets):
    grid, simple_budgets = baltic_grid_and_budgets
    answer_dict = {
        "xgcm": np.array([5.89988332e+08, 1.71073947e+08, 1.60436210e+09, 6.14312533e+08]),
        "xhistogram": np.array([6.11109007e+08, 6.16747397e+08, 1.16849868e+09, 4.80601100e+08])
    }
    for method in ["xgcm", "xhistogram"]:
        wmt = xwmt.WaterMassTransformations(grid, simple_budgets, method=method)
        total_wmt = wmt.integrate_transformations(
            "heat",
            bins = np.linspace(0., 4., 5),
            **kwargs
            )['material_transformation']
        assert np.all(np.isclose(
            total_wmt.values,
            answer_dict[method]
        ))


# salt
def test_functional_3d_salt(baltic_grid_and_budgets):
    grid, simple_budgets = baltic_grid_and_budgets
    answer_dict = {
        "xgcm": np.array([-6.92695360e+07, -2.92969936e+08, -5.65359683e+07,  1.31220262e+08]),
        "xhistogram": np.array([-7.09139741e+07, -2.94351853e+08, -2.40161246e+07,  6.65517384e+07])
    }
    for method in ["xgcm", "xhistogram"]:
        wmt = xwmt.WaterMassTransformations(grid, simple_budgets, method=method)
        total_wmt = wmt.integrate_transformations(
            "salt",
            bins = np.linspace(5., 9., 5),
            **kwargs
            )['material_transformation']
        assert np.all(np.isclose(
            total_wmt.values,
            answer_dict[method]
        ))
# sigma2
def test_functional_3d_sigma2(baltic_grid_and_budgets):
    grid, simple_budgets = baltic_grid_and_budgets
    answer_dict = {
        "xgcm": np.array([-3.89013506e+08,  1.11459836e+08,  3.97737451e+07,  7.12295765e+06]),
        "xhistogram": np.array([-3.76907156e+08,  9.13575231e+07,  4.22416664e+07,  6.72588395e+06])
    }
    for method in ["xgcm", "xhistogram"]:
        wmt = xwmt.WaterMassTransformations(grid, simple_budgets, method=method)
        total_wmt = wmt.integrate_transformations(
            "sigma2",
            bins=np.linspace(15., 19., 5),
            **kwargs
            )['material_transformation']
        assert np.all(np.isclose(
            total_wmt.values,
            answer_dict[method]
        ))
