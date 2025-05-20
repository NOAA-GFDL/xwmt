import pytest
import numpy as np
import xarray as xr
import xbudget
import xgcm
import xwmt

ds = xr.open_dataset("xwmb_test_data_Baltic_3d.20230830.nc", use_cftime=True).isel(time=0)
coords = {
    'X': {'center': 'xh', 'outer': 'xq'},
    'Y': {'center': 'yh', 'outer': 'yq'},
    'Z': {'center': 'zl', 'outer': 'zi'},
}
metrics = {
    ('X','Y'): "areacello",
}
grid = xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=None, autoparse_metadata=False)

budgets_dict = xbudget.load_preset_budget(model="MOM6")
xbudget.collect_budgets(grid, budgets_dict)
simple_budgets = xbudget.aggregate(budgets_dict)
wmt = xwmt.WaterMassTransformations(grid, simple_budgets, method="xgcm")

## Default parameters except: wide bin range to cover all cases and group processes
kwargs = {'bins': np.arange(-10, 100, 1.), 'group_processes': True}

# heat
def test_functional_3d_theta_default():
    total_wmt = wmt.integrate_transformations("heat", **kwargs)['material_transformation']
    assert np.isclose(
        total_wmt.sum().values,
        7156040943.980093,
    )


# salt
def test_functional_3d_salt_default():
    total_wmt = wmt.integrate_transformations("salt", **kwargs)['material_transformation']
    assert np.isclose(
        total_wmt.sum().values,
        -78005966.15053421,
    )

# sigma
def test_functional_3d_sigma0_default():
    total_wmt = wmt.integrate_transformations("sigma0", **kwargs)['material_transformation']
    assert np.isclose(
        total_wmt.sum().values,
        -550052085.4891472,
    )


def test_functional_3d_sigma1_default():
    total_wmt = wmt.integrate_transformations("sigma1", **kwargs)['material_transformation']
    assert np.isclose(
        total_wmt.sum().values,
        -550052085.4891474,
    )


def test_functional_3d_sigma2_default():
    total_wmt = wmt.integrate_transformations("sigma2", **kwargs)['material_transformation']
    assert np.isclose(
        total_wmt.sum().values,
        -550065820.7692934,
    )


def test_functional_3d_sigma3_default():
    total_wmt = wmt.integrate_transformations("sigma3", **kwargs)['material_transformation']
    assert np.isclose(
        total_wmt.sum().values,
        -550052085.4891478,
    )


def test_functional_3d_sigma4_default():
    total_wmt = wmt.integrate_transformations("sigma4", **kwargs)['material_transformation']
    assert np.isclose(
        total_wmt.sum().values,
        -550052085.489147,
    )
