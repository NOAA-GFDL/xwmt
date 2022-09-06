import pytest
import numpy as np
import xarray as xr
import xwmt

ds = xr.open_dataset("xwmt_test_data_baltic_3d.nc", use_cftime=True)

## Default settings
# sigma
def test_functional_3d_sigma0_default():
    total = xwmt.wmt(ds).G("sigma0")
    assert np.allclose(
        np.sum([total[process].sum().values for process in total.keys()]),
        2223515.7150945747,
    )


def test_functional_3d_sigma1_default():
    total = xwmt.wmt(ds).G("sigma1")
    assert np.allclose(
        np.sum([total[process].sum().values for process in total.keys()]),
        2365488.868501793,
    )


def test_functional_3d_sigma2_default():
    total = xwmt.wmt(ds).G("sigma2")
    assert np.allclose(
        np.sum([total[process].sum().values for process in total.keys()]),
        2451837.2042288743,
    )


def test_functional_3d_sigma3_default():
    total = xwmt.wmt(ds).G("sigma3")
    assert np.allclose(
        np.sum([total[process].sum().values for process in total.keys()]),
        2495522.8387764315,
    )


def test_functional_3d_sigma4_default():
    total = xwmt.wmt(ds).G("sigma4")
    assert np.allclose(
        np.sum([total[process].sum().values for process in total.keys()]),
        2573444.982510467,
    )


# heat
def test_functional_3d_theta_default():
    total = xwmt.wmt(ds).G("theta")
    assert np.allclose(
        np.sum([total[process].sum().values for process in total.keys()]),
        -342891148.09930146,
    )


# salt
def test_functional_3d_salt_default():
    total = xwmt.wmt(ds).G("salt")
    assert np.allclose(
        np.sum([total[process].sum().values for process in total.keys()]),
        -285494.1529950457,
    )


## Tendencies not grouped
def test_functional_3d_sigma0_grouped():
    total = xwmt.wmt(ds).G("sigma0", group_tend=True, group_process=True)
    assert np.allclose(total["forcing"].sum(), 1713222.37017132)
    assert np.allclose(total["diffusion"].sum(), 510293.34492326)


## xgcm
def test_functional_3d_sigma0_xgcm():
    total = xwmt.wmt(ds).G("sigma0", method="xgcm")
    assert np.allclose(
        np.sum([total[process].sum().values for process in total.keys()]),
        2228638.8884341754,
    )
