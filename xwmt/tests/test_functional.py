import pytest
import numpy as np
import xarray as xr
import xwmt

ds = xr.open_dataset("xwmt_test_data.nc", use_cftime=True)
ds = ds.rename({"xh": "x", "yh": "y", "geolat": "lat", "geolon": "lon"})

## Default settings
# sigma
def test_functional_sigma0_default():
    total = xwmt.swmt(ds).G("sigma0")
    assert np.allclose(total.sum(), -2.1251855e09, rtol=1e-4)
    
def test_functional_sigma1_default():
    total = xwmt.swmt(ds).G("sigma1")
    assert np.allclose(total.sum(), -1.8154245e09, rtol=1e-4)

def test_functional_sigma2_default():
    total = xwmt.swmt(ds).G("sigma2")
    assert np.allclose(total.sum(), -1.5430248e09, rtol=1e-4)
    
def test_functional_sigma3_default():
    total = xwmt.swmt(ds).G("sigma3")
    assert np.allclose(total.sum(), -1.32300659e09, rtol=1e-4)
    
def test_functional_sigma4_default():
    total = xwmt.swmt(ds).G("sigma4")
    assert np.allclose(total.sum(), -1.20211688e09, rtol=1e-4)

# heat
def test_functional_theta_default():
    total = xwmt.swmt(ds).G("theta")
    assert np.allclose(total.sum(), -1.65192249e09, rtol=1e-4)
# salt
def test_functional_salt_default():
    total = xwmt.swmt(ds).G("salt")
    assert np.allclose(total.sum(), -3.01411116e08, rtol=1e-4)

## Tendencies not grouped
def test_functional_sigma0_notgrouped():
    total = xwmt.swmt(ds).G("sigma0",group_tend=False)
    assert np.allclose(total["heat"].sum(), -2.8727879e09, rtol=1e-4)
    assert np.allclose(total["salt"].sum(), 7.47602401e08, rtol=1e-4)
    
## xgcm
def test_functional_sigma0_xgcm():
    total = xwmt.swmt(ds).G("sigma0",method="xgcm")
    assert np.allclose(total.sum(), -2.1251855e09, rtol=1e-4)