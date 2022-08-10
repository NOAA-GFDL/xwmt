import pytest
import numpy as np
import xarray as xr
import xwmt

ds = xr.open_dataset("xwmt_test_data.nc", use_cftime=True)
ds = ds.rename({"xh": "x", "yh": "y", "geolat": "lat", "geolon": "lon"})
    
def test_functional_default_sigma0():
    total = xwmt.swmt(ds).G("sigma0")
    assert np.allclose(total.sum(), -2.1251855e09)
    
def test_functional_default_sigma1():
    total = xwmt.swmt(ds).G("sigma1")
    assert np.allclose(total.sum(), -1.8154245e09)

def test_functional_default_sigma2():
    total = xwmt.swmt(ds).G("sigma2")
    assert np.allclose(total.sum(), -1.5430248e09)
    
def test_functional_default_sigma3():
    total = xwmt.swmt(ds).G("sigma3")
    assert np.allclose(total.sum(), -1.32300659e09)
    
def test_functional_default_sigma4():
    total = xwmt.swmt(ds).G("sigma4")
    assert np.allclose(total.sum(), -1.20211688e09)

def test_functional_default_theta():
    total = xwmt.swmt(ds).G("theta")
    assert np.allclose(total.sum(), -1.65192249e09)

def test_functional_default_salt():
    total = xwmt.swmt(ds).G("salt")
    assert np.allclose(total.sum(), -3.01411116e08)