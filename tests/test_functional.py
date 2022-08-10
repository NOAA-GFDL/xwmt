import pytest
import numpy as np
import xarray as xr
import xwmt

ds = xr.open_dataset("xwmt_test_data.nc", use_cftime=True)
ds = ds.rename({"xh": "x", "yh": "y", "geolat": "lat", "geolon": "lon"})
    
def test_functional():
    total = xwmt.swmt(ds).G("sigma0")
    assert np.allclose(total.sum(), -2.1251855e09)
