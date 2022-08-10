import pytest
import numpy as np
import xarray as xr
import xwmt


def test_functional():
    ds = xr.open_dataset("xwmt_test_data.nc", use_cftime=True)
    ds = ds.rename({"xh": "x", "yh": "y", "geolat": "lat", "geolon": "lon"})
    total = xwmt.swmt(ds).G("sigma0")
    assert np.allclose(total.sum(), -2.1251855e09)
