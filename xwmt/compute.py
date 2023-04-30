import warnings

import gsw
import numpy as np
import xarray as xr
import xgcm
from xhistogram.xarray import histogram

def zonal_mean(da, metrics):
    num = (da * metrics["areacello"] * metrics["wet"]).sum(dim=["xh"])
    denom = (da / da * metrics["areacello"] * metrics["wet"]).sum(dim=["xh"])
    return num / denom


## Functions to obtain hlamdot
def Jlammass_from_Qm_lm_l(Qm, lm, l):
    """
    Input
    -----
    Qm : xarray.DataArray
        massflux (e.g., wfo)
    lm : xarray.DataArray
        Scalar value of mass flux (e.g., tos, 0)
    l : xarray.DataArray
        Scalar field of ocean value (e.g., thetao, so)
    """
    return Qm * (lm - l)


def hlamdot_from_Jlam(grid, Jlam, dim):
    """
    Calculation of hlamdot (cell-depth integral of scalar tendency)
    provided various forms of input (fluxes, tendencies, intensive, extensive)
    """
    # For convergence, need to reverse the sign
    lamdot = -grid.derivative(Jlam, dim)
    hlamdot = lamdot * grid.get_metric(lamdot, "Z")
    return hlamdot


def calc_hlamdotmass(grid, dd):
    """
    Wrapper functions for boundary flux.
    """
    hlamdotmass = dd["boundary"]["flux"]
    if dd["boundary"][
        "mass"
    ]:  # If boundary flux specified as mass rather than tracer flux
        scalar_i = grid.interp(
            dd["scalar"]["array"], "Z", boundary="extend"
        ).chunk({grid.axes['Z'].coords['outer']: -1})
        Jlammass = Jlammass_from_Qm_lm_l(
            hlamdotmass, dd["boundary"]["scalar_in_mass"], scalar_i
        )
        hlamdotmass = hlamdot_from_Jlam(grid, Jlammass, dim="Z")
    return hlamdotmass


def hlamdot_from_Ldot_hlamdotmass(Ldot, hlamdotmass=None):
    """
    Advective surface flux
    """
    if hlamdotmass is not None:
        return Ldot + hlamdotmass.fillna(0)
    return Ldot


def hlamdot_from_lamdot_h(lamdot, h):
    return h * lamdot


def calc_hlamdot_tendency(grid, dd):
    """
    Wrapper functions to determine h times lambda_dot (vertically extensive tendency)
    """

    if dd["tendency"]["extensive"]:
        hlamdotmass = None

        if dd["tendency"]["boundary"]:
            hlamdotmass = calc_hlamdotmass(grid, dd)

        hlamdot = hlamdot_from_Ldot_hlamdotmass(dd["tendency"]["array"], hlamdotmass)

    else:
        hlamdot = hlamdot_from_lamdot_h(
            dd["tendency"]["array"], grid.get_metric(dd["tendency"]["array"], "Z")
        )

    return hlamdot

def bin_define(lmin, lmax, delta_l):
    """Specify the range and widths of the lambda bins"""
    return np.arange(lmin - delta_l / 2.0, lmax + delta_l / 2.0, delta_l)


def bin_percentile(l, percentile=[0.05, 0.95], nbins=100):
    """Specify the percentile and number of the bins"""
    l_sample = l.isel(z_l=0, time=0).chunk({"yh": -1, "xh": -1})
    vmin, vmax = l_sample.quantile(percentile, dim=l_sample.dims)
    return np.linspace(vmin, vmax, nbins)


def expand_surface_to_3d(surfaceflux, z):
    """Expand 2D surface array to 3D array with zeros below surface"""
    return surfaceflux.expand_dims({"z_i": z}).where(z == z[0], 0)
