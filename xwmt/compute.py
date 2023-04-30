import warnings

import gsw
import numpy as np
import xarray as xr
import xgcm
from xhistogram.xarray import histogram

def zonal_mean(da, metrics):
    num = (da * metrics["areacello"] * metrics["wet"]).sum(dim=["x"])
    denom = (da / da * metrics["areacello"] * metrics["wet"]).sum(dim=["x"])
    return num / denom


def get_xgcm_grid_vertical(ds, metrics=True, **kwargs):
    """
    Define xgcm grid object based on vertical axes only

    Parameters
    ----------
    ds : xarray.Dataset
        Single dataset with grid information (i.e., lev and lev_outer)
    metrics : boolean, optional

    Returns
    -------
    An object with multiple :class:`xgcm.Axis` objects representing different
    independent axes
    """

    # Copy static grid dataset variables
    ds_g = ds[["lev", "lev_outer"]].copy()

    # Define coordinates
    coords = {"Z": {"center": "lev", "outer": "lev_outer"}}

    if metrics:
        ds_g["dzt"] = xr.DataArray(
            data=ds["lev_outer"].diff("lev_outer").values,
            coords={"lev": ds["lev"]},
            dims=("lev"),
        )
        # Replace all NaNs with zeros
        ds_g["dzt"] = ds_g["dzt"].fillna(0.0)
        kwargs["metrics"] = {("Z",): ["dzt"]}

    xgrid = xgcm.Grid(ds_g, coords=coords, **kwargs)
    return xgrid


# Full version including all available axes and metrics
# TODO: Change to standard naming
# 'zl', 'z_l', 'rho2_l' -> 'lev'
# 'zi', 'z_i', 'rho2_i' -> 'lev_outer' etc.
def get_xgcm_grid(ds, ds_grid, grid="z", metrics=True, **kwargs):
    """
    Define xgcm grid object from non-static and static grid information.

    Parameters
    ----------
    ds : xarray.Dataset
        Contains non-static grid information (e.g., volcello)
    ds_grid : xarray.Dataset
        Contains static grid information (e.g., dxt, dyt)
    grid : str
        Specifies the diagnostic grid ['native','z','rho2']
    metrics : boolean, optional

    Returns
    -------
    An object with multiple :class:`xgcm.Axis` objects representing different
    independent axes
    """

    # Copy static grid dataset
    ds_g = ds_grid.copy()

    # Specify vertical index name in grid (center '_l' and outer '_i')
    vertind = {
        "native": ["zl", "zi"],
        "z": ["z_l", "z_i"],
        "rho2": ["rho2_l", "rho2_i"],
    }

    # Add vertical coordinate
    ds_g[vertind[grid][0]] = ds[vertind[grid][0]]
    ds_g[vertind[grid][1]] = ds[vertind[grid][1]]

    # Define coordinates
    coords = {
        "X": {"center": "xh", "right": "xq"},
        "Y": {"center": "yh", "right": "yq"},
        "Z": {"center": vertind[grid][0], "outer": vertind[grid][1]},
    }

    if metrics:

        # Define a nominal layer thickness
        ds_g["dzt"] = xr.DataArray(
            data=ds[vertind[grid][1]].diff(vertind[grid][1]).values,
            coords={vertind[grid][0]: ds[vertind[grid][0]]},
            dims=(vertind[grid][0]),
        )

        # Replace all NaNs with zeros
        ds_g["dxt"] = ds_g["dxt"].fillna(0.0)
        ds_g["dyt"] = ds_g["dyt"].fillna(0.0)
        ds_g["dzt"] = ds_g["dzt"].fillna(0.0)
        ds_g["areacello"] = ds_g["areacello"].fillna(0.0)

        metrics = {
            ("X",): ["dxt"],
            ("Y",): ["dyt"],
            ("Z",): ["dzt"],
            ("X", "Y"): ["areacello"],
        }

        # The volume of the cell can be different from areacello * dzt
        # We need to add volcello to the metrics
        if "volcello" in ds.keys():
            ds_g["volcello"] = ds["volcello"].fillna(0.0)
            metrics[("X", "Y", "Z")] = ["volcello"]
        else:
            warnings.warn("'volcello' is missing")

        xgrid = xgcm.Grid(ds_g, coords=coords, metrics=metrics, **kwargs)

    else:
        xgrid = xgcm.Grid(ds_g, coords=coords, **kwargs)

    return xgrid


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


def hlamdot_from_Jlam(xgrid, Jlam, dim):
    """
    Calculation of hlamdot (cell-depth integral of scalar tendency)
    provided various forms of input (fluxes, tendencies, intensive, extensive)
    """
    # For convergence, need to reverse the sign
    lamdot = -xgrid.derivative(Jlam, dim)
    hlamdot = lamdot * xgrid.get_metric(lamdot, "Z")
    return hlamdot


def calc_hlamdotmass(xgrid, dd):
    """
    Wrapper functions for boundary flux.
    """
    hlamdotmass = dd["boundary"]["flux"]
    if dd["boundary"][
        "mass"
    ]:  # If boundary flux specified as mass rather than tracer flux
        scalar_i = xgrid.interp(
            dd["scalar"]["array"], "Z", boundary="extend"
        ).chunk({"lev_outer": -1})
        Jlammass = Jlammass_from_Qm_lm_l(
            hlamdotmass, dd["boundary"]["scalar_in_mass"], scalar_i
        )
        hlamdotmass = hlamdot_from_Jlam(xgrid, Jlammass, dim="Z")
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


def calc_hlamdot_tendency(xgrid, dd):
    """
    Wrapper functions to determine h times lambda_dot (vertically extensive tendency)
    """

    if dd["tendency"]["extensive"]:
        hlamdotmass = None

        if dd["tendency"]["boundary"]:
            hlamdotmass = calc_hlamdotmass(xgrid, dd)

        hlamdot = hlamdot_from_Ldot_hlamdotmass(dd["tendency"]["array"], hlamdotmass)

    else:
        hlamdot = hlamdot_from_lamdot_h(
            dd["tendency"]["array"], xgrid.get_metric(dd["tendency"]["array"], "Z")
        )

    return hlamdot

def bin_define(lmin, lmax, delta_l):
    """Specify the range and widths of the lambda bins"""
    return np.arange(lmin - delta_l / 2.0, lmax + delta_l / 2.0, delta_l)


def bin_percentile(l, percentile=[0.05, 0.95], nbins=100):
    """Specify the percentile and number of the bins"""
    l_sample = l.isel(lev=0, time=0).chunk({"y": -1, "x": -1})
    vmin, vmax = l_sample.quantile(percentile, dim=l_sample.dims)
    return np.linspace(vmin, vmax, nbins)


def expand_surface_to_3d(surfaceflux, z):
    """Expand 2D surface array to 3D array with zeros below surface"""
    return surfaceflux.expand_dims({"lev_outer": z}).where(z == z[0], 0)
