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


## Functions to obtain hldot


def Jlmass_from_Qm_lm_l(Qm, lm, l):
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


def hldot_from_Jl(xgrid, Jl, dim):
    """
    Calculation of hldot (cell-depth integral of scalar tendency)
    provided various forms of input (fluxes, tendencies, intensive, extensive)
    """
    # For convergence, need to reverse the sign
    ldot = -xgrid.derivative(Jl, dim)
    hldot = ldot * xgrid.get_metric(ldot, "Z")
    return hldot


def calc_hldotmass(xgrid, dd):
    """
    Wrapper functions for boundary flux.
    """
    hldotmass = dd["boundary"]["flux"]
    if dd["boundary"][
        "mass"
    ]:  # If boundary flux specified as mass rather than tracer flux
        scalar_i = xgrid.interp(
            dd["scalar"]["array"], "Z", boundary="extrapolate"
        ).chunk({"lev_outer": -1})
        Jlmass = Jlmass_from_Qm_lm_l(
            hldotmass, dd["boundary"]["scalar_in_mass"], scalar_i
        )
        hldotmass = hldot_from_Jl(xgrid, Jlmass, dim="Z")
    return hldotmass


def hldot_from_Ldot_hldotmass(Ldot, hldotmass=None):
    """
    Advective surface flux
    """
    if hldotmass is not None:
        return Ldot + hldotmass.fillna(0)
    return Ldot


def hldot_from_ldot_h(ldot, h):
    return h * ldot


def calc_hldot_tendency(xgrid, dd):
    """
    Wrapper functions to determine h times lambda_dot (vertically extensive tendency)
    """

    if dd["tendency"]["extensive"]:
        hldotmass = None

        if dd["tendency"]["boundary"]:
            hldotmass = calc_hldotmass(xgrid, dd)

        hldot = hldot_from_Ldot_hldotmass(dd["tendency"]["array"], hldotmass)

    else:
        hldot = hldot_from_ldot_h(
            dd["tendency"]["array"], xgrid.get_metric(dd["tendency"]["array"], "Z")
        )

    return hldot


### Standalone and scaled-down version of ddterms functions for testing
def get_density(ds, grid, density_str="sigma0"):
    p = xr.apply_ufunc(gsw.p_from_z, -ds["lev"], grid["lat"], 0, 0, dask="parallelized")
    sa = xr.apply_ufunc(
        gsw.SA_from_SP, ds["so"], p, grid["lon"], grid["lat"], dask="parallelized"
    )
    ct = xr.apply_ufunc(gsw.CT_from_t, sa, ds["thetao"], p, dask="parallelized")

    # Calculate thermal expansion coefficient alpha
    alpha = xr.apply_ufunc(gsw.alpha, sa, ct, p, dask="parallelized")

    # Calculate the haline contraction coefficient beta
    beta = xr.apply_ufunc(gsw.beta, sa, ct, p, dask="parallelized")

    # Calculate potentail density (kg/m^3)
    if density_str == "sigma0":
        density = xr.apply_ufunc(gsw.sigma0, sa, ct, dask="parallelized")
    if density_str == "sigma1":
        density = xr.apply_ufunc(gsw.sigma1, sa, ct, dask="parallelized")
    if density_str == "sigma2":
        density = xr.apply_ufunc(gsw.sigma2, sa, ct, dask="parallelized")
    if density_str == "gamma_n":
        # TODO: Function to calculate neutral density (gamma_n) and other neutral variables (gamma)
        density = gamma_n

    return alpha, beta, density.rename(density_str)


def rho_tend(self, term):
    heat_tend = calc_hldot_tendency(xgrid, dd_heat)
    salt_tend = calc_hldot_tendency(xgrid, dd_salt)

    alpha = get_density()[0]
    beta = get_density()[1]

    # Density tendency due to heat flux (kg/s/m^2)
    rho_tend_heat = -(alpha / Cp) * heat_tend

    # Density tendency due to salt/salinity (kg/s/m^2)
    rho_tend_salt = beta * salt_tend

    return rho_tend_heat, rho_tend_salt


def calc_F_transformed(F, l, xgrid, bins):

    F_transformed = xgrid.transform(
        F, "Z", target=bins, target_data=l, method="conservative"
    ) / np.diff(bins)
    return F_transformed


def calc_G(F, l, grid, bins, method="xhistogram"):

    if method == "xhistogram":

        G = histogram(
            l.where(~np.isnan(F)),
            bins=[bins],
            dim=["x", "y", "lev"],
            weights=(F * grid["areacello"]).where(~np.isnan(F)),
        ) / np.diff(bins)

    elif method == "xgcm":

        G = (
            calc_F_transformed(F, l.copy().rename(l.name + "_bin"), xgrid, bins)
            * grid["areacello"]
        ).sum(["x", "y"])

    return G


def lbin_define(lmin, lmax, delta_l):
    """Specify the range and widths of the lambda bins"""
    return np.arange(lmin - delta_l / 2.0, lmax + delta_l / 2.0, delta_l)


def lbin_percentile(l, percentile=[0.05, 0.95], nbins=100):
    """Specify the percentile and number of the bins"""
    l_sample = l.isel(lev=0, time=0).chunk({"y": -1, "x": -1})
    vmin, vmax = l_sample.quantile(percentile, dim=l_sample.dims)
    return np.linspace(vmin, vmax, nbins)


def expand_surface_to_3D(surfaceflux, z):
    """Expand 2D surface array to 3D array with zeros below surface"""
    return surfaceflux.expand_dims({"lev_outer": z}).where(z == z[0], 0)
