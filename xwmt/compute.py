import warnings

import gsw
import numpy as np
import xarray as xr
import xgcm


def hlamdot_from_Jlam(grid, Jlam, dim, h=None):
    """
    Calculation of hlamdot (cell-depth integral of scalar tendency)
    from interfacial fluxes.

    Parameters
    ----------
    grid : xgcm.Grid
    Jlam : xr.DataArray
        Interfacial flux of the tracer.
    dim : str
        Axis along which to difference (e.g. "Z").
    h : xr.DataArray, optional
        Cell-center thickness metric. If None, it is looked up from the grid metrics.
    """
    # For convergence, need to reverse the sign
    dJlam = -grid.diff(Jlam, dim)
    if h is not None:
        h = h.where(h != 0.0)
    else:
        h = grid.get_metric(dJlam, "Z")
    lamdot = dJlam / h
    hlamdot = h.fillna(0.0) * lamdot.fillna(0.0)
    return hlamdot


def calc_hlamdot_tendency(grid, datadict, h=None):
    """
    Wrapper functions to determine h times lambda_dot (vertically extensive tendency)

    `h` is the cell-center thickness metric, forwarded to `hlamdot_from_Jlam` for the
    interfacial-flux case.
    """

    if "layer_integrated_tendency" in datadict:
        return datadict["layer_integrated_tendency"]

    elif "interfacial_flux" in datadict:
        return hlamdot_from_Jlam(grid, datadict["interfacial_flux"], "Z", h=h)
