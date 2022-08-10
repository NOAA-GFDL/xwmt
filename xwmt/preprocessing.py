import numpy as np
import xarray as xr
import pandas as pd


def rechunk_dataset(ds):
    for varname in list(ds.data_vars):
        chunks = {}
        for (dim, size) in zip(ds[varname].dims, ds[varname].shape):
            if dim in ["time", "time_bounds"]:
                chunks[dim] = 1
            else:
                chunks[dim] = size
        ds[varname] = ds[varname].chunk(chunks=chunks)

    return ds


def get_grid_info(grid, verbose=True):
    """
    Obtain all the relevant grid variables from a dataset.
    """
    vprint = print if verbose else lambda *a, **k: None
    grd = xr.Dataset()

    # Copy global attributes
    grd.attrs = grid.attrs

    # Look for all the possible names of relevant fields (areacello, deptho, wet)
    for key in ["areacello", "area_t", "area", "rA"] + ["ht", "deptho", "Depth", "wet"]:
        if key in grid:
            grd[key] = grid[key]
    dims = list(grd.dims)
    vprint("Check grid dimensions:", dims)

    # Look for all the possible names of coordinates (lon, lat)
    for key in ["lat", "geolat", "geolat_t", "YC"] + [
        "lon",
        "geolon",
        "geolon_t",
        "XC",
    ]:
        if key in grid and sorted(dims) == sorted(grid[key].dims):
            grd[key] = grid[key]
    return grd


def preprocessing(
    ds,
    grid,
    decode_times=True,
    shift_coords=False,
    reset_time_axis=False,
    rechunk=True,
    verbose=True,
):
    """
    Preprocessing routine for watermass transformation (WMT).

    Parameters
    ----------
    ds : xarray.Dataset
        Includes all the relevant variables for surface and/or 3d WMT
    grid : xarray.Dataset
        Includes all the relevant grid variables (e.g., cell area) and coordinates
    decode_times : boolean, optional
        Option to decode time axis according to CF conventions and convert to np.datetime64[ns] object. True by default.
    shift_coords : boolean, optional
        Option to shift lon and lat to always be -180 to 180 and -90 to 90, respectively. Flase by default.
    reset_time_axis : boolean, optional
        Option to reset time axis. New time coordinate will be created with pandas.date_range
        and given frequency (e.g., 'MS', 'YS'). False by default.
    rechunk : boolean, optional
        Whether to reset the chunk size to (1, nx, ny). True by default.

    Returns
    -------
    xarray.Dataset
        Merged Dataset containing all variables with common names and dimensions.
    """

    vprint = print if verbose else lambda *a, **k: None

    # Get grid information
    grid = get_grid_info(grid.squeeze(drop=True), verbose=verbose)

    # Combine grid info with relevant variables into a single dataset
    for key in ds.data_vars.keys():
        grid[key] = ds[key].squeeze(drop=True)

    # Add and rename coordinates
    coords = {"time_bnds": "time_bounds"}
    for (key, newkey) in coords.items():
        if newkey in list(ds.coords):
            grid[newkey] = ds[newkey]
        elif key in list(ds.coords):
            grid[newkey] = ds[key]
    ds = grid

    # Rename coordinates and grid variables to standard names
    dims = {
        "area_t": "areacello",
        "area": "areacello",
        "rA": "areacello",
        "geolat": "lat",
        "geolat_t": "lat",
        "YC": "lat",
        "latitude": "lat",
        "geolon": "lon",
        "geolon_t": "lon",
        "XC": "lon",
        "longitude": "lon",
        "ht": "deptho",
        "Depth": "deptho",
    }

    # Rename dimesnions to 'x' and 'y'
    coords = {
        "xh": "x",
        "xt_ocean": "x",
        "X": "x",
        "i": "x",
        "yh": "y",
        "yt_ocean": "y",
        "Y": "y",
        "j": "y",
        "z_i": "lev_outer",
        "Zp1": "lev_outer",
        "Zl": "lev_upper",
        "z_l": "lev",
        "Z": "lev",
    }
    newdims = {}
    for (key, newkey) in {**dims, **coords}.items():
        if key in list(ds.coords):
            newdims[key] = newkey
    vprint("Rename dimensions:", newdims)
    ds = ds.rename(newdims)

    # Rename variables to CMOR names
    datavars = {
        "surface_temp": "tos",
        "surface_salt": "sos",
        "net_sfc_heating": "hfds",
        "TFLUX": "hfds",
        "SFLUX": "sfdsi",
        "pme_river": "wfo",
        "oceFWflx": "wfo",
        "temp": "thetao",
        "THETA": "thetao",
        "salt": "so",
        "SALT": "so",
        "dif_ConvH": "opottempdiff",
        "forcH": "boundary_forcing_heat_tendency",
        "geoflx": "internal_heat_heat_tendency",
        "dif_ConvS": "osaltdiff",
        "forcS": "boundary_forcing_salt_tendency",
        "tendH": "opottemptend",
        "adv_ConvH": "T_advection_xy",
        "tendS": "osalttend",
        "adv_ConvS": "S_advection_xy",
    }

    newnames = {}
    for (key, newkey) in {**dims, **datavars}.items():
        # Rename when the new variable name does not exist
        if key in list(ds.data_vars) and not newkey in list(ds.data_vars):
            # vprint (key,newkey)
            newnames[key] = newkey
    vprint("Rename data variables", newnames)
    ds = ds.rename(newnames)

    # Convert units if different from standard
    for key in ["thetao", "tos"]:
        if key in ds:
            vprint("Temperature attributes:\n", key, ds[key].attrs)
            if "units" in ds[key].attrs and ds[key].attrs["units"] in [
                "degK",
                "degree K",
                "degrees K",
            ]:
                vprint("Convert from K to C:", key, ds[key].attrs["units"])
                ds[key] = xr.where(ds[key] > 100, ds[key] - 273.15, ds[key])

    if "sfdsi" in ds:
        vprint("Salt flux attributes:\n", "sfdsi", ds["sfdsi"].attrs)
        if "units" in ds["sfdsi"].attrs and ds["sfdsi"].attrs["units"] in [
            "g/m^2/s",
            "g m-2 s-1",
        ]:
            vprint("Convert from g to kg:", "sfdsi", ds["sfdsi"].attrs["units"])
            ds["sfdsi"] = ds["sfdsi"] / 1000.0
            ds["sfdsi"] = ds["sfdsi"].assign_attrs(units="kg m-2 s-1")

    # Create land mask wet from deptho (if it is missing)
    if "wet" in ds:
        vprint("wet exists")
    elif "deptho" in ds:
        vprint("wet is missing: Calculate wet using deptho")
        ds["wet"] = xr.where(~np.isnan(ds.deptho), 1, 0)
    else:
        vprint("[ERROR] Both wet and deptho is missing")

    # Create a all-zero field for sfdsi (when it is missing)
    if not "sfdsi" in ds and "hfds" in ds:
        vprint("sfdsi is missing: Add all-zero field for sfdsi based on hfds")
        ds["sfdsi"] = xr.zeros_like(ds["hfds"]).rename("sfdsi")
        # Remove all attributes
        ds["sfdsi"].attrs = {}
    elif not "sfdsi" in ds:
        vprint("[ERROR] Both sdfsi and hfds is missing")
        ds["sfdsi"] = xr.DataArray(
            data=np.zeros([ds.time.shape[0], ds.y.shape[0], ds.x.shape[0]]),
            coords=[ds.time, ds.y, ds.x],
        ).rename("sfdsi")

    # Decode time axis
    if decode_times:
        vprint("Decode time axis")
        ds["time"] = xr.decode_cf(ds["time"].to_dataset(name="dt")).dt
        if ds.time.dtype == "object" and not (
            "calendar_type" in ds.time.attrs
            and ds.time.attrs["calendar_type"].lower() == "noleap"
        ):
            ds["time"] = ds.indexes["time"].to_datetimeindex()

    if reset_time_axis:
        vprint("Recreate time axis")
        ds["time"] = pd.date_range(
            end=ds["time"].values[-1],
            periods=len(ds["time"]),
            freq="MS",
            normalize=True,
        )

    # Unify lon and lat to -180 to 180 and -90 to 90
    if shift_coords:
        if not all(ds.x.diff("x") > 0):
            vprint("Reindex x", m)
            ds = ds.reindex(x=ds.x.sortby("x"))
        if not all(ds.y.diff("y") > 0):
            vprint("Reindex y", m)
            ds = ds.reindex(y=ds.x.sortby("y"))

    # Rechunk merged dataset
    if rechunk:
        ds = rechunk_dataset(ds)

    return ds
