"""An idealized ocean basin for the xwmt example notebooks.

This builds a small, synthetic North-Atlantic-like basin entirely in memory, so
the quickstart and "bring your own model" notebooks run in seconds with no
download. The fields are simple analytic profiles rather than model output, but
they are chosen so the resulting water mass transformation looks like the real
thing: the subtropics are warm, salty and heated; the subpolar gyre is cold,
fresh and strongly cooled; and that cooling transforms light water into dense
water.

Nothing here is xwmt-specific machinery -- it is just an `xgcm.Grid` and an
xbudget recipe, exactly what you would build for your own model.
"""

import numpy as np
import xarray as xr
import xgcm

# Reference values matching the MOM6 defaults that xwmt assumes.
CP = 3992.0  # specific heat capacity [J/kg/K]
RHO_REF = 1035.0  # reference density [kg/m3]
R_EARTH = 6.371e6  # [m]

_DIMS3 = ("z_l", "yh", "xh")


def _centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def make_idealized_basin(nx=40, ny=40, nz=30):
    """Build an idealized North-Atlantic-like basin.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells in longitude, latitude and depth.

    Returns
    -------
    grid : xgcm.Grid
        A grid whose `_ds` holds the tracer fields and tendency diagnostics.
    recipe : dict
        A hand-written xbudget recipe describing the heat and salt budgets,
        ready to pass to `xwmt.WaterMassTransformations`.
    """
    # -- Geometry: a lon/lat box from 60W-0 and 20N-70N, 4000 m deep. ----------
    lon_i = np.linspace(-60.0, 0.0, nx + 1)
    lat_i = np.linspace(20.0, 70.0, ny + 1)
    # Stretch the vertical so layers are thin near the surface, where the
    # stratification and the surface-forced transformation actually live.
    z_i = 4000.0 * np.linspace(0.0, 1.0, nz + 1) ** 2

    lon, lat, z = _centers(lon_i), _centers(lat_i), _centers(z_i)
    dz = np.diff(z_i)

    # Broadcast helpers: depth varies along axis 0, latitude along axis 1.
    z3 = z[:, np.newaxis, np.newaxis]
    lat3 = lat[np.newaxis, :, np.newaxis]

    ds = xr.Dataset(
        coords={
            "xh": ("xh", lon),
            "yh": ("yh", lat),
            "z_l": ("z_l", z),
            "xq": ("xq", lon_i),
            "yq": ("yq", lat_i),
            "z_i": ("z_i", z_i),
        }
    )

    # Geographic coordinates, used by xwmt for the practical-to-absolute salinity
    # conversion that TEOS-10 needs. (Depth-to-pressure no longer needs them: it
    # uses a constant gravity unless `gravity="gsw"` is requested.) Real model
    # output usually carries these as 2D curvilinear fields, so give them the same
    # shape here even though this grid is a plain lon/lat box.
    ds = ds.assign_coords(
        {
            "lon": (("yh", "xh"), np.broadcast_to(lon[np.newaxis, :], (ny, nx)).copy()),
            "lat": (("yh", "xh"), np.broadcast_to(lat[:, np.newaxis], (ny, nx)).copy()),
        }
    )

    # Cell area on a sphere; the metric xwmt uses to area-integrate.
    area = (
        R_EARTH**2
        * np.cos(np.deg2rad(lat))[:, np.newaxis]
        * np.deg2rad(np.diff(lon_i))[np.newaxis, :]
        * np.deg2rad(np.diff(lat_i))[:, np.newaxis]
    )
    ds["areacello"] = (("yh", "xh"), area)
    ds["thkcello"] = (_DIMS3, np.broadcast_to(dz[:, None, None], (nz, ny, nx)).copy())

    # -- Tracers: warm salty subtropics, cold fresh subpolar, over a thermocline.
    sst = 2.0 + 26.0 * np.cos(np.deg2rad((lat3 - 20.0) * 90.0 / 50.0))
    sss = 34.5 + 2.0 * np.exp(-(((lat3 - 30.0) / 12.0) ** 2))
    thetao = 2.0 + (sst - 2.0) * np.exp(-z3 / 500.0)
    so = 34.8 + (sss - 34.8) * np.exp(-z3 / 800.0)

    ds["thetao"] = (_DIMS3, np.broadcast_to(thetao, (nz, ny, nx)).copy())
    ds["so"] = (_DIMS3, np.broadcast_to(so, (nz, ny, nx)).copy())

    # -- Surface forcing: heating in the subtropics, cooling in the subpolar gyre.
    # Positive `hfds` is heat into the ocean [W/m2].
    hfds = 100.0 * np.cos(np.pi * (lat - 20.0) / 45.0)
    # Net evaporation salinifies the subtropics; net precipitation freshens the
    # subpolar gyre. Positive `wfo` is freshwater into the ocean [kg/m2/s].
    wfo = -3.0e-5 * np.exp(-(((lat - 30.0) / 12.0) ** 2)) + 1.5e-5 * np.exp(
        -(((lat - 58.0) / 10.0) ** 2)
    )
    ds["hfds"] = (("yh", "xh"), np.broadcast_to(hfds[:, None], (ny, nx)).copy())
    ds["wfo"] = (("yh", "xh"), np.broadcast_to(wfo[:, None], (ny, nx)).copy())

    # xwmt expects tendencies integrated over each cell: heat in [W], salt in
    # [kg/s]. Surface fluxes converge entirely in the top layer, so the tendency
    # is zero everywhere below it.
    in_surface_layer = xr.zeros_like(ds.thetao)
    in_surface_layer[{"z_l": 0}] = 1.0

    ds["heat_tendency_surface"] = in_surface_layer * ds.hfds * ds.areacello
    # Freshwater dilutes salt, so a positive `wfo` lowers salinity. The 1e-3
    # converts salinity from g/kg to kg/kg, leaving a salt tendency in kg/s.
    ds["salt_tendency_surface"] = (
        in_surface_layer * -ds.wfo * ds.areacello * ds.so.isel(z_l=0) * 1.0e-3
    )

    # -- Interior mixing: vertical diffusion of heat, as a second process. -----
    # Build this as the convergence of a diffusive heat flux defined on the cell
    # interfaces, with no flux through the surface or the sea floor, rather than
    # as rho*cp*kappa*d2T/dz2 on the cell centres. Both are "the diffusion term",
    # but only this one conserves heat exactly: the interior fluxes telescope
    # away and the boundary fluxes are zero, so the tendency sums to zero over
    # the column instead of quietly creating heat.
    kappa = 1.0e-4  # [m2/s], a Munk-like interior diapycnal diffusivity

    # Diffusive flux at the interior interfaces; zero at the two boundaries.
    dT_dz_i = np.diff(thetao, axis=0) / np.diff(z)[:, np.newaxis, np.newaxis]
    flux_i = np.zeros((nz + 1, ny, nx))
    flux_i[1:-1, :, :] = (
        -RHO_REF * CP * kappa * np.broadcast_to(dT_dz_i, (nz - 1, ny, nx))
    )

    ds["heat_tendency_diffusion"] = (
        _DIMS3,
        -np.diff(flux_i, axis=0) * area[np.newaxis, :, :],
    )

    # -- Label everything. ----------------------------------------------------
    # Real model output carries `units`, and xwmt reads them: it derives the
    # units of the transformation rates it returns from the units of the tendency
    # and of the lambda, rather than assuming. An unlabelled input is not an
    # error -- xwmt omits the `units` attribute rather than guess -- but it does
    # mean the output cannot describe itself, and it warns to say so. A synthetic
    # dataset standing in for a model should behave like one.
    #
    # Salinity is labelled `g kg-1` rather than "psu" because the default
    # `s_var="absolute"` declares these to be absolute salinities. The two are the
    # same unit in any case: PSS-78 is dimensionless with a scale of 1e-3.
    labels = {
        "areacello": ("m2", "cell area"),
        "thkcello": ("m", "cell thickness"),
        "thetao": ("degC", "sea water conservative temperature"),
        "so": ("g kg-1", "sea water absolute salinity"),
        "hfds": ("W m-2", "downward heat flux at the sea surface"),
        "wfo": ("kg m-2 s-1", "downward freshwater flux at the sea surface"),
        "heat_tendency_surface": ("W", "heat tendency due to surface fluxes"),
        "salt_tendency_surface": ("kg s-1", "salt tendency due to surface fluxes"),
        "heat_tendency_diffusion": ("W", "heat tendency due to vertical diffusion"),
    }
    for name, (unit, long_name) in labels.items():
        ds[name].attrs.update({"units": unit, "long_name": long_name})
    ds["lon"].attrs["units"] = "degrees_east"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["z_l"].attrs.update({"units": "m", "long_name": "depth of cell centers"})
    ds["z_i"].attrs.update({"units": "m", "long_name": "depth of cell interfaces"})

    grid = xgcm.Grid(
        ds,
        coords={
            "X": {"center": "xh", "outer": "xq"},
            "Y": {"center": "yh", "outer": "yq"},
            "Z": {"center": "z_l", "outer": "z_i"},
        },
        metrics={("X", "Y"): "areacello"},
        padding={"X": "periodic", "Y": "extend", "Z": "extend"},
        autoparse_metadata=False,
    )

    # The xbudget recipe: which variable is lambda, and which variables hold the
    # tendency processes acting on it. `xbudget` builds this for MOM6, but it is
    # only a dict -- for a small case you can write it out by hand.
    recipe = {
        "mass": {"thickness": "thkcello"},
        "heat": {
            "lambda": "thetao",
            "rhs": {
                "surface_exchange_flux": "heat_tendency_surface",
                "diffusion": "heat_tendency_diffusion",
            },
        },
        "salt": {
            "lambda": "so",
            "rhs": {"surface_exchange_flux": "salt_tendency_surface"},
        },
    }

    return grid, recipe
