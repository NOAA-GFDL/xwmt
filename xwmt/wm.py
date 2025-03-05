import numpy as np
import xarray as xr
import xgcm
import gsw
import warnings

class WaterMass:
    """
    Stores water mass characteristics and supports methods for analyzing water masses on a numerical grid.
    """
    def __init__(
        self,
        grid,
        t_name="thetao",
        s_name="so",
        h_name="thkcello",
        teos10=True,
        cp=3992.0,
        rho_ref=1035.0,
        t_var="conservative",
        s_var="absolute",
        ):
        """
        Create a new WaterMass object from an input xgcm.Grid instance.

        Parameters
        ----------
        grid: xgcm.Grid
            Contains information about ocean model grid coordinates, metrics, and data variables.
        t_name: str (default: "thetao")
            Name of conservative temperature variable [in degrees Celsius] in ds.
        s_name: str (default: "so")
            Name of absolute salinity variable [in g/kg] in ds.
        h_name: str (default: "thkcello")
            Name of thickness variable [in m] in ds.
        teos10 : boolean (default: True)
            Get expansion/contraction coefficients from the Thermodynamic Equation Of Seawater - 2010 (TEOS-10),
            unless "alpha" and "beta" variables already present in `grid._ds`.
        cp: float (default: 3992.0)
            Value of specific heat capacity.
        rho_ref: float (default: 1035.0)
            Value of reference potential density, assuming Boussinesq approximation.
        t_var: str ("absolute" or "practical")
            Does variable `t_name` represent "conservative" and "potential" temperature?
        s_var: str ("absolute" or "practical")
            Does variable `s_name` represent "absolute" and "practical" temperature?
        """
        # Grid copy
        self.grid = xgcm.Grid(
            grid._ds.copy(),
            coords={
                **{ax:grid.axes[ax].coords for ax in grid.axes.keys()},
            },
            metrics={k:vv.name for (k,v) in grid._metrics.items() for vv in v},
            boundary={
                **{ax:grid.axes[ax]._boundary for ax in grid.axes.keys()},
            },
            autoparse_metadata=False
        )
        self.t_name = t_name
        self.t_var = t_var
        self.s_name = s_name
        self.s_var = s_var
        self.h_name = h_name
        self.teos10 = teos10
        self.cp = cp
        self.rho_ref = rho_ref
        
        if "Z_metrics" in vars(self.grid):
            pass
        elif self.h_name in self.grid._ds:
            # Conservatively interpolate thickness to cell interfaces, needed to
            # estimate depth of layer centers and compute surface flux divergences
            Z_center_extended = np.concatenate((
                self.grid._ds[self.grid.axes['Z'].coords['outer']][np.array([0])].values,
                self.grid._ds[self.grid.axes['Z'].coords['center']].values,
                self.grid._ds[self.grid.axes['Z'].coords['outer']][np.array([-1])].values
            ))
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                if "time" in self.grid._ds:
                    time_attrs = self.grid._ds.time.attrs.copy()
                self.grid._ds[f'{h_name}_i'] = self.grid.transform(
                    self.grid._ds[h_name].fillna(0.),
                    "Z",
                    Z_center_extended,
                    method="conservative",
                ).assign_coords({
                    self.grid.axes['Z'].coords['outer']:
                    self.grid._ds[self.grid.axes['Z'].coords['outer']].values
                })
            setattr(self.grid, "Z_metrics", {
                "center": self.grid._ds[self.h_name],
                "outer": self.grid._ds[f'{self.h_name}_i']
            })
            if "time" in self.grid._ds:
                self.grid._ds.time.attrs = time_attrs # For some reason these are not preserved by default
        elif "Z" not in self.grid.axes:
            self.grid._ds["z_l"] = xr.DataArray([0.5], dims=("z_l",))
            self.grid._ds["z_i"] = xr.DataArray([0, 1.], dims=("z_i",))
            self.grid._ds[f"{self.h_name}"] = xr.DataArray([1], dims=("z_l",))
            self.grid._ds[f"{self.h_name}_i"] = xr.DataArray([0.5, 0.5], dims=("z_i",))
            coords_2d = {ax:self.grid.axes[ax].coords for ax in self.grid.axes.keys()}
            coords_2d["Z"] = {"center": "z_l", "outer": "z_i"}
            metrics_2d = {k:vv.name for (k,v) in self.grid._metrics.items() for vv in v}
            boundary_2d = {ax:self.grid.axes[ax]._boundary for ax in self.grid.axes.keys()}
            boundary_2d["Z"] = "extend"
            self.grid = xgcm.Grid(
                self.grid._ds,
                coords=coords_2d,
                metrics=metrics_2d,
                boundary=boundary_2d,
                autoparse_metadata=False
            )
            setattr(self.grid, "Z_metrics", {
                "center": self.grid._ds[f"{self.h_name}"],
                "outer": self.grid._ds[f"{self.h_name}_i"]
            })
        self.grid._ds['z'] = (
            -self.grid.cumsum(self.grid.Z_metrics["outer"], "Z")
        ).chunk({self.grid.axes["Z"].coords["center"]: -1})
        self.grid._ds['z_interface'] = xr.where(
            self.grid.axes["Z"].coords["outer"] != self.grid.axes["Z"].coords["outer"][0],
            -self.grid.cumsum(self.grid.Z_metrics["center"], "Z", to="outer"),
            0.
        ).chunk({self.grid.axes["Z"].coords["outer"]: -1})
        
    def get_density(self, density_name="rho"):
        """
        Derive density variables from layer temperature, salinity, and thickness,
        and add them to the dataset (if not already present).
        Uses the TEOS10 algorithm from the `gsw` package by default, unless "alpha"
        and "beta" variables are already provided in `self.grid._ds`.

        Parameters
        ----------
        density_name: str (default: "rho")
            Name of density variable. Supported density variables are: 
            "rho" (in-situ), "sigma0", "sigma1", "sigma2", "sigma3", "sigma4"
            (corresponding to functions of the same name in the `gsw` package).

        Returns
        -------
        xr.DataArray
        """
        
        if self.t_name not in self.grid._ds:
            raise ValueError(f"ds must include temperature variable\
            defined by kwarg t_name (default: {self.t_name}).")
        if self.s_name not in self.grid._ds:
            raise ValueError(f"ds must include salinity variable\
            defined by kwarg s_name (default: {self.s_name}).")
        if self.h_name not in self.grid._ds:
            raise ValueError(f"ds must include thickness variable\
            defined by kwarg h_name (default: {self.h_name}).")
        
        if (
            "alpha" not in self.grid._ds or "beta" not in self.grid._ds or self.teos10
        ) and "p" not in vars(self):
            self.grid._ds['p'] = xr.apply_ufunc(
                gsw.p_from_z, self.grid._ds.z, self.grid._ds.lat, 0, 0, dask="parallelized"
            )

        if "sigma" in density_name:
            z_ref = density_name.replace("sigma", "")
            try:
                z_ref = -float(z_ref)*1000
            except:
                print("'density_name' is not of form 'sigmaX' where 'X' is a number.")
            
            p_ref = xr.apply_ufunc(
                gsw.p_from_z, z_ref, self.grid._ds.lat, 0, 0, dask="parallelized"
            )
        elif density_name == "rho":
            z_ref = self.grid._ds.z
            p_ref = self.grid._ds.p
        else:
            raise NameError(f"`density_name = {density_name}` is not a supported option.")
        
        # Prognostic temperature and salinity in MOM6 should be interpreted
        # as conservative temperature and absolute salinity (following McDougall
        # et al. 2021).
        if self.teos10 and "sa" not in self.grid._ds:
            if self.s_var == "absolute":
                self.grid._ds['sa'] = self.grid._ds[self.s_name]
            elif self.s_var == "practical":
                self.grid._ds['sa'] = xr.apply_ufunc(
                    gsw.SA_from_SP,
                    self.grid._ds[self.s_name],
                    self.grid._ds.p,
                    self.grid._ds.lon,
                    self.grid._ds.lat,
                    dask="parallelized",
                )
        if self.teos10 and "ct" not in self.grid._ds:
            if self.t_var == "conservative":
                self.grid._ds['ct'] = self.grid._ds[self.t_name]
            elif self.t_var == "potential":
                self.grid._ds['ct'] = xr.apply_ufunc(
                    gsw.CT_from_t,
                    self.grid._ds.sa,
                    self.grid._ds[self.t_name],
                    self.grid._ds.p,
                    dask="parallelized"
                )
        if not self.teos10 and ("sa" not in vars(self) or "ct" not in vars(self)):
            self.grid._ds['sa'] = self.grid._ds[self.s_name]
            self.grid._ds['ct'] = self.grid._ds[self.t_name]

        # Calculate thermal expansion coefficient alpha (1/K) at reference pressure
        if "alpha" not in self.grid._ds:
            self.grid._ds['alpha'] = xr.apply_ufunc(
                gsw.alpha, self.grid._ds.sa, self.grid._ds.ct, p_ref, dask="parallelized"
            )

        # Calculate the haline contraction coefficient beta (kg/g) at reference pressure
        if "beta" not in self.grid._ds:
            self.grid._ds['beta'] = xr.apply_ufunc(
                gsw.beta, self.grid._ds.sa, self.grid._ds.ct, p_ref, dask="parallelized"
            )

        # Calculate potential density (kg/m^3)
        if density_name not in self.grid._ds:
            if density_name == "rho":
                self.grid._ds[density_name] = xr.apply_ufunc(
                    getattr(gsw, density_name),
                    self.grid._ds.sa,
                    self.grid._ds.ct,
                    self.grid._ds.p,
                    dask="parallelized"
                ).rename(density_name)
            
            elif "sigma" in density_name:
                self.grid._ds[density_name] = xr.apply_ufunc(
                    getattr(gsw, density_name),
                    self.grid._ds.sa,
                    self.grid._ds.ct,
                    dask="parallelized"
                ).rename(density_name)

        return self.grid._ds[density_name]

    def get_outcrop_lev(self, position="center", incrop=False):
        """
        Find the vertical coordinate level that outcrops at the sea surface, broadcast
        across all other dimensions of the thickness variable (`self.grid._ds.h_name`).

        Parameters
        ----------
        position: str
            Position of the desired vertical coordinate in the `self.grid` instance of `xgcm.Grid`.
            Default: "center". Other supported option is "outer".
        incrop: bool
            Default: False. If True, returns the seafloor incrop level instead. 
        """
        z_coord = self.grid.axes['Z'].coords[position]
        dk = int(2*incrop - 1)
        h = self.grid.Z_metrics[position]
        cumh = h.sel(
                {z_coord: self.grid._ds[z_coord][::dk]}
            ).cumsum(z_coord)
        return cumh.idxmax(z_coord).where(cumh.isel({z_coord:-1})!=0.)
        
    def sel_outcrop_lev(self, da, incrop=False, position="center", **kwargs):
        """
        Select `da` at the vertical coordinate level that outcrops at the sea surface. Assumes
        `da` has the same dimensions as `self.grid.Z_metrics[position].sel(**kwargs)`, and broadcasts in all
        other dimensions than the vertical.

        Parameters
        ----------
        da : xr.DataArray
        incrop : bool
            Default: False. If True, returns the seafloor incrop level instead.
        position : str
            Position of the desired vertical coordinate in the `self.grid` instance of `xgcm.Grid`.
            Default: "center". Other supported option is "outer".
        **kwargs : **dict
            Passed to the `xr.DataArray.sel` method on the vertical thickness.

        Returns
        -------
        xr.DataArray
        """
        z_coord = self.grid.axes['Z'].coords[position]
        dk = int(2*incrop - 1)
        h = self.grid.Z_metrics[position].sel(**kwargs)
        if da.dims != h.dims:
            raise ValueError(
                "`da` must have the same dimensions as\
                `self.grid.Z_metrics[position].sel(**kwargs)`"
            )
        cumh = h.sel(
                {z_coord: self.grid._ds[z_coord][::dk]}
            ).cumsum(z_coord)
        return da.sel(
            {z_coord: cumh.idxmax(z_coord)}
        ).where(cumh.isel({z_coord:-1})!=0.)
    
    def expand_surface_array_vertically(self, da_surf, target_position="outer"):
        """
        Expand surface xr.DataArray (with no "Z"-dimension coordinate) in the vertical,
        filling with zeros in all layers except the one that outcrops.
        
        Parameters
        ----------
        da_surf: xarray.DataArray
            Variable that is to be expanded in the vertical.
        position : str
            Position of the desired vertical coordinate in the `self.grid` instance of `xgcm.Grid`.
            Default: "outer". Other supported option is "center".
        """
        z_coord = self.grid.axes['Z'].coords[target_position]
        return (
            da_surf.expand_dims({z_coord: self.grid._ds[z_coord]})
            .where(
                self.grid._ds[z_coord] ==
                self.get_outcrop_lev(position=target_position),
                0.
            )
        )

    def infer_bins(self, da, percentiles=[0., 1.], nbins=100, surface=False):
        """
        Specify bins based on the distribution of `da`, excluding outliers.
        
        Parameters
        ----------
        da: xarray.DataArray
            Variable used to determine bins.
        percentiles: list
            List of length 2 containing the upper and lower percentiles to bound the array of bins.
            Default: [0., 1.], i.e. min and max.
        nbins: int
            Number of bins. Default: 100.
        surface: bool
            Default: False. If True, compute percentiles only from the outcropping layer of `da`.
        """
        if surface:
            da=self.sel_outcrop_lev(da)
        if percentiles != [0., 1.]:
            vmin, vmax = da.quantile(percentiles, dim=da.dims)
        else:
            vmin, vmax = da.min(), da.max()
        return np.linspace(vmin, vmax, nbins)

    def zonal_mean(self, da, oceanmask_name="wet"):
        """
        Compute area-weighted zonal mean (along `X` grid axis).
        
        Parameters
        ----------
        da: xarray.DataArray
            Data array to be averaged.
        oceanmask_name: str
            Name of ocean mask xr.DataArray in `self.grid._ds`. Default: "wet".
        """
        x_name = grid.axes['X'].coords['center']
        area = self.grid.get_metric(da, ['X', 'Y'])
        num = (da * area * self.grid._ds[landmask_name]).sum(dim=x_name)
        denom = (area * self.grid._ds[landmask_name]).sum(dim=x_name)
        return num / denom

def add_gridcoords(grid, coords, boundary):
    new_grid = xgcm.Grid(
        grid._ds,
        coords={
            **{ax:grid.axes[ax].coords for ax in grid.axes.keys()},
            **coords
        },
        metrics={k:vv.name for (k,v) in grid._metrics.items() for vv in v},
        boundary={
            **{ax:grid.axes[ax]._boundary for ax in grid.axes.keys()},
            **boundary
        },
        autoparse_metadata=False
    )
    if "Z_metrics" in vars(grid):
        new_grid.Z_metrics = grid.Z_metrics

    return new_grid