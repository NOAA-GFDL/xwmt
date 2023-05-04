import numpy as np
import xarray as xr
import gsw
import warnings

class WaterMass:
    """
    A class object with multiple methods to do full 3d watermass transformation analysis.
    """
    def __init__(
        self,
        ds,
        grid,
        t_name="thetao",
        s_name="salt",
        h_name="thkcello",
        teos10=True,
        cp=3992.0,
        rho_ref=1035.0,
        ):
        """
        Create a new watermass object from an input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant tendencies and/or surface fluxes along with grid information.
        grid: xgcm.Grid
            Contains information about ocean model grid discretization, e.g. coordinates and metrics.
        t_name: str
            Name of potential temperature variable [in degrees Celsius] in ds.
        s_name: str
            Name of practical salinity variable [in psu] in ds.
        h_name: str
            Name of thickness variable [in m] in ds.
        teos10 : boolean, optional
            Use Thermodynamic Equation Of Seawater - 2010 (TEOS-10). True by default.
        cp: float
            Value of specific heat capacity.
        rho_ref: float
            Value of reference potential density. Note: WaterMass is assumed to be Boussinesq.
        """
        
        self.ds = ds.copy()
        self.grid = grid
        self.t_name = t_name
        self.s_name = s_name
        self.h_name = h_name
        self.teos10 = teos10
        self.cp = cp
        self.rho_ref = rho_ref
        
        if self.h_name in self.ds:
            # Conservatively interpolate thickness to cell interfaces, needed to
            # estimate depth of layer centers and compute surface flux divergences
            Z_center_extended = np.concatenate((
                self.ds[self.grid.axes['Z'].coords['outer']][np.array([0])].values,
                self.ds[self.grid.axes['Z'].coords['center']].values,
                self.ds[self.grid.axes['Z'].coords['outer']][np.array([-1])].values
            ))
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                self.ds[f'{h_name}_i'] = self.grid.transform(
                    self.ds[h_name].fillna(0.),
                    "Z",
                    Z_center_extended,
                    method="conservative"
                ).assign_coords({
                    self.grid.axes['Z'].coords['outer']:
                    self.ds[self.grid.axes['Z'].coords['outer']].values
                })
            setattr(self.grid, "Z_metrics", {
                    "center": self.ds[h_name],
                    "outer": self.ds[f'{h_name}_i']
                }
            )
        
    def get_density(self, density_name=None):
        """
        Compute density variable from layer temperature, salinity, and thickness.
        Uses the TEOS10 algorithm from the `gsw` package by default, unless "alpha"
        and "beta" variables are already provided in `self.ds`.

        Parameters
        ----------
        density_name: str
            Name of density variable. Supported density variables are: "sigma0",
            "sigma1", "sigma2", "sigma3", "sigma4" (corresponding to functions
            of the same name in the `gsw` package).
        """
        
        if self.t_name not in self.ds:
            raise ValueError(f"ds must include temperature variable\
            defined by kwarg t_name (default: {self.t_name}).")
        if self.s_name not in self.ds:
            raise ValueError(f"ds must include salinity variable\
            defined by kwarg t_name (default: {self.s_name}).")
        if self.h_name not in self.ds:
            raise ValueError(f"ds must include thickness variable\
            defined by kwarg t_name (default: {self.h_name}).")
        
        if (
            "alpha" not in self.ds or "beta" not in self.ds or self.teos10
        ) and "p" not in vars(self):
            self.ds['z'] = self.grid.cumsum(self.grid.Z_metrics["outer"], "Z")
            self.ds['p'] = xr.apply_ufunc(
                gsw.p_from_z, -self.ds.z, self.ds.lat, 0, 0, dask="parallelized"
            )
        if self.teos10 and "sa" not in self.ds:
            self.ds['sa'] = xr.apply_ufunc(
                gsw.SA_from_SP,
                self.ds[self.s_name],
                self.ds.p,
                self.ds.lon,
                self.ds.lat,
                dask="parallelized",
            )
        if self.teos10 and "ct" not in self.ds:
            self.ds['ct'] = xr.apply_ufunc(
                gsw.CT_from_t, self.ds.sa, self.ds[self.t_name], self.ds.p, dask="parallelized"
            )
        if not self.teos10 and ("sa" not in vars(self) or "ct" not in vars(self)):
            self.ds['sa'] = self.ds[self.s_name]
            self.ds['ct'] = self.ds[self.t_name]

        # Calculate thermal expansion coefficient alpha (1/K)
        if "alpha" not in self.ds:
            self.ds['alpha'] = xr.apply_ufunc(
                gsw.alpha, self.ds.sa, self.ds.ct, self.ds.p, dask="parallelized"
            )

        # Calculate the haline contraction coefficient beta (kg/g)
        if "beta" not in self.ds:
            self.ds['beta'] = xr.apply_ufunc(
                gsw.beta, self.ds.sa, self.ds.ct, self.ds.p, dask="parallelized"
            )

        # Calculate potential density (kg/m^3)
        if density_name is None:
            return None
        
        else:
            if density_name not in self.ds:
                if "sigma" in density_name:
                    density = xr.apply_ufunc(
                        getattr(gsw, density_name), self.ds.sa, self.ds.ct, dask="parallelized"
                    )
                else:
                    return None
            else:
                return self.ds[density_name]

            return density.rename(density_name)

    def get_outcrop_lev(self, position="center", incrop=False):
        """
        Find the vertical coordinate level that outcrops at the sea surface, broadcast
        across all other dimensions of the thickness variable (`self.ds.h_name`).

        Parameters
        ----------
        position: str
            Position of the desired vertical coordinate in the `self.grid` instance of `xgcm.Grid`.
            Default: "center". Other supported option is "outer".
        incrop: bool
            Default: False. If True, returns the seafloor incrop level instead. 
        """
        if self.h_name not in self.ds:
            raise ValueError(f"ds must include thickness variable\
            defined by kwarg t_name (default: {self.h_name}).")
        
        z_coord = self.grid.axes['Z'].coords[position]
        dk = int(2*incrop - 1)
        h = self.grid.Z_metrics[position]
        cumh = h.sel(
                {z_coord: self.ds[z_coord][::dk]}
            ).cumsum(z_coord)
        return cumh.idxmax(z_coord).where(cumh.isel({z_coord:-1})!=0.)
        
    def sel_outcrop_lev(self, da, incrop=False, position="center", **kwargs):
        """
        Select `da` at the vertical coordinate level that outcrops at the sea surface. Assumes
        `da` has the same dimensions as `self.grid.Z_metrics[position].sel(**kwargs)`, and broadcasts in all
        other dimensions than the vertical.

        Parameters
        ----------
        position: str
            Position of the desired vertical coordinate in the `self.grid` instance of `xgcm.Grid`.
            Default: "center". Other supported option is "outer".
        incrop: bool
            Default: False. If True, returns the seafloor incrop level instead.
        **kwargs:
            Passed to the `xr.DataArray.sel` method on the vertical thickness.
        """
        if self.h_name not in self.ds:
            raise ValueError(f"ds must include thickness variable\
            defined by kwarg t_name (default: {self.h_name}).")

        z_coord = self.grid.axes['Z'].coords[position]
        dk = int(2*incrop - 1)
        h = self.grid.Z_metrics[position].sel(**kwargs)
        if da.dims != h.dims:
            raise ValueError(
                "`da` must have the same dimensions as\
                `self.grid.Z_metrics[position].sel(**kwargs)`"
            )
        cumh = h.sel(
                {z_coord: self.ds[z_coord][::dk]}
            ).cumsum(z_coord)
        return da.sel(
            {z_coord: cumh.idxmax(z_coord)}
        ).where(cumh.isel({z_coord:-1})!=0.)
    
    def expand_surface_array_vertically(self, da_surf):
        """
        Expand surface xr.DataArray (with no "Z"-dimension coordinate) in the vertical,
        filling with zeros in all layers except the one that outcrops.
        
        Parameters
        ----------
        da_surf: xarray.DataArray
            Variable that is to be expanded in the vertical.
        """
        z_coord = self.grid.axes['Z'].coords['outer']
        return (
            da_surf.expand_dims({z_coord: self.ds[z_coord]})
            .where(
                self.ds[z_coord] ==
                self.get_outcrop_lev(position="outer"),
                0.
            )
        )

    def bin_percentile(self, da, percentiles=[0.05, 0.95], nbins=100, surface=False):
        """
        Specify bins based on the distribution of `da`, excluding outliers.
        
        Parameters
        ----------
        da: xarray.DataArray
            Variable used to determine bins.
        percentiles: list
            List of length 2 containing the upper and lower percentiles to bound the array of bins.
            Default: [0.05, 0.95].
        nbins: int
            Number of bins. Default: 100.
        surface: bool
            Default: False. If True, compute percentiles only from the outcropping layer of `da`.
        """
        if surface:
            da=self.sel_outcrop_lev(da)
        if "time" in da.dims:
            da=da.isel(time=0)
        vmin, vmax = da.quantile(percentiles, dim=da.dims)
        return np.linspace(vmin, vmax, nbins)

    def zonal_mean(self, da, oceanmask_name="wet"):
        """
        Compute area-weighted zonal mean.
        
        Parameters
        ----------
        da: xarray.DataArray
            Data array to be averaged.
        oceanmask_name: str
            Name of ocean mask xr.DataArray in `self.ds`. Default: "wet".
        """
        x_name = grid.axes['X'].coords['center']
        area = self.grid.get_metric(da, ['X', 'Y'])
        num = (da * area * self.ds[landmask_name]).sum(dim=x_name)
        denom = (area * self.ds[landmask_name]).sum(dim=x_name)
        return num / denom