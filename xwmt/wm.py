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
        teos10 : boolean, optional
            Use Thermodynamic Equation Of Seawater - 2010 (TEOS-10). True by default.
        """

        self.ds = ds.copy()
        self.grid = grid
        self.t_name = t_name
        self.s_name = s_name
        self.teos10 = teos10
        self.cp = cp
        self.rho_ref = rho_ref
        
    def get_density(self, density_str=None):
        # Variables needed to calculate alpha, beta and density
        if (
            "alpha" not in self.ds or "beta" not in self.ds or self.teos10
        ) and "p" not in vars(self):
            self.ds['p'] = xr.apply_ufunc(
                gsw.p_from_z, -self.ds.z_l, self.ds.lat, 0, 0, dask="parallelized"
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
        if density_str is None:
            return None
        
        else:
            if density_str not in self.ds:
                if "sigma" in density_str:
                    density = xr.apply_ufunc(
                        getattr(gsw, density_str), self.ds.sa, self.ds.ct, dask="parallelized"
                    )
                else:
                    return None
            else:
                return self.ds[density_str]

            return density.rename(density_str)