import warnings

import gsw
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram

from xwmt.compute import (
    Jlmass_from_Qm_lm_l,
    expand_surface_to_3D,
    get_xgcm_grid_vertical,
    hldot_from_Jl,
    hldot_from_Ldot_hldotmass,
    lbin_define,
)


class swmt:
    """
    A class object with multiple functions to do 2d surface watermass transformation analysis.
    """

    terms_dict = {"heat": "tos", "salt": "sos"}

    flux_heat_dict = {
        "total": "hfds",
        "latent": "hflso",
        "sensible": "hfsso",
        "longwave": "rlntds",
        "shortwave": "rsntds",
        "frazil_ice": "hfsifrazil",
        "mass_transfer": "heat_content_surfwater",
    }

    flux_salt_dict = {"total": "sfdsi", "basal_salt": "sfdsi"}

    flux_mass_dict = {
        "total": "wfo",
        "rain_and_ice": "prlq",
        "snow": "prsn",
        "evaporation": "evs",
        "rivers": "friver",
        "icebergs": "ficeberg",
    }

    lambdas_dict = {
        "heat": ["theta"],
        "salt": ["salt"],
        "density": ["sigma0", "sigma1", "sigma2", "sigma3", "sigma4"],
    }

    def lambdas(self, lstr=None):
        if lstr is None:
            return sum(self.lambdas_dict.values(), [])
        else:
            return self.lambdas_dict.get(lstr, None)

    def fluxes(self, lstr=None):
        if lstr == "mass":
            dic = self.flux_mass_dict
        elif lstr == "salt":
            dic = self.flux_salt_dict
        elif lstr == "heat":
            dic = self.flux_heat_dict
        else:
            return

        keys = [key for (key, val) in dic.items() if val is not None and val in self.ds]
        return keys

    def __init__(self, ds, Cp=3992.0, rho=1035.0, alpha=None, beta=None, teos10=True):
        """
        Create a new surface watermass transformation object from an input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant surface fluxes along with grid information.
        Cp : float, optional
            Specify value for the specific heat capacity (in J/kg/K). Cp=3992.0 by default.
        rho : float, optional
            Specify value for the reference seawater density (in kg/m^3). rho=1035.0 by default.
        alpha : float, optional
            Specify value for the thermal expansion coefficient (in 1/K). alpha=None by default.
            If alpha is not given (i.e., alpha=None), it is derived from salinty and temperature
            fields using `gsw_alpha`.
        beta : float, optional
            Specify value for the haline contraction coefficient (in kg/g). beta=None by default.
            If beta is not given (i.e., beta=None), it is derived from salinty and temperature
            fields using `gsw_beta`.
        teos10 : boolean, optional
            Use Thermodynamic Equation Of Seawater - 2010 (TEOS-10). True by default.
        """
        self.ds = ds.copy()
        self.Cp = Cp
        self.rho = rho
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        self.teos10 = teos10

        # Save all 2d variable names in ds that need to be expanded in the vertical
        self.variables = (
            list(self.terms_dict.values())
            + list(self.flux_heat_dict.values())
            + list(self.flux_salt_dict.values())
            + list(self.flux_mass_dict.values())
            + self.lambdas_dict["heat"]
            + self.lambdas_dict["salt"]
            + self.lambdas_dict["density"]
            + ["alpha", "beta"]
        )

        # Modify ds to use a pseudo vertical grid
        if (
            "lev_outer" not in self.ds
        ):  # TODO: Find a better way to check vertical dimensions using both lev_outer and lev
            self.ds["lev_outer"] = xr.DataArray(np.array([0.0, 5.0]), dims="lev_outer")
            self.ds["lev"] = xr.DataArray(np.array([2.5]), dims="lev")
            for var in self.ds.keys():
                if var in self.variables:
                    self.ds[var] = expand_surface_to_3D(
                        self.ds[var], self.ds["lev_outer"]
                    )
        # TODO: Add error message if lev and/or lev_outer in ds.

        # Create xgcm object with modified ds
        self.xgrid = get_xgcm_grid_vertical(self.ds, metrics=True, periodic=False)

    # Helper function to get variable name for given tendency
    def tend(self, tendency):
        return self.terms_dict.get(tendency, None)

    # Helper function to get variable name for given flux
    def flux(self, mass=None, salt=None, heat=None):
        if mass is not None:
            code = self.flux_mass_dict.get(mass, None)
        elif heat is not None:
            code = self.flux_heat_dict.get(heat, None)
        elif salt is not None:
            code = self.flux_salt_dict.get(salt, None)
        else:
            warnings.warn("Flux is not defined")
            return
        return code

    def dd(self, tendency, mass="total", salt="total", heat="total", decompose=None):

        tendcode = self.tend(tendency)
        fluxcode_mass = self.flux(mass=mass)

        if tendency == "heat":
            # Need to multiply mass flux by Cp to convert to energy flux (convert to W/m^2/degC)
            flux_mass = self.ds[fluxcode_mass] * self.Cp
            fluxcode_heat = self.flux(heat=heat)
            flux_arr = self.ds[fluxcode_heat]
            # Assume temperature of mass flux to be the same as sst
            scalar_in_mass = self.ds[self.tend("heat")]
        elif tendency == "salt":
            flux_mass = self.ds[fluxcode_mass]
            # Multiply salt tendency by 1000 to convert to g/m^2/s
            fluxcode_salt = self.flux(salt=salt)
            flux_arr = self.ds[fluxcode_salt] * 1000
            # Assume salinity of mass flux to be zero
            scalar_in_mass = xr.zeros_like(self.ds[self.tend("salt")]).rename(None)
            scalar_in_mass.attrs = {}
        else:
            warnings.warn("Tendency is not defined")
            return

        # When decompose option is used, other fluxes are set to zero
        if decompose == "mass":
            flux_arr = 0 * flux_arr
        if decompose == "heat":
            flux_mass = 0 * flux_mass
            if tendency == "salt":
                flux_arr = 0 * flux_arr
        if decompose == "salt":
            flux_mass = 0 * flux_mass
            if tendency == "heat":
                flux_arr = 0 * flux_arr

        return {
            "scalar": {"array": self.ds[tendcode]},
            "flux": {"array": flux_arr, "extensive": False, "boundary": True},
            "boundary": {
                "flux": flux_mass,
                "mass": True,
                "scalar_in_mass": scalar_in_mass,
            },
        }

    def calc_hldot_flux(self, dd):
        """
        Wrapper functions to determine h times lambda_dot from flux terms
        """

        if dd["flux"]["extensive"]:
            warnings.warn("Flux form must be intensive")
        else:
            hldotmass = None
            if dd["flux"]["boundary"]:
                if dd["boundary"]["mass"]:
                    Jlmass = Jlmass_from_Qm_lm_l(
                        dd["boundary"]["flux"],
                        dd["boundary"]["scalar_in_mass"],
                        dd["scalar"]["array"],
                    )
                    hldotmass = hldot_from_Jl(self.xgrid, Jlmass, dim="Z")
            hldot = hldot_from_Ldot_hldotmass(
                hldot_from_Jl(self.xgrid, dd["flux"]["array"], dim="Z"), hldotmass
            )
            return hldot

    def get_density(self, density_str=None):

        # Calculate pressure (dbar)
        if (
            "alpha" not in vars(self) or "beta" not in vars(self) or self.teos10
        ) and "p" not in vars(self):
            self.p = xr.apply_ufunc(
                gsw.p_from_z,
                -self.ds["lev_outer"],
                self.ds["lat"],
                0,
                0,
                dask="parallelized",
            )
        # Calculate absolute salinity (g/kg)
        if self.teos10 and "sa" not in vars(self):
            self.sa = xr.apply_ufunc(
                gsw.SA_from_SP,
                self.ds[self.tend("salt")]
                .where(self.ds["lev_outer"] == 0)
                .where(self.ds["wet"] == 1),
                self.p,
                self.ds["lon"],
                self.ds["lat"],
                dask="parallelized",
            )
        # Calculate conservative temperature (degC)
        if self.teos10 and "ct" not in vars(self):
            self.ct = xr.apply_ufunc(
                gsw.CT_from_t,
                self.sa,
                self.ds[self.tend("heat")]
                .where(self.ds["lev_outer"] == 0)
                .where(self.ds["wet"] == 1),
                self.p,
                dask="parallelized",
            )
        if not self.teos10 and ("sa" not in vars(self) or "ct" not in vars(self)):
            self.sa = self.ds.so
            self.ct = self.ds.thetao

        # Calculate thermal expansion coefficient alpha (1/K)
        if "alpha" not in vars(self):
            if "alpha" in self.ds:
                self.alpha = self.ds.alpha
            else:
                self.alpha = xr.apply_ufunc(
                    gsw.alpha, self.sa, self.ct, self.p, dask="parallelized"
                )

        # Calculate the haline contraction coefficient beta (kg/g)
        if "beta" not in vars(self):
            if "beta" in self.ds:
                self.beta = self.ds.beta
            else:
                self.beta = xr.apply_ufunc(
                    gsw.beta, self.sa, self.ct, self.p, dask="parallelized"
                )

        # Calculate potential density (kg/m^3)
        if density_str not in self.ds:
            if density_str == "sigma0":
                density = xr.apply_ufunc(
                    gsw.sigma0, self.sa, self.ct, dask="parallelized"
                )
            elif density_str == "sigma1":
                density = xr.apply_ufunc(
                    gsw.sigma1, self.sa, self.ct, dask="parallelized"
                )
            elif density_str == "sigma2":
                density = xr.apply_ufunc(
                    gsw.sigma2, self.sa, self.ct, dask="parallelized"
                )
            elif density_str == "sigma3":
                density = xr.apply_ufunc(
                    gsw.sigma3, self.sa, self.ct, dask="parallelized"
                )
            elif density_str == "sigma4":
                density = xr.apply_ufunc(
                    gsw.sigma4, self.sa, self.ct, dask="parallelized"
                )
            else:
                return self.alpha, self.beta, None
        else:
            return self.alpha, self.beta, self.ds[density_str]

        return self.alpha, self.beta, density.rename(density_str)

    def rho_tend(self, mass="total", salt="total", heat="total", decompose=None):

        if "alpha" in vars(self) and "beta" in vars(self):
            alpha, beta = self.alpha, self.beta
        else:
            (alpha, beta, _) = self.get_density()
        alpha = alpha.sum("lev_outer").expand_dims("lev")
        beta = beta.sum("lev_outer").expand_dims("lev")

        heat_dd = self.dd("heat", mass=mass, salt=salt, heat=heat, decompose=decompose)
        heat_tend = self.calc_hldot_flux(heat_dd)

        salt_dd = self.dd("salt", mass=mass, salt=salt, heat=heat, decompose=decompose)
        salt_tend = self.calc_hldot_flux(salt_dd)

        # Density tendency due to heat flux (kg/s/m^2)
        rho_tend_heat = -(alpha / self.Cp) * heat_tend

        # Density tendency due to salt/salinity (kg/s/m^2)
        rho_tend_salt = beta * salt_tend

        return rho_tend_heat, rho_tend_salt

    def calc_Fl(self, lstr, mass="total", salt="total", heat="total", decompose=None):
        """
        Get transformation rate (* m/s) and corresponding lambda

        Parameters
        ----------
        lstr : str
            Specifies lambda (e.g., 'theta', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of
            available lambdas.
        mass : str, optional
            Specifies mass flux term (e.g., 'rain_and_ice', 'evaporation', etc.). 'total' by
            default. Use `fluxes('mass')` to list all available terms.
        salt : str, optional
            Specifies salt flux term (e.g., 'basal_salt'). 'total' by default.
            Use `fluxes('salt')` to list all available terms.
        heat : array like, optional
            Specifies heat flux term (e.g., 'latent', 'sensible', etc.). 'total' by default.
            Use `fluxes('heat')` to list all available terms.
        decompose : str, optional {'mass','salt','heat'}
            Separate surface flux into components.

        Returns
        -------
        F : Transformation rates
        l : lambda
        """

        # Get F from tendency of heat (in W/m^2), lambda = theta
        if lstr == "theta":
            # degC m/s
            dd = self.dd("heat", mass=mass, salt=salt, heat=heat, decompose=decompose)
            if dd is not None:
                F = -self.calc_hldot_flux(dd) / (self.rho * self.Cp)
                l = (
                    dd["scalar"]["array"]
                    .sum("lev_outer")
                    .where(self.ds.wet == 1)
                    .expand_dims("lev")
                )
                return F, l

        # Get F from tendency of salt (in g/s/m^2), lambda = salt
        elif lstr == "salt":
            # g/kg m/s
            dd = self.dd("salt", mass=mass, salt=salt, heat=heat, decompose=decompose)
            if dd is not None:
                F = -self.calc_hldot_flux(dd) / self.rho
                l = (
                    dd["scalar"]["array"]
                    .sum("lev_outer")
                    .where(self.ds.wet == 1)
                    .expand_dims("lev")
                )
                return F, l

        # Get F from tendencies of density (in kg/s/m^2), lambda = density
        # Here we want to output 2 transformation rates:
        # (1) transformation due to heat tend, (2) transformation due to salt tend.
        elif lstr in self.lambdas("density"):
            # kg/m^3 m/s
            F = {}
            rhos = self.rho_tend(mass=mass, salt=salt, heat=heat, decompose=decompose)
            for idx, tend in enumerate(self.terms_dict.keys()):
                F[tend] = rhos[idx]
            (alpha, beta, l) = self.get_density(lstr)
            l = l.sum("lev_outer").where(self.ds["wet"] == 1).expand_dims("lev")
            return F, l

        return (None, None)

    def lbin_percentile(self, l, percentile=[0.05, 0.95], bins=30):
        """Specify the percentile and number of the bins"""
        l_sample = l.isel(time=0).chunk({"y": -1, "x": -1})
        vmin, vmax = l_sample.quantile(percentile, dim=l_sample.dims)
        return np.linspace(vmin, vmax, bins)

    def calc_F_transformed(
        self, lstr, bins=None, mass="total", salt="total", heat="total", decompose=None
    ):
        """
        Transform to lambda space
        """

        F, l = self.calc_Fl(lstr, mass=mass, salt=salt, heat=heat, decompose=decompose)

        if bins is None:
            bins = self.lbin_percentile(
                l
            )  # automatically find the right range based on the distribution in l

        if lstr in self.lambdas("density"):
            F_transformed = []
            for tend in self.terms_dict.keys():
                if F[tend] is not None:
                    F_transformed.append(
                        (
                            self.xgrid.transform(
                                F[tend],
                                "Z",
                                target=bins,
                                target_data=l,
                                method="conservative",
                            )
                            / np.diff(bins)
                        ).rename(tend)
                    )
            F_transformed = xr.merge(F_transformed)
        else:
            F_transformed = self.xgrid.transform(
                F, "Z", target=bins, target_data=l, method="conservative"
            ) / np.diff(bins)
        return F_transformed

    def calc_G(
        self,
        lstr,
        method="xhistogram",
        bins=None,
        mass="total",
        salt="total",
        heat="total",
        decompose=None,
    ):
        """
        Water mass transformation (G)
        """

        if method == "xhistogram" and lstr in self.lambdas("density"):
            F, l = self.calc_Fl(
                lstr, mass=mass, salt=salt, heat=heat, decompose=decompose
            )
            if bins is None and l is not None:
                bins = self.lbin_percentile(
                    l
                )  # automatically find the right range based on the distribution in l
            G = []
            for (tend, code) in self.terms_dict.items():
                if F[tend] is not None:
                    _G = (
                        (
                            histogram(
                                l.where(~np.isnan(F[tend])),
                                bins=[bins],
                                dim=["x", "y", "lev"],
                                weights=(F[tend] * self.ds["areacello"]).where(
                                    ~np.isnan(F[tend])
                                ),
                            )
                            / np.diff(bins)
                        )
                        .rename({l.name + "_bin": l.name})
                        .rename(tend)
                    )
                    G.append(_G)
            return xr.merge(G)
        elif method == "xhistogram":
            F, l = self.calc_Fl(
                lstr, mass=mass, salt=salt, heat=heat, decompose=decompose
            )
            if bins is None and l is not None:
                bins = self.lbin_percentile(
                    l
                )  # automatically find the right range based on the distribution in l
            if F is not None and l is not None:
                G = (
                    (
                        histogram(
                            l.where(~np.isnan(F)),
                            bins=[bins],
                            dim=["x", "y", "lev"],
                            weights=(F * self.ds["areacello"]).where(~np.isnan(F)),
                        )
                        / np.diff(bins)
                    )
                    .rename({l.name + "_bin": l.name})
                    .rename(lstr)
                )
                return G
        elif method == "xgcm":
            F_transformed = self.calc_F_transformed(
                lstr, bins=bins, mass=mass, salt=salt, heat=heat, decompose=decompose
            )
            if F_transformed is not None and len(F_transformed):
                G = (F_transformed * self.ds["areacello"]).sum(["x", "y"])
                return G
            return F_transformed

    # Calculate the sum of grouped terms
    def _sum(self, ds, newterm, terms):
        das = []
        for term in terms:
            if term in ds:
                das.append(ds[term])
                del ds[term]
        if len(das):
            ds[newterm] = sum(das)
        return ds

    def G(self, lstr, *args, **kwargs):
        """
        Water mass transformation (G)

        Parameters
        ----------
        lstr : str
            Specifies lambda (e.g., 'theta', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of
            available lambdas.
        term : str, optional
            Specifies process term (e.g., 'boundary forcing', 'vertical diffusion', etc.).
            Use `processes()` to list all available terms.
        method : str {'xhistogram' (default), 'xgcm'}
            The calculation can be either done with xhistogram (default) or the xgcm `transform`.
            If not specified, default will be used.
        bins : array like, optional
            np.array with lambda values specifying the edges for each bin. If not specidied, array
            will be automatically derived from
            the scalar field of lambda (e.g., temperature).
        group_tend : boolean, optional
            Specify whether heat and salt tendencies are summed together (True) or
            kept separated (False). True by default.
        mass : str, optional
            Specifies mass flux term (e.g., 'rain_and_ice', 'evaporation', etc.).
            'total' by default. Use `fluxes('mass')` to list all available terms.
        salt : str, optional
            Specifies salt flux term (e.g., 'basal_salt'). 'total' by default.
            Use `fluxes('salt')` to list all available terms.
        heat : array like, optional
            Specifies heat flux term (e.g., 'latent', 'sensible', etc.). 'total' by default.
            Use `fluxes('heat')` to list all available terms.
        decompose : str, optional {'mass','salt','heat'}
            Decompose watermass transformation for a given set of surface fluxes (mass, salt or
            heat fluxes). None by default.
            This method will overwrite group_tend, mass, salt and heat arguments.
            To calculate water mass trasnformation for a specifc flux term use mass, salt or
            heat argument.

        Returns
        -------
        G : {xarray.DataArray, xarray.Dataset}
            The water mass transformation along lambda. G is xarray.DataArray for decompose=None
            and group_tend=True.
            G is xarray.DataSet for decompose={'mass','salt','heat'} or group_tend=False.
        """

        # Extract the default function args
        decompose = kwargs.get("decompose", None)
        group_tend = kwargs.pop("group_tend", True)

        if group_tend == True and decompose is None:
            G = self.calc_G(lstr, *args, **kwargs)
            self._sum(G, "total", ["heat", "salt"])
        elif group_tend == False and decompose is None:
            G = self.calc_G(lstr, *args, **kwargs)
        elif lstr == "theta" and decompose == "heat":
            keys = [key for key in self.fluxes("heat")]
            G = []
            for key in keys:
                _G = self.calc_G(lstr, heat=key, *args, **kwargs).rename(key)
                G.append(_G)
            G = xr.merge(G)
        elif lstr in self.lambdas("density") and decompose == "heat":
            keys = [key for key in self.fluxes("heat")]
            G = []
            for key in keys:
                _G = (
                    self.calc_G(lstr, heat=key, *args, **kwargs)
                    .rename({"heat": key})
                    .drop("salt")
                )
                G.append(_G)
            G = xr.merge(G)
        elif lstr in self.lambdas("density") and decompose == "salt":
            keys = [key for key in self.fluxes("salt")]
            G = []
            for key in keys:
                _G = (
                    self.calc_G(lstr, salt=key, *args, **kwargs)
                    .rename({"salt": key})
                    .drop("heat")
                )
                G.append(_G)
            G = xr.merge(G)
        elif lstr in self.lambdas("density") and decompose == "mass":
            keys = [key for key in self.fluxes("mass")]
            G = []
            for key in keys:
                _G = self.calc_G(lstr, mass=key, *args, **kwargs)
                if group_tend == True:
                    self._sum(_G, key, ["heat", "salt"])
                else:
                    _G = _G.rename({"heat": key + "_heat", "salt": key + "_salt"})
                G.append(_G)
            G = xr.merge(G)

        if isinstance(G, xr.Dataset) and len(G) == 1:
            return G[list(G.data_vars)[0]]
        else:
            return G

    def F(self, lstr, group_tend=True, **kwargs):
        """
        Wrapper function for calc_F_transformed() with additional group_tend argument
        """

        F_transformed = self.calc_F_transformed(lstr, **kwargs)
        if group_tend:
            self._sum(F_transformed, "total", ["heat", "salt"])
            if len(F_transformed) == 1:
                return F_transformed[list(F_transformed.data_vars)[0]]
            else:
                return F_transformed
        return F_transformed

    def isosurface_mean(self, lstr, val, ti=None, tf=None, dl=0.1, **kwargs):
        """
        Mean transformation across lambda isosurface(s).

        Parameters
        ----------
        lstr : str
            Specifies lambda (e.g., 'theta', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of
            available lambdas.
        val : float or ndarray
            Value(s) of lambda for which isosurface(s) is/are defined
        ti : str
            Starting date. ti=None by default.
        tf : str
            End date. tf=None by default.
        dl : float
            Width of lamba bin (delta) for which isosurface(s) is/are defined.
        group_tend : boolean, optional
            Specify whether heat and salt tendencies are summed together (True) or kept separated
            (False). True by default.
        mass : str, optional
            Specifies mass flux term (e.g., 'rain_and_ice', 'evaporation', etc.).
            'total' by default.
            Use `fluxes('mass')` to list all available terms.
        salt : str, optional
            Specifies salt flux term (e.g., 'basal_salt'). 'total' by default.
            Use `fluxes('salt')` to list all available terms.
        heat : array like, optional
            Specifies heat flux term (e.g., 'latent', 'sensible', etc.). 'total' by default.
            Use `fluxes('heat')` to list all available terms.
        decompose : str, optional {'mass','salt','heat'}
            Decompose watermass transformation for a given set of surface fluxes (mass, salt or
            heat fluxes). None by default.
            This method will overwrite group_tend, mass, salt and heat arguments.
            To calculate water mass trasnformation for a specifc flux term use mass, salt or
            heat argument.

        Returns
        -------
        F_mean : {xarray.DataArray, xarray.Dataset}
            Spatial field of mean transformation at a given (set of) lambda value(s).
            F_mean is xarray.DataArray for decompose=None and group_tend=True.
            F_mean is xarray.DataSet for decompose={'mass','salt','heat'} or group_tend=False.
        """

        if lstr not in self.lambdas("density"):
            tendency = [k for k, v in self.lambdas_dict.items() if v[0] == lstr]
            if len(tendency) == 1:
                tendcode = self.terms_dict.get(tendency[0], None)
            else:
                warnings.warn("Tendency is not defined")
                return
        else:
            tendcode = lstr

        # Define bins based on val
        kwargs["bins"] = lbin_define(np.min(val) - dl, np.max(val) + dl, dl)

        # Calculate spatiotemporal field of transformation
        F = self.F(lstr, **kwargs)
        # TODO: Preferred method should be ndays_standard if calendar type is 'noleap'. Thus, avoiding to load the full time array
        if "time_bounds" in self.ds:
            # Extract intervals (units are in ns)
            deltat = self.ds.time_bounds[:, 1].values - self.ds.time_bounds[:, 0].values
            # Convert intervals to days
            dt = xr.DataArray(
                deltat, coords=[self.ds.time], dims=["time"], name="days per month"
            ) / np.timedelta64(1, "D")
        elif (
            "calendar_type" in self.ds.time.attrs
            and self.ds.time.attrs["calendar_type"].lower() == "noleap"
        ):
            # Number of days in each month
            n_years = len(np.unique(F.time.dt.year))
            # Monthly data
            dm = np.diff(F.indexes["time"].month)
            udm = np.unique([m + 12 if m == -11 else m for m in dm])
            if np.array_equal(udm, [1]):
                ndays_standard = np.array(
                    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                )
                assert np.sum(ndays_standard) == 365
                dt = xr.DataArray(
                    ndays_standard[F.time.dt.month.values - 1],
                    coords=[self.ds.time],
                    dims=["time"],
                    name="days per month",
                )
            # Annual data
            dy = np.diff(F.indexes["time"].year)
            udy = np.unique(dy)
            if np.array_equal(udy, [1]):
                dt = xr.DataArray(
                    np.tile(365, n_years),
                    coords=[self.ds.time],
                    dims=["time"],
                    name="days per year",
                )
        else:
            # TODO: Create dt with ndays_standard but output warning that calendar_type is not specified.
            # warnings.warn('Unsupported calendar type')
            print("Unsupported calendar type", self.ds.time.attrs)
            return

        # Convert to dask array for lazy calculations
        dt = dt.chunk(1)
        F_mean = (
            F.sel({tendcode: val}, method="nearest").sel(time=slice(ti, tf))
            * dt.sel(time=slice(ti, tf))
        ).sum("time") / dt.sel(time=slice(ti, tf)).sum("time")
        return F_mean
