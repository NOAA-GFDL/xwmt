import copy
import numpy as np
import xarray as xr
import gsw
import warnings

from xwmt.wm import WaterMass
from xwmt.compute import (
    calc_hlamdot_tendency,
    bin_define,
)

class WaterMassTransformations(WaterMass):
    """
    A class
    """
    def __init__(
        self,
        ds,
        grid,
        budgets_dict,
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
        budgets_dict: dict
            Nested dictionary containing information about groupings of tendency variable names.
            See `xwmt/conventions` for examples of how this dictionary should be structued.
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

        super().__init__(
            ds,
            grid,
            t_name=t_name,
            s_name=s_name,
            h_name=h_name,
            teos10=teos10,
            cp=cp,
            rho_ref=rho_ref
        )
        
        self.terms_dict = {
            "heat": self.t_name,
            "salt": self.s_name,
        }
        
        self.budgets_dict = copy.deepcopy(budgets_dict)
        for (component, cdict) in self.budgets_dict.items():
            if 'surface_flux' in cdict:
                self.budgets_dict[component]['surface_flux'] = {
                    f"surface_flux_{term}":v
                    for (term,v) in cdict['surface_flux'].items()
                }

        for (term, bdict) in self.budgets_dict.items():
            setattr(self, f"processes_{term}_dict", {})
            for ptype, _processes in bdict.items():
                if ptype in ["lhs", "rhs", "surface_flux"]:
                    getattr(self, f"processes_{term}_dict").update(_processes)

        self.lambdas_dict = {
            "heat": "temperature",
            "salt": "salinity",
            "density": ["sigma0", "sigma1", "sigma2", "sigma3", "sigma4"],
        }

    def lambdas(self, lambda_name=None):
        if lambda_name is None:
            return sum(self.lambdas_dict.values(), [])
        else:
            return self.lambdas_dict.get(lambda_name, None)

    # Helper function to get variable name for given process term
    def process(self, tendency, term):
        # Organize by scalar and tendency
        if tendency == "heat":
            termcode = self.processes_heat_dict.get(term, None)
        elif tendency == "salt":
            termcode = self.processes_salt_dict.get(term, None)
        else:
            warnings.warn(f"Tendency {tendency} is not defined")
            return
        tendcode = self.terms_dict.get(tendency, None)
        return (tendcode, termcode)

    # Helper function to list available processes
    def processes(self, check=True):
        processes = self.processes_heat_dict.keys() | self.processes_salt_dict.keys()
        if check:
            _processes = []
            for process in processes:
                p1 = self.processes_salt_dict.get(process, None)
                p2 = self.processes_heat_dict.get(process, None)
                if ((p1 is None) or (p1 is not None and p1 in self.ds)) and (
                    (p2 is None) or (p2 is not None and p2 in self.ds)
                ):
                    _processes.append(process)
            return _processes
        else:
            return processes

    def datadict(self, tendency, term):
        (tendcode, termcode) = self.process(tendency, term)
        # tendcode: tendency form (heat or salt)
        # termcode: process term (e.g., boundary_forcing)
        if termcode is None or termcode not in self.ds:
            return

        if tendency == "salt":
            # Multiply salt tendency by 1000 to convert to g/m^2/s
            tend_arr = self.ds[termcode] * 1000
        else:
            tend_arr = self.ds[termcode]

        n_zcoords = len([
            c for c in self.grid.axes['Z'].coords.values()
            if c in self.ds[termcode].dims
        ])
        if n_zcoords>0:
            return {
                "scalar": {"array": self.ds[tendcode]},
                "tendency": {
                    "array": tend_arr,
                    "extensive": True,
                    "boundary": False
                },
            }
        # if no vertical coordinate in tend_arr, assume it is a surface flux
        elif n_zcoords==0:
            if tendency == "heat":
                # Need to multiply mass flux by cp to convert
                # to energy flux (in W/m^2/degC)
                mass_flux = self.expand_surface_array_vertically(
                    self.ds["wfo"] * self.cp,
                )
                scalar_in_mass = self.expand_surface_array_vertically(
                    self.ds["tos"],
                )
            elif tendency == "salt":
                mass_flux = self.expand_surface_array_vertically(
                    self.ds["wfo"],
                )
                scalar_in_mass = self.expand_surface_array_vertically(
                    xr.zeros_like(self.ds["sos"]),
                )
            else:
                raise ValueError(f"termcode {termcode} not yet supported.")
            return {
                "scalar": {"array": self.ds[tendcode]},
                "tendency": {
                    "array": self.expand_surface_array_vertically(
                        tend_arr
                    ),
                    "extensive": True,
                    "boundary": True
                },
                "boundary": {
                    "flux": mass_flux,
                    "mass": True,
                    "scalar_in_mass": scalar_in_mass,
                },
            }

    def rho_tend(self, term):
        """
        Calculate the tendency of the locally-referenced potential density.
        """

        if "alpha" not in self.ds or "beta" not in self.ds:
            self.get_density()

        # Either heat or salt tendency/flux may not be used
        rho_tend_heat, rho_tend_salt = None, None

        datadict = self.datadict("heat", term)
        if datadict is not None:
            heat_tend = calc_hlamdot_tendency(self.grid, self.datadict("heat", term))
            # Density tendency due to heat flux (kg/s/m^2)
            rho_tend_heat = -(self.ds.alpha / self.cp) * heat_tend

        datadict = self.datadict("salt", term)
        if datadict is not None:
            salt_tend = calc_hlamdot_tendency(self.grid, self.datadict("salt", term))
            # Density tendency due to salt/salinity (kg/s/m^2)
            rho_tend_salt = self.ds.beta * salt_tend

        return rho_tend_heat, rho_tend_salt

    def calc_hlamdot_and_lambda(self, lambda_name, term):
        """
        Get layer-integrated extensive tracer tendencies (* m/s) and corresponding scalar field of lambda
        lambda_name: str
            Specifies lambda
        term: str
            Specifies process term
        """

        # Get layer-integrated potential temperature tendency
        # from tendency of heat (in W/m^2), lambda = temperature
        if lambda_name == "temperature":
            datadict = self.datadict("heat", term)
            if datadict is not None:
                hlamdot = calc_hlamdot_tendency(self.grid, datadict) / (self.rho_ref * self.cp)
                lam = datadict["scalar"]["array"]

        # Get layer-integrated salinity tendency
        # from tendency of salt (in g/s/m^2), lambda = salinity
        elif lambda_name == "salinity":
            datadict = self.datadict("salt", term)
            if datadict is not None:
                hlamdot = calc_hlamdot_tendency(self.grid, datadict) / self.rho_ref
                # TODO: Accurately define salinity field (What does this mean? - HFD)
                lam = datadict["scalar"]["array"]

        # Get layer-integrated potential density tendencies (in kg/s/m^2)
        # from heat and salt, lambda = density
        # Here we want to output 2 separate components of the transformation rates:
        # (1) transformation due to heat tend, (2) transformation due to salt tend
        elif lambda_name in self.lambdas("density"):
            rhos = self.rho_tend(term)
            hlamdot = {}
            for idx, tend in enumerate(self.terms_dict.keys()):
                hlamdot[tend] = rhos[idx]
            lam = self.get_density(lambda_name)
        
        else:
            raise ValueError(f"{lambda_name} is not a supported lambda.")
        
        try:
            return hlamdot, lam
        
        except NameError:
            return None, None

    def transform_hlamdot(self, lambda_name, term, bins=None):
        """
        Compute extensive tendencies and transform them into lambda space
        along the vertical ("Z") dimension.
        """

        hlamdot, lam = self.calc_hlamdot_and_lambda(lambda_name, term)
        if hlamdot is None:
            return

        if bins is None:
            bins = self.bin_percentile(lam, surface=True)

        # Interpolate lambda to the cell interfaces
        lam_i = (
            self.grid.interp(lam, "Z", boundary="extend")
            .chunk({self.grid.axes['Z'].coords['outer']: -1})
            .rename(lam.name)
        )

        if lambda_name in self.lambdas("density"):
            hlamdot_transformed = []
            for tend in self.terms_dict.keys():
                (tendcode, termcode) = self.process(tend, term)
                if hlamdot[tend] is not None:
                    hlamdot_transformed.append(
                        (
                            self.grid.transform(
                                hlamdot[tend],
                                "Z",
                                target=bins,
                                target_data=lam_i,
                                method="conservative",
                            )
                            / np.diff(bins)
                        ).rename(termcode)
                    )
            hlamdot_transformed = xr.merge(hlamdot_transformed)
        else:
            (tendcode, termcode) = self.process(
                "salt" if lambda_name == "salinity" else "heat", term
            )
            hlamdot_transformed = (
                self.grid.transform(
                    hlamdot, "Z", target=bins, target_data=lam_i, method="conservative"
                )
                / np.diff(bins)
            ).rename(termcode)
        return hlamdot_transformed

    def transform_hlamdot_and_integrate(self, lambda_name, term=None, bins=None):
        """
        Compute extensive tendencies, transform them into lambda space
        along the vertical ("Z") dimension, and integrate along the
        horizontal dimensions ("X", "Y").
        """

        # If term is not given, use all available process terms
        if term is None:
            wmts = []
            for term in self.processes(False):
                wmt = self.transform_hlamdot_and_integrate(lambda_name, term, bins)
                if wmt is not None:
                    wmts.append(wmt)
            return xr.merge(wmts)

        hlamdot_transformed = self.transform_hlamdot(lambda_name, term, bins=bins)
        if hlamdot_transformed is not None and len(hlamdot_transformed):
            dA = self.grid.get_metric(hlamdot_transformed, ['X', 'Y'])
            wmt = (hlamdot_transformed * dA).sum(dA.dims)
            # rename dataarray only (not dataset)
            if isinstance(wmt, xr.DataArray):
                return wmt.rename(hlamdot_transformed.name)
            return wmt
        return hlamdot_transformed

    ### Helper function to groups terms based on density components (sum_components)
    ### and physical processes (group_processes)
    # Calculate the sum of grouped terms
    def _sum_terms(self, ds, newterm, terms):
        das = []
        for term in terms:
            if term in ds:
                das.append(ds[term])
        if len(das):
            ds[newterm] = sum(das)

    def _group_processes(self, hlamdot):
        if hlamdot is None:
            return
        for component in ["heat", "salt"]:
            process_dict = getattr(self, f"processes_{component}_dict")
            self._sum_terms(
                hlamdot,
                f"external_forcing_{component}",
                [
                    process_dict["boundary_forcing"],
                    process_dict["frazil_ice"],
                    process_dict["geothermal"],
                ],
            )
            self._sum_terms(
                hlamdot,
                f"diffusion_{component}",
                [
                    process_dict["vertical_diffusion"],
                    process_dict["neutral_diffusion"]
                ]
            )
            self._sum_terms(
                hlamdot,
                f"advection_{component}",
                [
                    process_dict["horizontal_advection"],
                    process_dict["vertical_advection"]
                ]
            )
            self._sum_terms(
                hlamdot,
                f"diabatic_forcing_{component}",
                [
                    f"external_forcing_{component}",
                    f"diffusion_{component}"
                ]
            )
            self._sum_terms(
                hlamdot,
                f"total_tendency_{component}",
                [
                    f"advection_{component}",
                    f"diabatic_forcing_{component}"
                ]
            )
        return hlamdot

    def _sum_components(self, hlamdot, group_processes = False):
        if hlamdot is None:
            return
        
        for proc in self.processes():
            self._sum_terms(
                hlamdot,
                proc,
                [
                    getattr(self, f"processes_{component}_dict")[proc]
                    for component in ["heat", "salt"]
                ]
            )
        if group_processes:
            for proc in ["external_forcing", "diffusion", "advection", "diabatic_forcing", "total_tendency"]:
                self._sum_terms(hlamdot, proc, [f"{proc}_{component}" for component in ["heat", "salt"]])
        return hlamdot

    def map_transformations(self, lambda_name, term=None, sum_components=True, group_processes=False, **kwargs):
        """
        Wrapper function for transform_hlamdot() to group terms based
        on tendency terms (heat, salt) and processes.
        """
        # If term is not given, use all available process terms
        if term is None:
            _local_transformations = []
            for term in self.processes(False):
                _local_transformation = self.transform_hlamdot(lambda_name, term, sum_components=False, group_processes=False, **kwargs)
                if _local_transformation is not None:
                    _local_transformations.append(_local_transformation)
            local_transformations = xr.merge(_local_transformations)
        else:
            # If term is given
            local_transformations = self.transform_hlamdot(lambda_name, term, **kwargs)
            if isinstance(local_transformation, xr.DataArray):
                local_transformations = local_transformations.to_dataset()

        if group_processes:
            local_transformations = self._group_processes(local_transformations)
        if sum_components:
            local_transformations = self._sum_components(local_transformations, group_processes=group_processes)

        return local_transformations

    def integrate_transformations(self, lambda_name, *args, **kwargs):
        """
        Computed horizontally-integrated water mass transformations.

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'temperature', 'salinity', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        term : str, optional
            Specifies process term (e.g., 'boundary_forcing', 'vertical_diffusion', etc.). Use `processes()` to list all available terms.
        bins : array like, optional
            np.array with lambda values specifying the edges for each bin. If not specidied, array will be automatically derived from
            the scalar field of lambda (e.g., temperature).
        sum_components : boolean, optional
            Specify whether heat and salt tendencies are summed together (True) or kept separated (False). True by default.
        group_processes: boolean, optional
            Specify whether process terms are summed to categories forcing and diffusion. False by default.

        Returns
        -------
        G : xarray.Dataset
            Dataset containing components of water mass transformations, possibly grouped as
            specified by the arguments.
        """

        # Extract default function args
        group_processes = kwargs.pop("group_processes", False)
        sum_components = kwargs.pop("sum_components", True)
        # call the base function
        transformations = self.transform_hlamdot_and_integrate(lambda_name, *args, **kwargs)

        # process this function arguments
        if group_processes:
            transformations = self._group_processes(transformations)
        if sum_components:
            transformations = self._sum_components(transformations, group_processes=group_processes)

        return transformations

    def isosurface_mean(self, *args, ti=None, tf=None, dl=0.1, **kwargs):
        """
        Mean transformation across lambda isosurface(s).

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'temperature', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        term : str, optional
            Specifies process term (e.g., 'boundary_forcing', 'vertical_diffusion', etc.). Use `processes()` to list all available terms.
        val : float or ndarray
            Value(s) of lambda for which isosurface(s) is/are defined
        ti : str
            Starting date. ti=None by default.
        tf : str
            End date. tf=None by default.
        dl : float
            Width of lamba bin (delta) for which isosurface(s) is/are defined.
        sum_components : boolean, optional
            Specify whether heat and salt tendencies are summed together (True) or kept separated (False). True by default.
        group_processes : boolean, optional
            Specify whether process terms are summed to categories forcing and diffusion. False by default.

        Returns
        -------
        local_transformations_mean : {xarray.DataArray, xarray.Dataset}
            Spatial field of mean transformation at a given (set of) lambda value(s).
            local_transformations_mean is xarray.DataSet when multiple terms are included (term=None) or sum_components=False.
        """

        if len(args) == 3:
            (lambda_name, term, val) = args
        elif len(args) == 2:
            (lambda_name, val) = args
            term = None
        else:
            warnings.warn(
                "isosurface_mean() requires arguments (lambda_name, term, val,...) or (lambda_name, val,...)"
            )
            return

        if lambda_name not in self.lambdas("density"):
            tendency = [k for k, v in self.lambdas_dict.items() if v[0] == lambda_name]
            if len(tendency) == 1:
                tendcode = self.terms_dict.get(tendency[0], None)
            else:
                warnings.warn("Tendency is not defined")
                return
        else:
            tendcode = lambda_name

        # Define bins based on val
        kwargs["bins"] = bin_define(np.min(val) - dl, np.max(val) + dl, dl)

        # Calculate spatiotemporal field of transformation
        local_transformations = self.map_transformations(lambda_name, term, **kwargs)
        # TODO: Preferred method should be ndays_standard if calendar type is 'noleap'. Thus, avoiding to load the full time array
        if (
            "calendar_type" in self.ds.time.attrs
            and self.ds.time.attrs["calendar_type"].lower() == "noleap"
        ):
            # Number of days in each month
            n_years = len(np.unique(local_transformations.time.dt.year))
            # Monthly data
            dm = np.diff(local_transformations.indexes["time"].month)
            udm = np.unique([m + 12 if m == -11 else m for m in dm])
            if np.array_equal(udm, [1]):
                ndays_standard = np.array(
                    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                )
                assert np.sum(ndays_standard) == 365
                dt = xr.DataArray(
                    ndays_standard[local_transformations.time.dt.month.values - 1],
                    coords=[self.ds.time],
                    dims=["time"],
                    name="days per month",
                )
            # Annual data
            dy = np.diff(local_transformations.indexes["time"].year)
            udy = np.unique(dy)
            if np.array_equal(udy, [1]):
                dt = xr.DataArray(
                    np.tile(365, n_years),
                    coords=[self.ds.time],
                    dims=["time"],
                    name="days per year",
                )
        elif "time_bounds" in self.ds:
            # Extract intervals (units are in ns)
            deltat = self.ds.time_bounds[:, 1].values - self.ds.time_bounds[:, 0].values
            # Convert intervals to days
            dt = xr.DataArray(
                deltat, coords=[self.ds.time], dims=["time"], name="days per month"
            ) / np.timedelta64(1, "D")
        elif "time_bnds" in self.ds:
            # Extract intervals (units are in ns)
            deltat = self.ds.time_bnds[:, 1].values - self.ds.time_bnds[:, 0].values
            # Convert intervals to days
            dt = xr.DataArray(
                deltat, coords=[self.ds.time], dims=["time"], name="days per month"
            ) / np.timedelta64(1, "D")
        else:
            # TODO: Create dt with ndays_standard but output warning that calendar_type is not specified.
            # warnings.warn('Unsupported calendar type')
            print("Unsupported calendar type", self.ds.time.attrs)
            return

        # Convert to dask array for lazy calculations
        dt = dt.chunk(1)
        local_transformations_mean = (
            local_transformations.sel({tendcode: val}, method="nearest").sel(time=slice(ti, tf))
            * dt.sel(time=slice(ti, tf))
        ).sum("time") / dt.sel(time=slice(ti, tf)).sum("time")
        return local_transformations_mean
