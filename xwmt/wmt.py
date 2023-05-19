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
    A class object with multiple methods to do full 3d watermass transformation analysis.
    """
    def __init__(
        self,
        grid,
        budgets_dict,
        teos10=True,
        cp=3992.0,
        rho_ref=1035.0,
        ):
        """
        Create a new watermass object from an input dataset.

        Parameters
        ----------
        grid: xgcm.Grid
            Contains information about ocean model grid coordinates, metrics, and data variables.
        budgets_dict: dict
            Nested dictionary containing information about lambda and tendency variable names.
            See `xwmt/conventions` for examples of how this dictionary should be structured.
        teos10 : boolean, optional
            Use Thermodynamic Equation Of Seawater - 2010 (TEOS-10). True by default.
        cp: float
            Value of specific heat capacity.
        rho_ref: float
            Value of reference potential density. Note: WaterMass is assumed to be Boussinesq.
        """
        
        super().__init__(
            grid,
            t_name=budgets_dict["heat"]["lambda"],
            s_name=budgets_dict["salt"]["lambda"],
            h_name=budgets_dict["mass"]["thickness"],
            teos10=teos10,
            cp=cp,
            rho_ref=rho_ref
        )
        
        self.component_dict = {
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
            if 'surface_mass_flux' in cdict:
                for density_var in ["heat", "salt"]:
                    self.budgets_dict[density_var]['surface_flux'] = {
                        **self.budgets_dict[density_var]['surface_flux'],
                        **{f"surface_mass_flux_{term}":v
                           for (term,v) in cdict['surface_mass_flux'].items()}
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
        """
        Return dictionary of desired lambdas, defaulting to all (temperature, salinity, and all densities).

        Parameters
        ----------
        lambda_name: str or list of str
            Lambda(s) of interest.

        Returns
        -------
        list
        """
        if lambda_name is None:
            return sum(self.lambdas_dict.values(), [])
        else:
            return self.lambdas_dict.get(lambda_name, None)

    def process_names(self, component, term):
        """
        Get a tuple containing the names of variables for the density component (temperature or salinity)
        and the 'process'-specific tendency corresponding to a general 'term' in the tendency equation.

        Parameters
        ----------
        lambda_name: str or list of str
            Lambda(s) of interest.

        Returns
        -------
        names : tuple
            `(component_name, process)`
        """
        if component == "heat":
            process = self.processes_heat_dict.get(term, None)
        elif component == "salt":
            process = self.processes_salt_dict.get(term, None)
        elif "surface_mass_flux" in term:
            process = self.processes_mass_dict.get(term, None)
        else:
            warnings.warn(f"Component {component} is not defined")
            return
        component_name = self.component_dict.get(component, None)
        return (component_name, process)

    def available_processes(self, available=True):
        """
        Get a list of all tendency processes that are both specified by `budgets_dict` and available in
        the dataset.

        Parameters
        ----------
        available: bool
            Default True. If False, include processes that are not available in the dataset.

        Returns
        -------
        names : tuple
            `(component_name, process)`
        """
        processes = (
            self.processes_heat_dict.keys() |
            self.processes_salt_dict.keys() |
            self.processes_mass_dict.keys()
        )
        if available:
            _processes = []
            for process in processes:
                p1 = self.processes_heat_dict.get(process, None)
                p2 = self.processes_salt_dict.get(process, None)
                p3 = self.processes_mass_dict.get(process, None)
                if (((p1 is None) or (p1 is not None and p1 in self.ds)) and
                    ((p2 is None) or (p2 is not None and p2 in self.ds)) and
                    ((p3 is None) or (p3 is not None and p3 in self.ds))
                ):
                    _processes.append(process)
            return _processes
        else:
            return processes

    def datadict(self, component, term):
        """
        Get a dictionary that organizes the variables and metadata necessary to evaluate tendency terms.

        Parameters
        ----------
        component: str
            Either "heat" or "salt".
        term: str
            Name of tendency term

        Returns
        -------
        ddict : dict
        """
        
        (component_name, process) = self.process_names(component, term)
        
        if process is None or process not in self.ds:
            return

        if component == "salt":
            # Multiply salt tendency by 1000 to convert to g/m^2/s
            tend_arr = self.ds[process] * 1000
        else:
            tend_arr = self.ds[process]
        scalar = self.ds[component_name]

        n_zcoords = len([
            c for c in self.grid.axes['Z'].coords.values()
            if c in self.ds[process].dims
        ])
        
        if term[:7]!="surface":
            return {
                "scalar": {"array": scalar},
                "tendency": {
                    "array": tend_arr,
                    "extensive": True,
                    "boundary": False
                },
            }
        
        elif term[:7]=="surface":
            z_coord = self.grid.axes['Z'].coords["center"]
            
            if "surface_mass_flux" in term:
                mass_flux = self.expand_surface_array_vertically(
                    self.ds[process],
                )
                tend_arr = xr.zeros_like(mass_flux)
                
            else:
                if n_zcoords == 0:
                    tend_arr = self.expand_surface_array_vertically(tend_arr)
                mass_flux = xr.zeros_like(tend_arr)
            
            if component == "heat":
                if n_zcoords == 0:
                    scalar = (
                        self.ds[self.budgets_dict["heat"]["surface_lambda"]]
                        .expand_dims({z_coord:self.ds[z_coord]})
                        .rename(self.budgets_dict["heat"]["lambda"])
                    )
                
                # Need to multiply mass flux by cp to convert
                # to energy flux (in W/m^2/degC)
                mass_flux *= self.cp
                scalar_in_mass = self.expand_surface_array_vertically(
                    self.ds[self.budgets_dict["heat"]["surface_lambda"]],
                )
            elif component == "salt":
                if n_zcoords == 0:
                    scalar = (
                        self.ds[self.budgets_dict["salt"]["surface_lambda"]]
                        .expand_dims({z_coord:self.ds[z_coord]})
                        .rename(self.budgets_dict["salt"]["lambda"])
                    )
                
                scalar_in_mass = self.expand_surface_array_vertically(
                    xr.zeros_like(self.ds[self.budgets_dict["salt"]["surface_lambda"]]),
                )
            else:
                raise ValueError(f"process {process} not yet supported.")
                
            return {
                "scalar": {
                    "array": scalar
                },
                "tendency": {
                    "array": tend_arr,
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
        Get density tendency 'term' from underlying heat and salt tendencies. 

        Parameters
        ----------
        term: str
            Name of tendency term

        Returns
        -------
        rho_tend_heat, rho_tend_salt
            The two distinct components contributing to the overall density tendency.
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
        
        Parameters
        ----------
        lambda_name : str
            Specifies lambda
        term : str
            Specifies process term
            
        Returns
        ----------
        hlamdot, lam : xr.DataArray, xr.DataArray
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
            for idx, tend in enumerate(self.component_dict.keys()):
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
        Lazily compute extensive tendencies and transform them into lambda space
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
            for tend in self.component_dict.keys():
                (component_name, process) = self.process_names(tend, term)
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
                        ).rename(f"{term}_{tend}")
                    )
                        
            hlamdot_transformed = xr.merge(hlamdot_transformed)
        else:
            (component_name, process) = self.process_names(
                "salt" if lambda_name == "salinity" else "heat", term
            )
            hlamdot_transformed = (
                self.grid.transform(
                    hlamdot, "Z", target=bins, target_data=lam_i, method="conservative"
                )
                / np.diff(bins)
            ).rename(f"{term}")
        return hlamdot_transformed

    def transform_hlamdot_and_integrate(self, lambda_name, term=None, bins=None):
        """
        Lazily compute extensive tendencies, transform them into lambda space
        along the vertical ("Z") dimension, and integrate along the
        horizontal dimensions ("X", "Y").
        """

        # If term is not given, use all available process terms
        if term is None:
            wmts = []
            for term in self.available_processes():
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
    def _sum_terms(self, ds_terms, newterm, terms):
        das = []
        for term in terms:
            if term in ds_terms:
                das.append(ds_terms[term])
        if len(das):
            ds_terms[newterm] = sum(das)

    def _group_processes(self, hlamdot):
        if hlamdot is None:
            return
        for suffix in ["", "_heat", "_salt"]:
            self._sum_terms(
                hlamdot,
                f"external_forcing{suffix}",
                [
                    f"boundary_forcing{suffix}",
                    f"frazil_ice{suffix}",
                    f"geothermal{suffix}",
                ],
            )
            self._sum_terms(
                hlamdot,
                f"diffusion{suffix}",
                [
                    f"vertical_diffusion{suffix}",
                    f"neutral_diffusion{suffix}"
                ]
            )
            self._sum_terms(
                hlamdot,
                f"advection{suffix}",
                [
                    f"horizontal_advection{suffix}",
                    f"vertical_advection{suffix}"
                ]
            )
            self._sum_terms(
                hlamdot,
                f"diabatic_forcing{suffix}",
                [
                    f"external_forcing{suffix}",
                    f"diffusion{suffix}"
                ]
            )
            self._sum_terms(
                hlamdot,
                f"total_tendency{suffix}",
                [
                    f"advection{suffix}",
                    f"diabatic_forcing{suffix}"
                ]
            )
        return hlamdot

    def _sum_components(self, hlamdot, group_processes = False):
        if hlamdot is None:
            return
        
        for proc in self.available_processes():
            proc_list = [
                proc+f"_{component}"
                for component in ["heat", "salt"]
            ]
            self._sum_terms(
                hlamdot,
                proc,
                proc_list
            )
        if group_processes:
            for proc in [
                "external_forcing",
                "diffusion",
                "advection",
                "diabatic_forcing",
                "total_tendency"
            ]:
                self._sum_terms(
                    hlamdot,
                    proc,
                    [f"{proc}_{component}" for component in ["heat", "salt"]]
                )
        return hlamdot

    def map_transformations(self, lambda_name, term=None, sum_components=True, group_processes=False, **kwargs):
        """
        Wrapper function for transform_hlamdot() to group terms based
        on tendency terms (heat, salt) and processes.
        """
        # If term is not given, use all available process terms
        if term is None:
            _local_transformations = []
            for term in self.available_processes(False):
                _local_transformation = self.map_transformations(lambda_name, term, sum_components=False, group_processes=False, **kwargs)
                if _local_transformation is not None:
                    _local_transformations.append(_local_transformation)
            local_transformations = xr.merge(_local_transformations)
        else:
            # If term is given
            local_transformations = self.transform_hlamdot(lambda_name, term, **kwargs)
            if isinstance(local_transformations, xr.DataArray):
                local_transformations = local_transformations.to_dataset()

        if group_processes:
            local_transformations = self._group_processes(local_transformations)
        if sum_components:
            local_transformations = self._sum_components(local_transformations, group_processes=group_processes)

        return local_transformations

    def integrate_transformations(self, lambda_name, *args, **kwargs):
        """
        Lazily compute horizontally-integrated water mass transformations.

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
        transformations : xarray.Dataset
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