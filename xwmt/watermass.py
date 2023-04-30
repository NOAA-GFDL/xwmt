import numpy as np
import xarray as xr
from xhistogram.xarray import histogram
import gsw
import warnings

from xwmt.compute import (
    Jlammass_from_Qm_lm_l,
    calc_hlamdot_tendency,
    expand_surface_to_3d,
    get_xgcm_grid_vertical,
    hlamdot_from_Jlam,
    hlamdot_from_Ldot_hlamdotmass,
    bin_define,
    bin_percentile,
)

class WaterMass:
    """
    A class object with multiple methods to define a water mass from gridded model data.
    """

    def __init__(self, ds, grid):
        """
        Create a new water mass object from an input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant tendencies and/or surface fluxes along with grid information.
        cp : float, optional
            Specify value for the specific heat capacity (in J/kg/K). cp=3992.0 by default.
        rho_ref : float, optional
            Specify value for the reference seawater density (in kg/m^3). rho_ref=1035.0 by default.
        alpha : float, optional
            Specify value for the thermal expansion coefficient (in 1/K). alpha=None by default.
            If alpha is not given (i.e., alpha=None), it is derived from salinty and temperature fields using `gsw_alpha`.
        beta : float, optional
            Specify value for the haline contraction coefficient (in kg/g). beta=None by default.
            If beta is not given (i.e., beta=None), it is derived from salinty and temperature fields using `gsw_beta`.
        teos10 : boolean, optional
            Use Thermodynamic Equation Of Seawater - 2010 (TEOS-10). True by default.
        """

        self.ds = ds.copy()
        self.grid = grid

    def transform_variable(self, var, lam, bins):
        lam_i = (
            self.grid.interp(lam, "Z", boundary="extend")
            .chunk({self.grid.axes['Z'].coords['outer']: -1})
            .rename(lam.name)
        )
        return self.grid.transform(var, "Z", target=bins, target_data=lam_i, method="conservative")
    
    def cumint_vertical(self, var):
        pass
    
class WaterMassTransformations(WaterMass):
    
    def __init__(self, ds, grid, budgets_dict):
        super().__init__(ds, grid)
        self.budgets_dict = budgets_dict
    
    def extensive_tracer_budget(self, tracer):
        if not hasattr(self, "layer_tendencies"):
            self.layer_tendencies = {}
            
        self.layer_tendencies[tracer] = {}
        for side, terms in self.budgets_dict[tracer].items():
            if side not in ['lhs', 'rhs']: continue
            self.layer_tendencies[tracer][side] = {}
            for term,varname in terms.items():
                if varname not in self.ds: continue
                var = self.ds[varname].copy()
                
                
                self.layer_tendencies[tracer][side]['term'] = var
                
        return self.layer_tendencies
    
    def map_transformations(self, tracer, bins):
        if hasattr(self, "layer_tendencies"):
            if tracer in self.layer_tendencies:
                pass
            else:
                self.extensive_tracer_budget(tracer)
        else:
            self.extensive_tracer_budget(tracer)
        if not hasattr(self, "local_transformations"):
            self.local_transformations = {}

        self.local_transformations[tracer] = {}
        for side, terms in self.layer_tendencies[tracer].items():
            self.local_transformations[tracer][side] = {
                term: (
                    self.transform_variable(
                        var,
                        self.ds[self.budgets_dict[tracer]['lambda']],
                        bins
                    )/np.diff(bins)
                )
                for term,var in terms.items()
            }
        return self.local_transformations
    
    def integrate_transformation_rates(self, tracer, bins):
        if hasattr(self, "local_transformations"):
            if tracer in self.local_transformations:
                pass
            else:
                self.map_transformations(tracer, bins)
        else:
            self.map_transformations(tracer, bins)
        if not hasattr(self, "transformation_rates"):
            self.transformation_rates = {}
            
        self.transformation_rates[tracer] = {}
        for side, terms in self.local_transformations[tracer].items():
            self.transformation_rates[tracer][side] = {
                term: var.sum([
                    self.grid.axes['X'].coords['center'],
                    self.grid.axes['Y'].coords['center']
                ])
                for term,var in terms.items()
            }
        return self.transformation_rates
    
    def check_budget_closes(self):
        pass
