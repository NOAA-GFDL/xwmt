import numpy as np
import xarray as xr
from xhistogram.xarray import histogram
import gsw
import warnings

from xwmt.compute import (
    Jlmass_from_Qm_lm_l,
    calc_hldot_tendency,
    expand_surface_to_3D,
    get_xgcm_grid_vertical,
    hldot_from_Jl,
    hldot_from_Ldot_hldotmass,
    lbin_define,
    lbin_percentile,
)

class wmt():
    '''
    A class object with multiple functions to do full 3d watermass transformation analysis.
    '''

    def __init__(self, ds, Cp=3992.0, rho=1035.0, alpha=None, beta=None, teos10=True):
        '''
        Create a new watermass transformation object from an input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant tendencies and/or surface fluxes along with grid information.
        Cp : float, optional
            Specify value for the specific heat capacity (in J/kg/K). Cp=3992.0 by default.
        rho : float, optional
            Specify value for the reference seawater density (in kg/m^3). rho=1035.0 by default.
        alpha : float, optional
            Specify value for the thermal expansion coefficient (in 1/K). alpha=None by default.
            If alpha is not given (i.e., alpha=None), it is derived from salinty and temperature fields using `gsw_alpha`.
        beta : float, optional
            Specify value for the haline contraction coefficient (in kg/g). beta=None by default.
            If beta is not given (i.e., beta=None), it is derived from salinty and temperature fields using `gsw_beta`.
        teos10 : boolean, optional
            Use Thermodynamic Equation Of Seawater - 2010 (TEOS-10). True by default.
        '''

        self.ds = ds.copy()
        self.xgrid = get_xgcm_grid_vertical(self.ds, periodic=False)
        self.Cp = Cp
        self.rho = rho
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        self.teos10 = teos10

    # Set of terms for (1) heat and (2) salt fluxes
    # Use processes as default, fluxes when surface=True
    terms_dict = {
        'heat': 'thetao',
        'salt': 'so'
    }

    processes_heat_dict = {
        'boundary forcing': 'boundary_forcing_heat_tendency',
        'vertical diffusion': 'opottempdiff',
        'neutral diffusion': 'opottemppmdiff',
        'frazil ice': 'frazil_heat_tendency',
        'geothermal': 'internal_heat_heat_tendency'
    }

    processes_salt_dict = {
        'boundary forcing': 'boundary_forcing_salt_tendency',
        'vertical diffusion': 'osaltdiff',
        'neutral diffusion': 'osaltpmdiff',
        'frazil ice': None,
        'geothermal': None
    }

    lambdas_dict = {
        'heat': ['theta'],
        'salt': ['salt'],
        'density': ['sigma0','sigma1','sigma2','sigma3','sigma4']
    }

    def lambdas(self, lstr=None):
        if lstr is None:
            return sum(self.lambdas_dict.values(), [])
        else:
            return self.lambdas_dict.get(lstr, None)

    # Helper function to get variable name for given process term
    def process(self, tendency, term):
        # Organize by scalar and tendency
        if tendency == 'heat':
            termcode = self.processes_heat_dict.get(term, None)
        elif tendency == 'salt':
            termcode = self.processes_salt_dict.get(term, None)
        else:
            warnings.warn('Tendency is not defined')
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
                if ((p1 is None) or (p1 is not None and p1 in self.ds)) and \
                   ((p2 is None) or (p2 is not None and p2 in self.ds)):
                    _processes.append(process)
            return _processes
        else:
            return processes


    def dd(self, tendency, term):
        (tendcode, termcode) = self.process(tendency, term)
        # tendcode: tendency form (heat or salt)
        # termcode: process term (e.g., boundary forcing)
        if termcode is None or termcode not in self.ds:
            return

        if tendency == 'salt':
            # Multiply salt tendency by 1000 to convert to g/m^2/s 
            tend_arr = self.ds[termcode]*1000
        else:
            tend_arr = self.ds[termcode]

        if term == 'boundary forcing':
            if termcode == 'boundary_forcing_heat_tendency':
                # Need to multiply mass flux by Cp to convert to energy flux (convert to W/m^2/degC)
                flux = expand_surface_to_3D(self.ds['wfo'], self.ds['lev_outer'])*self.Cp
                scalar_in_mass = expand_surface_to_3D(self.ds['tos'], self.ds['lev_outer'])
            else:
                flux = expand_surface_to_3D(self.ds['wfo'], self.ds['lev_outer'])
                scalar_in_mass = expand_surface_to_3D(xr.zeros_like(self.ds['sos']), self.ds['lev_outer'])
            return {
                'scalar':  {'array': self.ds[tendcode]}, 
                'tendency':{'array': tend_arr, 'extensive': True, 'boundary': True},
                'boundary':{'flux': flux, 'mass': True, 'scalar_in_mass': scalar_in_mass}
            }
        else:
            return {
                'scalar':  {'array': self.ds[tendcode]},
                'tendency':{'array': tend_arr, 'extensive': True, 'boundary': False}
            }

    def get_density(self, density_str=None):

        # Variables needed to calculate alpha, beta and density
        if ('alpha' not in vars(self) or 'beta' not in vars(self) or self.teos10) and 'p' not in vars(self):
            self.p = xr.apply_ufunc(gsw.p_from_z, -self.ds['lev'],
                           self.ds['lat'], 0, 0, dask='parallelized')
        if self.teos10 and 'sa' not in vars(self):
            self.sa = xr.apply_ufunc(gsw.SA_from_SP, self.ds['so'], self.p,
                            self.ds['lon'], self.ds['lat'], dask='parallelized')
        if self.teos10 and 'ct' not in vars(self):
            self.ct = xr.apply_ufunc(gsw.CT_from_t, self.sa, self.ds['thetao'], self.p, dask='parallelized')
        if not self.teos10 and ('sa' not in vars(self) or 'ct' not in vars(self)):
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

    def rho_tend(self, term):
        '''
        Calculate the tendency of the locally-referenced potential density.
        '''

        if 'alpha' in vars(self) and 'beta' in vars(self):
            alpha, beta = self.alpha, self.beta
        else:
            (alpha,beta,_) = self.get_density()

        # Either heat or salt tendency/flux may not be used
        rho_tend_heat, rho_tend_salt = None, None

        dd = self.dd('heat', term)
        if dd is not None:
            heat_tend = calc_hldot_tendency(self.xgrid, self.dd('heat', term))
            # Density tendency due to heat flux (kg/s/m^2)
            rho_tend_heat = -(alpha/self.Cp)*heat_tend

        dd = self.dd('salt', term)
        if dd is not None:
            salt_tend = calc_hldot_tendency(self.xgrid, self.dd('salt', term))
            # Density tendency due to salt/salinity (kg/s/m^2)
            rho_tend_salt = beta*salt_tend

        return rho_tend_heat, rho_tend_salt


    def calc_Fl(self, lstr, term):
        '''
        Get transformation rate (* m/s) and corresponding scalar field of lambda
        lstr: str
            Specifies lambda
        term: str
            Specifies process term
        '''

        # Get F from tendency of heat (in W/m^2), lambda = theta
        if lstr == 'theta':
            dd = self.dd('heat', term)
            if dd is not None:
                # Transformation rate (degC m/s)
                F = calc_hldot_tendency(self.xgrid, dd)/(self.rho*self.Cp)
                # Scalar field (degC)
                l = dd['scalar']['array']
                return F, l

        # Get F from tendency of salt (in g/s/m^2), lambda = salt
        elif lstr == 'salt':
            dd = self.dd('salt', term)
            if dd is not None:
                # Transformation rate (g/kg m/s)
                F = calc_hldot_tendency(self.xgrid, dd)/self.rho
                # Scalar field (salinity units, psu, g/kg)
                # TODO: Accurately define salinity field 
                l = dd['scalar']['array']
                return F, l

        # Get F from tendencies of density (in kg/s/m^2), lambda = density
        # Here we want to output 2 transformation rates:
        # (1) transformation due to heat tend, (2) transformation due to salt tend  
        elif lstr in self.lambdas('density'):
            F = {}
            rhos = self.rho_tend(term)
            for idx, tend in enumerate(self.terms_dict.keys()):
                # Transformation rate (kg/m^3 m/s)
                F[tend] = rhos[idx]
            # Scalar field (kg/m^3)
            l = self.get_density(lstr)[2]
            return F, l

        return (None, None)


    def calc_F_transformed(self, lstr, term, bins=None):
        '''
        Transform to lambda space
        '''

        F,l = self.calc_Fl(lstr, term)
        if F is None:
            return

        if bins is None:
            bins = lbin_percentile(l) # automatically find the right range based on the distribution in l

        # Interpolate lambda to the cell interfaces
        l_i = self.xgrid.interp(l,'Z',boundary='extrapolate').chunk({'lev_outer':-1}).rename(l.name)

        if lstr in self.lambdas('density'):
            F_transformed = []
            for tend in self.terms_dict.keys():
                (tendcode, termcode) = self.process(tend,term)
                if F[tend] is not None:
                    F_transformed.append( (self.xgrid.transform(F[tend], 'Z', target=bins, target_data=l_i,
                                                      method='conservative')/np.diff(bins)).rename(termcode) )
            F_transformed = xr.merge(F_transformed)
        else:
            (tendcode, termcode) = self.process('salt' if lstr == 'salt' else 'heat',term)
            F_transformed = (self.xgrid.transform(F, 'Z', target=bins, target_data=l_i,
                                            method='conservative')/np.diff(bins)).rename(termcode)
        return F_transformed


    def calc_G(self, lstr, term=None, method='xhistogram', bins=None):
        '''
        Water mass transformation (G)
        '''

        # If term is not given, use all available process terms
        if term is None:
            Gs = []
            for term in self.processes(False):
                _G = self.calc_G(lstr, term, method, bins)
                if _G is not None:
                    Gs.append(_G)
            return xr.merge(Gs)

        if method == 'xhistogram' and lstr in self.lambdas('density'):
            F,l = self.calc_Fl(lstr, term)
            if bins is None and l is not None:
                bins = lbin_percentile(l) # automatically find the right range based on the distribution in l
            G = []
            for tend in self.terms_dict.keys():
                (tendcode, termcode) = self.process(tend,term)
                if termcode is not None and F[tend] is not None:
                    _G = (histogram(l.where(~np.isnan(F[tend])), bins = [bins], dim = ['x','y','lev'], 
                                     weights = (F[tend]*self.ds['areacello'])\
                                     .where(~np.isnan(F[tend])))/np.diff(bins))\
                    .rename({l.name+'_bin': l.name}).rename(termcode)
                    G.append(_G)
            return xr.merge(G)
        elif method == 'xhistogram':
            F,l = self.calc_Fl(lstr, term)
            if bins is None and l is not None:
                bins = lbin_percentile(l) # automatically find the right range based on the distribution in l
            if F is not None and l is not None:
                (tendcode, termcode) = self.process('salt' if lstr == 'salt' else 'heat',term)
                G = (histogram(l.where(~np.isnan(F)), bins = [bins], dim = ['x','y','lev'], 
                           weights = (F*self.ds['areacello']).where(~np.isnan(F)))/np.diff(bins))\
                .rename({l.name+'_bin': l.name}).rename(termcode)
                return G
        elif method == 'xgcm':
            F_transformed = self.calc_F_transformed(lstr,term,bins=bins)
            if F_transformed is not None and len(F_transformed):
                G = (F_transformed*self.ds['areacello']).sum(['x','y'])
                # rename dataarray only (not dataset)
                if isinstance(G, xr.DataArray):
                    return G.rename(F_transformed.name)
                return G
            return F_transformed

    ### Helper function to groups terms based on tendencies (group_tend) and processes (group_process)
    # Calculate the sum of grouped terms
    def _sum(self, ds, newterm, terms):
        das = []
        for term in terms:
            if term in ds:
                das.append(ds[term])
                del ds[term]
        if len(das):
            ds[newterm] = sum(das)

    # group_process == True and group_tend == False
    def _group_process(self, F):
        if F is None:
            return
        self._sum(F, 'forcing_heat',   ['boundary_forcing_heat_tendency', 'frazil_heat_tendency', 'internal_heat_heat_tendency'])
        self._sum(F, 'diffusion_heat', ['opottempdiff', 'opottemppmdiff'])
        self._sum(F, 'forcing_salt',   ['boundary_forcing_salt_tendency'])
        self._sum(F, 'diffusion_salt', ['osaltdiff', 'osaltpmdiff'])
        return F

    # group_process == True and group_tend == True
    def _group_process_tend(self, F):
        if F is None:
            return
        self._sum(F, 'forcing',  ['boundary_forcing_heat_tendency','frazil_heat_tendency','internal_heat_heat_tendency','boundary_forcing_salt_tendency'])
        self._sum(F, 'diffusion',['opottempdiff', 'opottemppmdiff', 'osaltdiff', 'osaltpmdiff'])
        return F

    # group_process == False and group_tend == True
    def _group_tend(self, F):
        if F is None:
            return
        self._sum(F, 'boundary_forcing',   ['boundary_forcing_heat_tendency', 'boundary_forcing_salt_tendency'])
        self._sum(F, 'vertical_diffusion', ['opottempdiff', 'osaltdiff'])
        self._sum(F, 'neutral_diffusion',  ['opottemppmdiff', 'osaltpmdiff'])
        self._sum(F, 'frazil_ice',         ['frazil_heat_tendency'])
        self._sum(F, 'geothermal',         ['internal_heat_heat_tendency'])
        return F

    def F(self, lstr, term=None, group_tend=True, group_process=False, **kwargs):
        '''
        Wrapper function for calc_F_transformed() to group terms based on tendency terms (heat, salt) and processes.
        '''

        # If term is not given, use all available process terms
        if term is None:
            Fs = []
            for term in self.processes(False):
                _F = self.F(lstr, term, group_tend=False, group_process=False, **kwargs)
                if _F is not None:
                    Fs.append(_F)
            F = xr.merge(Fs)
        else:
            # If term is given
            F = self.calc_F_transformed(lstr, term, **kwargs)
            if isinstance(F,xr.DataArray):
                F = F.to_dataset()

        if group_process == True and group_tend == False:
            F = self._group_process(F)
        elif group_process == True and group_tend == True:
            F = self._group_process_tend(F)
        elif group_process == False and group_tend == True:
            F = self._group_tend(F)

        if isinstance(F,xr.Dataset) and len(F) == 1:
            return F[list(F.data_vars)[0]]
        else:
            return F

    def G(self, lstr, *args, **kwargs):
        '''
        Water mass transformation (G)
        
        Parameters
        ----------
        lstr : str
            Specifies lambda (e.g., 'theta', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas. 
        term : str, optional
            Specifies process term (e.g., 'boundary forcing', 'vertical diffusion', etc.). Use `processes()` to list all available terms.
        method : str {'xhistogram' (default), 'xgcm'}
            The calculation can be either done with xhistogram (default) or the xgcm `transform`. If not specified, default will be used.
        bins : array like, optional
            np.array with lambda values specifying the edges for each bin. If not specidied, array will be automatically derived from
            the scalar field of lambda (e.g., temperature).
        group_tend : boolean, optional
            Specify whether heat and salt tendencies are summed together (True) or kept separated (False). True by default.
        group_process : boolean, optional
            Specify whether process terms are summed to categories forcing and diffusion. False by default.

        Returns
        -------
        G : {xarray.DataArray, xarray.Dataset}
            The water mass transformation along lamba for each time. G is xarray.DataArray when term is specified and group_tend=True.
            G is xarray.DataSet when multiple terms are included (term=None) or group_tend=False.
        '''

        # Extract default function args
        group_process = kwargs.pop("group_process",False)
        group_tend = kwargs.pop("group_tend",True)
        # call the base function
        G = self.calc_G(lstr, *args, **kwargs)

        # process this function arguments
        if group_process == True and group_tend == False:
            G = self._group_process(G)
        elif group_process == True and group_tend == True:
            G = self._group_process_tend(G)
        elif group_process == False and group_tend == True:
            G = self._group_tend(G)

        if isinstance(G,xr.Dataset) and len(G) == 1:
            return G[list(G.data_vars)[0]]
        else:
            return G

    def isosurface_mean(self, *args, ti=None, tf=None, dl=0.1, **kwargs):
        '''
        Mean transformation across lambda isosurface(s).
        
        Parameters
        ----------
        lstr : str
            Specifies lambda (e.g., 'theta', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        term : str, optional
            Specifies process term (e.g., 'boundary forcing', 'vertical diffusion', etc.). Use `processes()` to list all available terms.
        val : float or ndarray
            Value(s) of lambda for which isosurface(s) is/are defined
        ti : str
            Starting date. ti=None by default.
        tf : str
            End date. tf=None by default.
        dl : float
            Width of lamba bin (delta) for which isosurface(s) is/are defined. 
        method : str {'xhistogram' (default), 'xgcm'}
            The calculation can be either done with xhistogram (default) or the xgcm `transform`. If not specified, default will be used.
        group_tend : boolean, optional
            Specify whether heat and salt tendencies are summed together (True) or kept separated (False). True by default.
        group_process : boolean, optional
            Specify whether process terms are summed to categories forcing and diffusion. False by default.

        Returns
        -------
        F_mean : {xarray.DataArray, xarray.Dataset}
            Spatial field of mean transformation at a given (set of) lambda value(s). F_mean is xarray.DataArray when term is specified and group_tend=True.
            F_mean is xarray.DataSet when multiple terms are included (term=None) or group_tend=False.
        '''

        if len(args) == 3:
            (lstr, term, val) = args
        elif len(args) == 2:
            (lstr, val) = args
            term = None
        else:
            warnings.warn('isosurface_mean() requires arguments (lstr, term, val,...) or (lstr, val,...)')
            return

        if lstr not in self.lambdas('density'):
            tendency = [k for k,v in self.lambdas_dict.items() if v[0]==lstr]
            if len(tendency) == 1:
                tendcode = self.terms_dict.get(tendency[0], None)
            else:
                warnings.warn('Tendency is not defined')
                return
        else:
            tendcode = lstr

        # Define bins based on val
        kwargs['bins'] = lbin_define(np.min(val)-dl,np.max(val)+dl,dl)

        # Calculate spatiotemporal field of transformation
        F = self.F(lstr, term, **kwargs)
        # TODO: Preferred method should be ndays_standard if calendar type is 'noleap'. Thus, avoiding to load the full time array
        if 'calendar_type' in self.ds.time.attrs and self.ds.time.attrs['calendar_type'].lower() == 'noleap':
            # Number of days in each month
            n_years = len(np.unique(F.time.dt.year))
            # Monthly data
            dm = np.diff(F.indexes['time'].month)
            udm = np.unique([m+12 if m==-11 else m for m in dm])
            if np.array_equal(udm,[1]):
                ndays_standard = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
                assert(np.sum(ndays_standard) == 365)
                dt = xr.DataArray(ndays_standard[F.time.dt.month.values-1], coords = [self.ds.time], dims = ['time'], name = 'days per month')
            # Annual data
            dy = np.diff(F.indexes['time'].year)
            udy = np.unique(dy)
            if np.array_equal(udy,[1]):
                dt = xr.DataArray(np.tile(365, n_years), coords = [self.ds.time], dims = ['time'], name = 'days per year')
        elif 'time_bounds' in self.ds:
            # Extract intervals (units are in ns)
            deltat = self.ds.time_bounds[:,1].values - self.ds.time_bounds[:,0].values
            # Convert intervals to days
            dt = xr.DataArray(deltat, coords = [self.ds.time],
                                  dims = ['time'], name = 'days per month')/np.timedelta64(1,'D')
        elif 'time_bnds' in self.ds:
            # Extract intervals (units are in ns)
            deltat = self.ds.time_bnds[:,1].values - self.ds.time_bnds[:,0].values
            # Convert intervals to days
            dt = xr.DataArray(deltat, coords = [self.ds.time],
                                  dims = ['time'], name = 'days per month')/np.timedelta64(1,'D')
        else:
            # TODO: Create dt with ndays_standard but output warning that calendar_type is not specified.
            #warnings.warn('Unsupported calendar type')
            print ('Unsupported calendar type', self.ds.time.attrs)
            return

        # Convert to dask array for lazy calculations
        dt = dt.chunk(1)
        F_mean = (F.sel({tendcode: val}, method='nearest').sel(time=slice(ti,tf))\
                    *dt.sel(time=slice(ti,tf))).sum('time')/dt.sel(time=slice(ti,tf)).sum('time')
        return F_mean
