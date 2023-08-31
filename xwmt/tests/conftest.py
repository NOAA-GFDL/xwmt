import pytest
import os
import numpy as np
import xarray as xr
import xgcm
import xwmt
import urllib.request

fname = 'xwmb_test_data_Baltic_3d.20230830.nc'
ftp_path = 'ftp://ftp.gfdl.noaa.gov/perm/John.Krasting/xwmt/'
if not os.path.isfile(fname):
    print(f'Downloading test dataset from {ftp_path}{fname}')
    urllib.request.urlretrieve(f'{ftp_path}{fname}', fname)

class Helpers:

    def __init__(self):
        pass

    def idealized_transformations(self, extensive_tendency, lam_profile, Nz=1e3, Nlam=8):
        bins = np.linspace(0., 1., int(Nlam)+1)
        
        ds = xr.Dataset()
        ds = ds.assign_coords({
            'z_i': xr.DataArray(np.linspace(0., 1., int(Nz)+1), dims=("z_i",)),
            'z_l': xr.DataArray(np.linspace(1. /Nz, 1. - 1. /Nz, int(Nz)), dims=("z_l",)),
        })
        ds = ds.assign_coords({'dz': xr.DataArray(np.diff(ds.z_i.values), dims=("z_l",))})
        extensive_tendency_method = getattr(self, extensive_tendency)
        ds['tendency_name'] = xr.DataArray(extensive_tendency_method(ds.z_i.values), coords=(ds.z_l,))*ds.dz
        lam_profile_method = getattr(self, lam_profile)
        ds['temperature'] = xr.DataArray(lam_profile_method(ds.z_l.values), coords=(ds.z_l,))

        # expand to horizontal dimension and add grid metrics
        ds = ds.expand_dims(dim=('x', 'y')).assign_coords({'x':xr.DataArray([1.], dims=('x',)), 'y':xr.DataArray([1.], dims=('y',))})
        ds = ds.assign_coords({'rA': xr.DataArray([[1.]], dims=('x','y',))})

        metrics = {
            ('X', 'Y'): ['rA'] # Areas
        }
        coords = {
            'X': {'center': 'x',},
            'Y': {'center': 'y',},
            'Z': {'center': 'z_l', 'outer': 'z_i'},
        }
        grid = xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=False, autoparse_metadata=False)
        
        budget_dict = {
            'mass': {
                'lambda': None,
                'rhs': {},
                'lhs': {}
            },
            'heat': {
                'lambda': 'temperature',
                'lhs': {'tendency':'tendency_name'},
                'rhs': {}
            },
            'salt': {
                'lambda': None,
                'rhs': {},
                'lhs': {}
            }
        }

        wmt = xwmt.WaterMassTransformations(grid, budget_dict, cp=1., rho_ref=1., method="xgcm")
        T = wmt.integrate_transformations("heat", bins=bins, sum_components=False)
        T = T.assign_coords({'temperature_i': xr.DataArray(bins, dims=("temperature_i",))})
        return T
    
    def mean_absolute_relative_errors(self, wmt_xwmt, wmt_local_exact, wmt_layer_exact):
        def absolute_relative_errors(wmt, wmt_ref):
            return np.abs((wmt - wmt_ref)/wmt_ref).where(np.abs(wmt_ref)>1.e-5).mean(skipna=True).values
        wmt_local_exact_method = getattr(self, wmt_local_exact)
        wmt_layer_exact_method = getattr(self, wmt_layer_exact)
        return (
            absolute_relative_errors(
                wmt_xwmt['tendency'],
                wmt_layer_exact_method(wmt_xwmt.temperature_i.values)
            ),
            absolute_relative_errors(
                wmt_xwmt['tendency'],
                wmt_local_exact_method(wmt_xwmt.temperature.values)
            )
        )

    # Extensive (layer-integrated) analytical tendency profiles
    def diffusive_extensive_tendency(self, z_i):
        def f(z):
            return -np.cos(2*np.pi*z)/(2*np.pi)
        return np.diff(f(z_i))/np.diff(z_i)

    def constant_extensive_tendency(self, z_i):
        return np.diff(z_i)/np.diff(z_i)

    def constant_plus_diffusion_tendency(self, z_i):
        return self.diffusive_extensive_tendency(z_i) + self.constant_extensive_tendency(z_i)

    def differential_heating_layer(self, z_i):
        sign = 2*np.float64(z_i>0.5)-1
        out = np.diff(sign*z_i)/np.diff(z_i)
        out[(z_i[:-1]<=0.5)&(0.5<=z_i[1:])] = 0.
        return out

    # Stratification profiles
    def lam_const_dlamdz(self, z):
        return z

    def lam_linear_dlamdz(self, z):
        return z**2

    def lam_overturning_dlamdz(self, z):
        return 1 - (2*z - 1)**2

    def lam_vanishing_dlamdz(self, z):
        sign = 2*np.float64(z>0.5)-1
        return (1 + sign*(2*z - 1)**2)/2.

    # Analytical point-wise water mass transformations
    def constant_plus_diffusion_local_wmt_dlamdz_constant(self, lam):
        return np.sin(2*np.pi*lam) + 1.

    def constant_plus_diffusion_local_wmt_dlamdz_linear(self, lam):
        return (np.sin(2*np.pi*np.sqrt(lam)) + 1.)/(2*np.sqrt(lam))

    def constant_plus_diffusion_local_wmt_dlamdz_overturning(self, lam):
        return 1/(2*np.sqrt(1-lam))

    def differential_heating_local_wmt_dlamdz_vanishing(self, lam):
        sign = 2*np.float64(lam>0.5)-1
        return sign/(2*np.sqrt(sign*(2*lam-1)))

    # Analytical layer-averaged water mass transformations
    def constant_plus_diffusion_layer_wmt_dlamdz_constant(self, lam_bins):
        def f(lam):
            return -np.cos(2*np.pi*lam)/(2*np.pi) + lam
        return np.diff(f(lam_bins))/np.diff(lam_bins)

    def constant_plus_diffusion_layer_wmt_dlamdz_linear(self, lam_bins):
        def f(lam):
            return -np.cos(2*np.pi*np.sqrt(lam))/(2*np.pi) + np.sqrt(lam)
        return np.diff(f(lam_bins))/np.diff(lam_bins)

    def constant_plus_diffusion_layer_wmt_dlamdz_overturning(self, lam_bins):
        def f(lam):
            return 1 - np.sqrt(1 - lam)
        return np.diff(f(lam_bins))/np.diff(lam_bins)

    def differential_heating_layer_wmt_dlamdz_vanishing(self, lam_bins):
        def f(lam):
            sign = 2*np.float64(lam>0.5)-1
            u = 1 + sign*np.sqrt(sign*(2*lam-1))
            return sign*0.5*u
        out = np.diff(f(lam_bins))/np.diff(lam_bins)
        out[(lam_bins[:-1]<=0.5)&(0.5<=lam_bins[1:])] = np.nan
        return out

    exps = {
        "extensive_tendency": [
            "constant_plus_diffusion_tendency",
            "constant_plus_diffusion_tendency",
            "constant_plus_diffusion_tendency",
            "differential_heating_layer"
        ],
        "lam_profile": [
            "lam_const_dlamdz",
            "lam_linear_dlamdz",
            "lam_overturning_dlamdz",
            "lam_vanishing_dlamdz"
        ],
        "local_wmt": [
            "constant_plus_diffusion_local_wmt_dlamdz_constant",
            "constant_plus_diffusion_local_wmt_dlamdz_linear",
            "constant_plus_diffusion_local_wmt_dlamdz_overturning",
            "differential_heating_local_wmt_dlamdz_vanishing"
        ],
        "layer_wmt":[
            "constant_plus_diffusion_layer_wmt_dlamdz_constant",
            "constant_plus_diffusion_layer_wmt_dlamdz_linear",
            "constant_plus_diffusion_layer_wmt_dlamdz_overturning",
            "differential_heating_layer_wmt_dlamdz_vanishing"
        ]
    }

@pytest.fixture
def helpers():
    return Helpers()
