import seawater as sw
import fastjmd95
import momlevel
import gsw
import xarray as xr

eos_list = ["TEOS-10", "EOS-80", "JMD-95", "Wright-97"]
def _raise_eos_error():
    raise ValueError(f"This eos is not currently supported; choose from f{eos_list}.")

def rho(t, s, p, eos):
    if eos=="TEOS-10":
        return xr.apply_ufunc(
                gsw.rho, s, t, p, dask="parallelized"
            )
    elif eos=="EOS-80":
        return xr.apply_ufunc(
                sw.dens, s, t, p, dask="parallelized"
        )
    elif eos=="JMD-95":
        return fastjmd95.rho(s, t, p)
    elif eos=="Wright-97":
        return momlevel.derived.calc_rho(t, s, p, eos="Wright")
    else:
        _raise_eos_error()

def alpha(t, s, p, eos):
    if eos=="TEOS-10":
        return xr.apply_ufunc(
                gsw.alpha, s, t, p, dask="parallelized"
            )
    elif eos=="EOS-80":
        return xr.apply_ufunc(
                sw.alpha, s, t, p, dask="parallelized"
        )
    elif eos=="JMD-95":
        return -fastjmd95.drhodt(s, t, p)/rho(t, s, p, eos="JMD-95")
    elif eos=="Wright-97":
        return momlevel.derived.calc_alpha(t, s, p, eos="Wright")
    else:
        _raise_eos_error()

def beta(t, s, p, eos):
    if eos=="TEOS-10":
        return xr.apply_ufunc(
                gsw.beta, s, t, p, dask="parallelized"
            )
    elif eos=="EOS-80":
        return xr.apply_ufunc(
                sw.beta, s, t, p, dask="parallelized"
        )
    elif eos=="JMD-95":
        return fastjmd95.drhods(s, t, p)/rho(t, s, p, eos="JMD-95")
    elif eos=="Wright-97":
        return momlevel.derived.calc_beta(t, s, p, eos="Wright")
    else:
        _raise_eos_error()