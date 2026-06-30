"""Equation-of-state handling for xwmt, delegated to the ``xeos`` package.

xwmt derives seawater density (and the thermal-expansion ``alpha`` / haline-
contraction ``beta`` coefficients used to split density transformations into
heat- and salt-driven components) from layer temperature and salinity. The
actual equation-of-state (EOS) evaluation is delegated to ``xeos``
(https://github.com/hdrake/xeos), so *any* EOS that ``xeos`` supports can be
selected — ideally the same EOS used by the ocean model that produced the data.
See :func:`list_eos` (or ``xeos.list_eos()``) for the available formulations.

Two helpers bridge xwmt's data conventions and ``xeos``:

- :func:`resolve_eos` turns a user-facing ``eos`` specification (a canonical
  ``xeos`` id such as ``"teos10"`` or ``"wright97-full"``, an already-built
  :class:`xeos.EquationOfState`, or ``None``) into an
  :class:`xeos.EquationOfState` (or ``None``).
- :func:`convert_ts` converts the dataset's temperature/salinity — whose kinds
  the user declares via ``t_var``/``s_var`` — into the temperature/salinity
  *kinds* the chosen EOS expects, using ``gsw``. This guards against the silent-
  corruption failure mode where, e.g., conservative temperature is fed to an EOS
  that expects potential temperature.
"""

import gsw
import xarray as xr
import xeos
from xeos.conventions import TemperatureKind, SalinityKind

__all__ = ["list_eos", "resolve_eos", "convert_ts"]


def list_eos():
    """Return the sorted list of canonical EOS ids supported by ``xeos``."""
    return xeos.list_eos()


def resolve_eos(eos):
    """Resolve an ``eos`` specification to an :class:`xeos.EquationOfState`.

    Parameters
    ----------
    eos : None, str, or xeos.EquationOfState
        - ``None``: no EOS; the caller must supply ``alpha``, ``beta`` and the
          requested density variable directly in the dataset.
        - ``str``: a canonical ``xeos`` EOS id (see :func:`list_eos`), e.g.
          ``"teos10"``, ``"wright97-full"``, ``"jmd95"``.
        - :class:`xeos.EquationOfState`: used as-is (e.g. one built with
          ``xeos.from_model(...)`` or a parameterised
          ``xeos.equation_of_state("linear", rho0=..., talpha=..., sbeta=...)``).

    Returns
    -------
    xeos.EquationOfState or None
    """
    if eos is None or isinstance(eos, xeos.EquationOfState):
        return eos
    if isinstance(eos, str):
        return xeos.equation_of_state(eos)
    raise TypeError(
        "`eos` must be None, a str EOS id, or an xeos.EquationOfState, "
        f"got {type(eos).__name__}."
    )


# Map xwmt's declared input kinds onto xeos convention enums.
_T_KIND = {
    "conservative": TemperatureKind.CONSERVATIVE,
    "potential": TemperatureKind.POTENTIAL,
    "in-situ": TemperatureKind.INSITU,
}
_S_KIND = {
    "absolute": SalinityKind.ABSOLUTE,
    "practical": SalinityKind.PRACTICAL,
}


def convert_ts(t, s, eos, t_var, s_var, p, lon=None, lat=None):
    """Convert ``(t, s)`` into the temperature/salinity kinds that ``eos`` expects.

    All conversions go through absolute salinity / conservative temperature (the
    "hubs" that ``gsw``'s conversion routines are written around) and are applied
    lazily via :func:`xarray.apply_ufunc`, so xarray labels and dask laziness are
    preserved. When the input kinds already match what ``eos`` wants (e.g. the
    default conservative/absolute inputs for a TEOS-10 EOS), the inputs are
    returned unchanged with no conversion.

    Parameters
    ----------
    t, s : xr.DataArray
        Temperature and salinity whose kinds are declared by ``t_var``/``s_var``.
    eos : xeos.EquationOfState
        Target EOS; its ``temperature``/``salinity`` properties give the required kinds.
    t_var : {"conservative", "potential", "in-situ"}
        Kind of the input temperature ``t``.
    s_var : {"absolute", "practical"}
        Kind of the input salinity ``s``.
    p : xr.DataArray
        In-situ sea pressure [dbar], used by the salinity and in-situ-temperature
        conversions.
    lon, lat : xr.DataArray, optional
        Longitude/latitude, used for the practical<->absolute salinity conversion.

    Returns
    -------
    (temp, salt) : tuple of xr.DataArray
        Temperature and salinity in the kinds required by ``eos``.
    """
    if t_var not in _T_KIND:
        raise ValueError(f"`t_var` must be one of {sorted(_T_KIND)}, got {t_var!r}.")
    if s_var not in _S_KIND:
        raise ValueError(f"`s_var` must be one of {sorted(_S_KIND)}, got {s_var!r}.")
    t_in, s_in = _T_KIND[t_var], _S_KIND[s_var]
    t_out, s_out = eos.temperature, eos.salinity

    # Absolute salinity is the hub: gsw's temperature conversions all take SA.
    if s_in is SalinityKind.ABSOLUTE:
        sa = s
    else:
        sa = xr.apply_ufunc(gsw.SA_from_SP, s, p, lon, lat, dask="parallelized")

    # Produce the temperature kind the EOS wants.
    if t_out is TemperatureKind.INSITU:
        # No xeos backend currently expects in-situ temperature.
        raise NotImplementedError(
            f"EOS {eos.id!r} expects in-situ temperature, which xwmt does not supply."
        )
    if t_in is t_out:
        temp = t
    else:
        # Conservative temperature is the hub for temperature conversions.
        if t_in is TemperatureKind.CONSERVATIVE:
            ct = t
        elif t_in is TemperatureKind.POTENTIAL:
            ct = xr.apply_ufunc(gsw.CT_from_pt, sa, t, dask="parallelized")
        else:  # in-situ
            ct = xr.apply_ufunc(gsw.CT_from_t, sa, t, p, dask="parallelized")
        if t_out is TemperatureKind.CONSERVATIVE:
            temp = ct
        else:  # potential
            temp = xr.apply_ufunc(gsw.pt_from_CT, sa, ct, dask="parallelized")

    # Produce the salinity kind the EOS wants.
    if s_out is SalinityKind.ABSOLUTE:
        salt = sa
    elif s_in is SalinityKind.PRACTICAL:
        salt = s
    else:  # absolute -> practical
        salt = xr.apply_ufunc(gsw.SP_from_SA, sa, p, lon, lat, dask="parallelized")

    return temp, salt
