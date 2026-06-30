import numpy as np
import xarray as xr
import xgcm
import gsw
import warnings

from xwmt.eos import resolve_eos, convert_ts


class WaterMass:
    """
    Stores water mass characteristics and supports methods for analyzing water masses on a numerical grid.
    """

    def __init__(
        self,
        grid,
        t_name="thetao",
        s_name="so",
        h_name="thkcello",
        eos="teos10",
        cp=3992.0,
        rho_ref=1035.0,
        t_var="conservative",
        s_var="absolute",
        teos10=None,
    ):
        """
        Create a new WaterMass object from an input xgcm.Grid instance.

        Parameters
        ----------
        grid: xgcm.Grid
            Contains information about ocean model grid coordinates, metrics, and data variables.
        t_name: str (default: "thetao")
            Name of the temperature variable [in degrees Celsius] in ds.
            Its kind (conservative/potential/in-situ) is declared by `t_var`.
        s_name: str (default: "so")
            Name of the salinity variable in ds. Its kind (absolute/practical)
            is declared by `s_var`.
        h_name: str (default: "thkcello")
            Name of thickness variable [in m] in ds.
        eos : str, xeos.EquationOfState, or None (default: "teos10")
            Equation of state used to derive density and the expansion/contraction
            coefficients `alpha`/`beta`. Either a canonical `xeos` EOS id (see
            `xwmt.eos.list_eos()` / `xeos.list_eos()`, e.g. "teos10",
            "wright97-full", "jmd95"), an already-built `xeos.EquationOfState`
            (e.g. from `xeos.from_model(...)`), or None. If None, "alpha", "beta"
            and the requested density variable must already be present in `grid._ds`.
            Temperature/salinity are automatically converted (via `gsw`) from the
            kinds declared by `t_var`/`s_var` to the kinds the EOS expects.
        cp: float (default: 3992.0)
            Value of specific heat capacity.
        rho_ref: float (default: 1035.0)
            Value of reference potential density, assuming Boussinesq approximation.
        t_var: str ("conservative", "potential", or "in-situ")
            Does variable `t_name` represent "conservative", "potential", or "in-situ" temperature?
        s_var: str ("absolute" or "practical")
            Does variable `s_name` represent "absolute" or "practical" salinity?
        teos10 : bool, optional
            Deprecated. Use `eos` instead. `teos10=True` maps to `eos="teos10"`
            and `teos10=False` maps to `eos=None` (alpha/beta provided in `grid._ds`).
        """
        if teos10 is not None:
            warnings.warn(
                "`teos10` is deprecated; use `eos` instead "
                "(`teos10=True` -> `eos='teos10'`, `teos10=False` -> `eos=None`).",
                DeprecationWarning,
                stacklevel=2,
            )
            # Only honor the legacy flag if `eos` was left at its default, so an
            # explicit `eos=...` always takes precedence.
            if eos == "teos10":
                eos = "teos10" if teos10 else None
        # Work on an isolated deep copy so we never mutate the caller's grid/dataset.
        self.grid = _rebuild_grid(grid, deep=True)
        self.t_name = t_name
        self.t_var = t_var
        self.s_name = s_name
        self.s_var = s_var
        self.h_name = h_name
        self.eos = resolve_eos(eos)
        self.cp = cp
        self.rho_ref = rho_ref
        # Remember whether the user supplied alpha/beta up front: if so, those are
        # honored as-is and never overwritten by EOS-derived values.
        self._user_alpha = "alpha" in self.grid._ds
        self._user_beta = "beta" in self.grid._ds

        self._build_vertical_metrics()
        self._compute_depth_coordinates()

    def _build_vertical_metrics(self):
        """
        Populate `self.Z_metrics` with the cell-center and cell-interface thickness
        metrics. For data with a thickness variable, interpolate thickness to interfaces;
        for purely 2D/surface data with no "Z" axis, synthesize a single-layer Z axis.
        """
        if hasattr(self, "Z_metrics"):
            return
        if self.h_name in self.grid._ds:
            self._interpolate_thickness_to_interfaces()
            self.Z_metrics = {
                "center": self.grid._ds[self.h_name],
                "outer": self.grid._ds[f"{self.h_name}_i"],
            }
        elif "Z" not in self.grid.axes:
            self.grid._ds["z_l"] = xr.DataArray([0.5], dims=("z_l",))
            self.grid._ds["z_i"] = xr.DataArray([0, 1.0], dims=("z_i",))
            self.grid._ds[f"{self.h_name}"] = xr.DataArray([1], dims=("z_l",))
            self.grid._ds[f"{self.h_name}_i"] = xr.DataArray([0.5, 0.5], dims=("z_i",))
            self.grid = _rebuild_grid(
                self.grid,
                extra_coords={"Z": {"center": "z_l", "outer": "z_i"}},
                extra_boundary={"Z": "extend"},
            )
            self.Z_metrics = {
                "center": self.grid._ds[f"{self.h_name}"],
                "outer": self.grid._ds[f"{self.h_name}_i"],
            }

    def _interpolate_thickness_to_interfaces(self):
        """
        Conservatively interpolate layer thickness to cell interfaces (`{h_name}_i`),
        needed to estimate depths of layer centers and compute surface flux divergences.
        """
        zc, zi = self._zc, self._zi
        Z_center_extended = np.concatenate(
            (
                self.grid._ds[zi][np.array([0])].values,
                self.grid._ds[zc].values,
                self.grid._ds[zi][np.array([-1])].values,
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            if "time" in self.grid._ds:
                time_attrs = self.grid._ds.time.attrs.copy()
            self.grid._ds[f"{self.h_name}_i"] = self.grid.transform(
                self.grid._ds[self.h_name].fillna(0.0),
                "Z",
                Z_center_extended,
                method="conservative",
            ).assign_coords({zi: self.grid._ds[zi].values})
        if "time" in self.grid._ds:
            self.grid._ds.time.attrs = (
                time_attrs  # For some reason these are not preserved by default
            )

    def _compute_depth_coordinates(self):
        """Compute layer-center depth `z` and interface depth `z_interface` from Z_metrics."""
        self.grid._ds["z"] = (-self.grid.cumsum(self.Z_metrics["outer"], "Z")).chunk(
            {self._zc: -1}
        )
        self.grid._ds["z_interface"] = xr.where(
            self.grid.axes["Z"].coords["outer"]
            != self.grid.axes["Z"].coords["outer"][0],
            -self.grid.cumsum(self.Z_metrics["center"], "Z", to="outer"),
            0.0,
        ).chunk({self._zi: -1})

    @property
    def _xc(self):
        """Name of the X (zonal) center coordinate."""
        return self.grid.axes["X"].coords["center"]

    @property
    def _yc(self):
        """Name of the Y (meridional) center coordinate."""
        return self.grid.axes["Y"].coords["center"]

    @property
    def _zc(self):
        """Name of the Z (vertical) center coordinate."""
        return self.grid.axes["Z"].coords["center"]

    @property
    def _zi(self):
        """Name of the Z (vertical) outer/interface coordinate."""
        return self.grid.axes["Z"].coords["outer"]

    @property
    def _horizontal_dims(self):
        """Names of the horizontal (X, Y) center coordinates."""
        return [self._xc, self._yc]

    def get_density(self, density_name="rho"):
        """
        Derive a density variable from layer temperature and salinity, along with
        the thermal-expansion (`alpha`) and haline-contraction (`beta`) coefficients,
        and add them to the dataset (if not already present).

        Density and the coefficients are evaluated with the equation of state
        selected at construction (`self.eos`), delegated to the `xeos` package.
        Temperature and salinity are first converted (via `gsw`) from the kinds
        declared by `t_var`/`s_var` to the kinds the EOS expects. If `self.eos`
        is None, "alpha", "beta" and `density_name` must already be present in
        `self.grid._ds`.

        Parameters
        ----------
        density_name: str (default: "rho")
            Name of density variable. Supported density variables are:
            "rho" (in-situ density at the local pressure) and "sigma0", "sigma1",
            "sigma2", "sigma3", "sigma4" (potential density anomaly referenced to
            0, 1000, ..., 4000 dbar).

        Returns
        -------
        xr.DataArray
        """

        if self.t_name not in self.grid._ds:
            raise ValueError(f"ds must include temperature variable\
            defined by kwarg t_name (default: {self.t_name}).")
        if self.s_name not in self.grid._ds:
            raise ValueError(f"ds must include salinity variable\
            defined by kwarg s_name (default: {self.s_name}).")
        if self.h_name not in self.grid._ds:
            raise ValueError(f"ds must include thickness variable\
            defined by kwarg h_name (default: {self.h_name}).")

        # With no EOS, alpha/beta/density must all be supplied by the caller.
        if self.eos is None:
            for required in ("alpha", "beta", density_name):
                if required not in self.grid._ds:
                    raise ValueError(
                        f"With `eos=None`, {required!r} must already be present in "
                        f"`grid._ds` (got variables: {list(self.grid._ds.data_vars)})."
                    )
            return self.grid._ds[density_name]

        # In-situ sea pressure [dbar] from depth, used by the EOS and by the
        # temperature/salinity kind conversions.
        if "p" not in self.grid._ds.data_vars:
            self.grid._ds["p"] = xr.apply_ufunc(
                gsw.p_from_z,
                self.grid._ds.z,
                self.grid._ds.lat,
                0,
                0,
                dask="parallelized",
            )

        # `p_ref` is the pressure at which alpha/beta are evaluated; `p_density` is
        # the pressure at which the density variable itself is evaluated. For "rho"
        # both are the in-situ pressure. For "sigmaX" the density is referenced to
        # exactly X*1000 dbar (the defining reference pressure of the sigmaX anomaly),
        # while alpha/beta retain the historical `p_from_z(-X*1000 m)` reference.
        if "sigma" in density_name:
            ref_km = density_name.replace("sigma", "")
            try:
                ref_km = float(ref_km)
            except ValueError as e:
                raise ValueError(
                    f"`density_name = {density_name}` is not of form 'sigmaX' where 'X' is a number."
                ) from e
            p_ref = xr.apply_ufunc(
                gsw.p_from_z,
                -ref_km * 1000,
                self.grid._ds.lat,
                0,
                0,
                dask="parallelized",
            )
            p_density = ref_km * 1000.0
        elif density_name == "rho":
            p_ref = self.grid._ds.p
            p_density = self.grid._ds.p
        else:
            raise NameError(
                f"`density_name = {density_name}` is not a supported option."
            )

        # Convert temperature/salinity to the kinds the chosen EOS expects.
        temp, salt = convert_ts(
            self.grid._ds[self.t_name],
            self.grid._ds[self.s_name],
            self.eos,
            self.t_var,
            self.s_var,
            self.grid._ds.p,
            lon=self.grid._ds.get("lon"),
            lat=self.grid._ds.get("lat"),
        )

        # Thermal expansion coefficient alpha and haline contraction coefficient beta
        # at the reference pressure (unless supplied by the caller).
        if not self._user_alpha:
            self.grid._ds["alpha"] = self.eos.alpha(temp, salt, p_ref)
        if not self._user_beta:
            self.grid._ds["beta"] = self.eos.beta(temp, salt, p_ref)

        # Density (kg/m^3): in-situ for "rho", potential-density anomaly for "sigmaX".
        if density_name not in self.grid._ds:
            density = self.eos.rho(temp, salt, p_density)
            if "sigma" in density_name:
                density = density - 1000.0
            self.grid._ds[density_name] = density.rename(density_name)

        return self.grid._ds[density_name]

    def get_outcrop_lev(self, position="center", incrop=False, min_thickness=1e-6):
        """
        Return the first vertical level (starting from the relevant boundary)
        whose thickness exceeds `min_thickness`.

        Assumes the native Z index increases downward (surface -> bottom).

        Parameters
        ----------
        position : {"center", "outer"}
            Vertical grid position for the coordinate/metric.
        incrop : bool
            False (default): search from sea surface downward ("outcrop").
            True: search from seafloor upward ("incrop").
        min_thickness : float
            Default: 1e-6. Minimum layer thickness required to count as the first "real" cell.

        Returns
        -------
        xr.DataArray
            Vertical coordinate value, broadcast across other dims, masked where no
            layer exceeds `min_thickness`.
        """
        z_coord = self.grid.axes["Z"].coords[position]
        h = self.Z_metrics[position]

        # Native order is surface -> bottom. For incrop, reverse to bottom -> surface.
        z_native = self.grid._ds[z_coord]
        z_order = z_native[::-1] if incrop else z_native

        h_ord = h.sel({z_coord: z_order})

        thick = h_ord > min_thickness

        # True only at the first thick cell in boundary->interior order
        first = (thick.cumsum(z_coord) == 1) & thick

        out = first.astype("int8").idxmax(z_coord)

        # Mask columns where no thick cell exists
        return out.where(thick.any(z_coord))

    def sel_outcrop_lev(
        self, da, incrop=False, min_thickness=1e-6, position="center", **kwargs
    ):
        """
        Select `da` at the first vertical level (starting from the relevant boundary)
        whose thickness exceeds `min_thickness`.

        Assumes the native Z index increases downward (surface -> bottom).

        Parameters
        ----------
        da : xr.DataArray
            DataArray to select from. Must have the same dims as the thickness metric
            after applying `**kwargs`.
        incrop : bool
            Default: False. If True, selects the first thick level from the bottom upward.
        min_thickness : float
            Minimum layer thickness required to count as the first "real" cell.
        position : {"center", "outer"}
            Vertical grid position for the coordinate/metric.
        **kwargs : dict
            Passed to `.sel(**kwargs)` on the thickness metric (and used to validate dims).

        Returns
        -------
        xr.DataArray
            `da` selected at the diagnosed level, masked where no level exceeds `min_thickness`.
        """
        z_coord = self.grid.axes["Z"].coords[position]

        # Thickness metric, subset as requested
        h = self.Z_metrics[position].sel(**kwargs)

        missing_dims = set(h.dims) - set(da.dims)
        if missing_dims:
            raise ValueError(
                f"`da` is missing required dimensions {missing_dims}. "
                f"`da.dims={da.dims}`, `h.dims={h.dims}`"
            )

        # Order so the relevant boundary is first along z (native is surface->bottom)
        z_native = self.grid._ds[z_coord]
        z_order = z_native[::-1] if incrop else z_native

        h_ord = h.sel({z_coord: z_order})
        da_ord = da.sel({z_coord: z_order})

        thick = h_ord > min_thickness

        # First thick cell in this boundary->interior ordering
        first = (thick.cumsum(z_coord) == 1) & thick
        lev = first.astype("int8").idxmax(z_coord)

        has_thick = thick.any(z_coord)

        # Select from the *original* `da` (order doesn’t matter once we have coord values)
        return da.sel({z_coord: lev}).where(has_thick)

    def expand_surface_array_vertically(self, da_surf, target_position="outer"):
        """
        Expand surface xr.DataArray (with no "Z"-dimension coordinate) in the vertical,
        filling with zeros in all layers except the one that outcrops.

        Parameters
        ----------
        da_surf: xarray.DataArray
            Variable that is to be expanded in the vertical.
        position : str
            Position of the desired vertical coordinate in the `self.grid` instance of `xgcm.Grid`.
            Default: "outer". Other supported option is "center".
        """
        z_coord = self.grid.axes["Z"].coords[target_position]
        return da_surf.expand_dims({z_coord: self.grid._ds[z_coord]}).where(
            self.grid._ds[z_coord] == self.get_outcrop_lev(position=target_position),
            0.0,
        )

    def infer_bins(self, da, percentiles=[0.0, 1.0], nbins=100, surface=False):
        """
        Specify bins based on the distribution of `da`, excluding outliers.

        Parameters
        ----------
        da: xarray.DataArray
            Variable used to determine bins.
        percentiles: list
            List of length 2 containing the upper and lower percentiles to bound the array of bins.
            Default: [0., 1.], i.e. min and max.
        nbins: int
            Number of bins. Default: 100.
        surface: bool
            Default: False. If True, compute percentiles only from the outcropping layer of `da`.
        """
        if surface:
            da = self.sel_outcrop_lev(da)
        if percentiles != [0.0, 1.0]:
            vmin, vmax = da.quantile(percentiles, dim=da.dims)
        else:
            vmin, vmax = da.min(), da.max()
        return np.linspace(vmin, vmax, nbins)


def _rebuild_grid(grid, extra_coords=None, extra_boundary=None, deep=False):
    """
    Reconstruct an `xgcm.Grid` from an existing one, preserving its coords, metrics, and
    boundary settings, and optionally adding more via `extra_coords`/`extra_boundary`.

    Parameters
    ----------
    grid : xgcm.Grid
        Source grid to copy the configuration (and dataset) from.
    extra_coords : dict, optional
        Additional `coords` entries to merge in (e.g. a new "Z" axis).
    extra_boundary : dict, optional
        Additional `boundary` entries to merge in.
    deep : bool (default: False)
        If True, deep-copy the underlying dataset so the source is never mutated.
    """
    ds = grid._ds.copy() if deep else grid._ds
    return xgcm.Grid(
        ds,
        coords={
            **{ax: grid.axes[ax].coords for ax in grid.axes.keys()},
            **(extra_coords or {}),
        },
        metrics={k: vv.name for (k, v) in grid._metrics.items() for vv in v},
        boundary={
            **{ax: grid.axes[ax]._boundary for ax in grid.axes.keys()},
            **(extra_boundary or {}),
        },
        autoparse_metadata=False,
    )


def add_gridcoords(grid, coords, boundary):
    new_grid = _rebuild_grid(grid, extra_coords=coords, extra_boundary=boundary)
    # Preserve a Z_metrics attribute if a caller attached one to the raw grid.
    if "Z_metrics" in vars(grid):
        new_grid.Z_metrics = grid.Z_metrics

    return new_grid
