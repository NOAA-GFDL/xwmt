import numpy as np
import xarray as xr
import xgcm
import gsw
import warnings

from xwmt import attrs as _attrs
from xwmt.eos import resolve_eos, convert_ts

# Sentinel for the `eos` default, so an explicitly-passed `eos` (even the string
# "teos10") can be distinguished from the unset default when reconciling the
# deprecated `teos10` keyword.
_EOS_UNSET = object()


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
        eos=_EOS_UNSET,
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
            Name of the layer-thickness variable [in m] in ds. This is the mass
            budget's prognostic state variable and a core requirement of
            WaterMass: it is interpolated to interfaces and integrated to build
            the vertical metrics (`Z_metrics`) and depth coordinates. When
            constructed via `WaterMassTransformations`, it is taken from the mass
            budget's `thickness` metadata (`recipe["mass"]["thickness"]`).
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
            Deprecated. Use `eos` instead. `teos10=True` selects `eos="teos10"`.
            `teos10=False` maps to `eos=None`, which (unlike the old `teos10=False`,
            which computed alpha/beta/density from gsw) requires "alpha", "beta" and
            the density variable to be present in `grid._ds`; to compute density from
            a specific equation of state, pass an explicit `eos=` id instead. An
            explicitly-passed `eos` always takes precedence over `teos10`.
        """
        if teos10 is not None:
            warnings.warn(
                "`teos10` is deprecated; use `eos` instead. `teos10=True` selects "
                "`eos='teos10'`; `teos10=False` maps to `eos=None`, which now requires "
                "alpha/beta/density to be supplied in `grid._ds` (it no longer falls "
                "back to gsw). Pass an explicit `eos=` id to compute density from a "
                "specific equation of state.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Only honor the legacy flag if `eos` was left unset, so an explicit
            # `eos=...` (even "teos10") always takes precedence.
            if eos is _EOS_UNSET:
                eos = "teos10" if teos10 else None
        if eos is _EOS_UNSET:
            eos = "teos10"
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
            for name in (self.h_name, f"{self.h_name}_i"):
                self.grid._ds[name].attrs.update(
                    _attrs.synthetic_thickness_attrs(name.endswith("_i"))
                )
            self.grid = _rebuild_grid(
                self.grid,
                extra_coords={"Z": {"center": "z_l", "outer": "z_i"}},
                extra_padding={"Z": "extend"},
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
        self._describe_derived(
            f"{self.h_name}_i",
            _attrs.thickness_interface_attrs(self.grid._ds.get(self.h_name)),
        )

    def _compute_depth_coordinates(self):
        """Compute layer-center height `z` and interface height `z_interface` from Z_metrics."""
        self.grid._ds["z"] = (-self.grid.cumsum(self.Z_metrics["outer"], "Z")).chunk(
            {self._zc: -1}
        )
        self.grid._ds["z_interface"] = xr.where(
            self.grid.axes["Z"].coords["outer"]
            != self.grid.axes["Z"].coords["outer"][0],
            -self.grid.cumsum(self.Z_metrics["center"], "Z", to="outer"),
            0.0,
        ).chunk({self._zi: -1})
        for name in ("z", "z_interface"):
            self._describe_derived(name, _attrs.derived_field_attrs(name))

    def _describe_derived(self, name, attrs, keep_identity=False):
        """
        Describe a field that xwmt derived onto its own copy of the dataset.

        Fields built by arithmetic inherit the attributes of whichever input
        xarray took them from -- left alone, the pressure field `p` reports the
        layer thickness's "Cell Thickness" in metres. Those inherited attributes
        are stripped before xwmt's own are applied.

        `keep_identity=True` preserves `units`/`long_name`/`standard_name`, for the
        fields an upstream package annotates deliberately -- `xeos` describes the
        `alpha`, `beta` and density arrays it returns, and those descriptions win
        over anything xwmt would assert.
        """
        if name not in self.grid._ds:
            return
        da = self.grid._ds[name]
        _attrs.strip_inherited_attrs(da, identity=not keep_identity)
        _attrs.set_default_attrs(da, attrs)

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
    def _facedim(self):
        """
        Name of the face/tile dimension for multi-tile grids (e.g. the 13-tile
        ECCO/LLC lat-lon-cap grid), or None for an ordinary single-tile grid.

        Read from the underlying `xgcm.Grid`'s face-connection metadata (xgcm
        stores it as `_facedim` when the grid is built with `face_connections`),
        following the same accessor convention as the `sectionate` package.
        """
        return getattr(self.grid, "_facedim", None)

    @property
    def _horizontal_dims(self):
        """
        Names of the horizontal dimensions to reduce over when integrating: the
        X and Y center coordinates, plus the face/tile dimension for multi-tile
        grids. Including the face dimension is what lets horizontal integrations
        (and lambda-space binning) broadcast across tiles on multi-tile grids
        such as ECCOv4r4 (see GitHub issue #59).
        """
        facedim = self._facedim
        return ([facedim] if facedim is not None else []) + [self._xc, self._yc]

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

        Notes
        -----
        Unless they were supplied by the caller at construction, "alpha" and "beta"
        are (re)computed at *this* call's reference pressure and written to
        `self.grid._ds`. Calling `get_density` for different `density_name`s on the
        same instance therefore overwrites them (each density gets alpha/beta at its
        own reference pressure); hold a copy if you need an earlier call's values.
        The converted EOS-input temperature/salinity are used internally and are not
        added to `self.grid._ds`.
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
            self._describe_derived("p", _attrs.derived_field_attrs("p"))

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
            # The depth argument is a bare scalar, so `apply_ufunc` hands the
            # result `lat`'s attributes -- a reference pressure labelled
            # "degrees_north". Describe it as what it is.
            p_ref.attrs = _attrs.derived_field_attrs("p")
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
        # Only fields xwmt computed here are re-described: a caller-supplied alpha,
        # beta or density comes with whatever metadata its author intended, and it
        # is not xwmt's to strip.
        if not self._user_alpha:
            self.grid._ds["alpha"] = self.eos.alpha(temp, salt, p_ref)
            # `keep_identity`: the EOS backend describes what it returns better
            # than xwmt could, so only the sampling attributes that rode in from
            # the model's temperature and salinity diagnostics are cleared.
            self._describe_derived("alpha", {}, keep_identity=True)
        if not self._user_beta:
            self.grid._ds["beta"] = self.eos.beta(temp, salt, p_ref)
            self._describe_derived("beta", {}, keep_identity=True)

        # Density (kg/m^3): in-situ for "rho", potential-density anomaly for "sigmaX".
        if density_name not in self.grid._ds:
            density = self.eos.rho(temp, salt, p_density)
            if "sigma" in density_name:
                density = density - 1000.0
            self.grid._ds[density_name] = density.rename(density_name)
            # Subtracting 1000 drops the EOS backend's attributes, and they would
            # describe an in-situ density rather than a potential density anomaly
            # anyway, so xwmt owns the sigmaX metadata but defers on "rho".
            self._describe_derived(
                density_name,
                _attrs.derived_field_attrs(density_name),
                keep_identity=density_name == "rho",
            )

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

    def infer_bins(
        self, da, percentiles=(0.0, 1.0), nbins=100, surface=False, spacing="linear"
    ):
        """
        Infer bin edges from the distribution of `da`.

        The two arguments do separate jobs: `percentiles` fixes where the bins
        *start and end*, and `spacing` fixes how the edges are distributed
        *between* those endpoints.

        Parameters
        ----------
        da: xarray.DataArray
            Variable used to determine bins.
        percentiles: sequence of float
            Length-2 sequence giving the lower and upper quantiles that bound the
            bins, as fractions in [0, 1] (not 0-100, despite the name, which is
            kept for backwards compatibility). Default: (0., 1.), i.e. the min and
            max of `da`. Use e.g. (0.01, 0.99) to keep outliers from stretching the
            range. Note this only bounds the range; on its own it says nothing
            about how the edges in between are spaced.
        nbins: int
            Number of bin edges, so `nbins - 1` bins. Default: 100.
        surface: bool
            Default: False. If True, infer bins only from the outcropping layer of `da`.
        spacing: {"linear", "quantiles"}
            How to distribute the edges across the range. Default: "linear",
            equally spaced in the value of `da` -- bins of equal width, holding
            unequal amounts of water. "quantiles" instead places the edges at the
            quantiles `np.linspace(*percentiles, nbins)` of `da`, giving bins that
            each hold roughly the same number of grid cells and so resolve
            whichever range of lambda the water actually occupies. The two agree
            when `da` is uniformly distributed.

        Returns
        -------
        numpy.ndarray
            `nbins` bin edges, increasing.

        Notes
        -----
        Bin edges have to be concrete numbers, so this is one of the few places
        that forces computation of a lazy `da`. Pass `bins=` explicitly to keep a
        pipeline fully lazy.
        """
        if spacing not in ("linear", "quantiles"):
            raise ValueError(
                f"`spacing` must be 'linear' or 'quantiles', got {spacing!r}."
            )
        if len(percentiles) != 2:
            raise ValueError(
                f"`percentiles` must have length 2, got {len(percentiles)}."
            )
        lower, upper = (float(p) for p in percentiles)
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(
                f"`percentiles` must be increasing fractions within [0, 1], got "
                f"{tuple(percentiles)!r}."
            )

        if surface:
            da = self.sel_outcrop_lev(da)

        if spacing == "quantiles":
            edges = da.quantile(np.linspace(lower, upper, nbins), dim=da.dims)
            edges = np.asarray(edges, dtype=float)
            # Quantiles of a field with plateaus (a masked basin, a well-mixed
            # layer) can repeat, and a zero-width bin divides by zero downstream.
            # Say so rather than returning silently degenerate bins.
            if np.any(np.diff(edges) == 0.0):
                warnings.warn(
                    f"`spacing='quantiles'` produced repeated bin edges for "
                    f"'{da.name}': its distribution is too concentrated to split "
                    f"into {nbins - 1} equally populated bins. The resulting "
                    f"zero-width bins will give infinite or undefined "
                    f"transformation rates; use fewer bins or spacing='linear'."
                )
            return edges

        if (lower, upper) != (0.0, 1.0):
            vmin, vmax = da.quantile([lower, upper], dim=da.dims)
        else:
            vmin, vmax = da.min(), da.max()
        # `float()` both computes a lazy result and, crucially, hands numpy plain
        # scalars: `np.linspace` on 0-d DataArrays tries to wrap its 1-D output
        # back into a 0-d DataArray and raises.
        return np.linspace(float(vmin), float(vmax), nbins)


def _rebuild_grid(grid, extra_coords=None, extra_padding=None, deep=False):
    """
    Reconstruct an `xgcm.Grid` from an existing one, preserving its coords, metrics, and
    padding settings, and optionally adding more via `extra_coords`/`extra_padding`.

    Parameters
    ----------
    grid : xgcm.Grid
        Source grid to copy the configuration (and dataset) from.
    extra_coords : dict, optional
        Additional `coords` entries to merge in (e.g. a new "Z" axis).
    extra_padding : dict, optional
        Additional `padding` entries to merge in.
    deep : bool (default: False)
        If True, deep-copy the underlying dataset so the source is never mutated.
    """
    ds = grid._ds.copy() if deep else grid._ds
    # Preserve multi-tile topology (e.g. the ECCO/LLC `face_connections`) across
    # reconstruction. Without this, the deep copy made in `WaterMass.__init__`
    # would silently drop the face dimension (xgcm's `_facedim`), and horizontal
    # integrations would no longer broadcast across tiles (see issue #59). xgcm
    # stores the original mapping on the grid as `_face_connections`.
    face_connections = getattr(grid, "_face_connections", None)
    extra = {"face_connections": face_connections} if face_connections else {}
    return xgcm.Grid(
        ds,
        coords={
            **{ax: grid.axes[ax].coords for ax in grid.axes.keys()},
            **(extra_coords or {}),
        },
        metrics={k: vv.name for (k, v) in grid._metrics.items() for vv in v},
        padding={
            **{ax: grid.axes[ax].padding for ax in grid.axes.keys()},
            **(extra_padding or {}),
        },
        autoparse_metadata=False,
        **extra,
    )


def add_gridcoords(grid, coords, padding):
    new_grid = _rebuild_grid(grid, extra_coords=coords, extra_padding=padding)
    # Preserve a Z_metrics attribute if a caller attached one to the raw grid.
    if "Z_metrics" in vars(grid):
        new_grid.Z_metrics = grid.Z_metrics

    return new_grid
