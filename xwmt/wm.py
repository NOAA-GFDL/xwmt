import numpy as np
import xarray as xr
import xgcm
import gsw
import warnings


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
        teos10=True,
        cp=3992.0,
        rho_ref=1035.0,
        t_var="conservative",
        s_var="absolute",
    ):
        """
        Create a new WaterMass object from an input xgcm.Grid instance.

        Parameters
        ----------
        grid: xgcm.Grid
            Contains information about ocean model grid coordinates, metrics, and data variables.
        t_name: str (default: "thetao")
            Name of conservative temperature variable [in degrees Celsius] in ds.
        s_name: str (default: "so")
            Name of absolute salinity variable [in g/kg] in ds.
        h_name: str (default: "thkcello")
            Name of thickness variable [in m] in ds.
        teos10 : boolean (default: True)
            Get expansion/contraction coefficients from the Thermodynamic Equation Of Seawater - 2010 (TEOS-10),
            unless "alpha" and "beta" variables already present in `grid._ds`.
        cp: float (default: 3992.0)
            Value of specific heat capacity.
        rho_ref: float (default: 1035.0)
            Value of reference potential density, assuming Boussinesq approximation.
        t_var: str ("conservative", "potential", or "in-situ")
            Does variable `t_name` represent "conservative", "potential", or "in-situ" temperature?
        s_var: str ("absolute" or "practical")
            Does variable `s_name` represent "absolute" or "practical" salinity?
        """
        # Work on an isolated deep copy so we never mutate the caller's grid/dataset.
        self.grid = _rebuild_grid(grid, deep=True)
        self.t_name = t_name
        self.t_var = t_var
        self.s_name = s_name
        self.s_var = s_var
        self.h_name = h_name
        self.teos10 = teos10
        self.cp = cp
        self.rho_ref = rho_ref

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
        Derive density variables from layer temperature, salinity, and thickness,
        and add them to the dataset (if not already present).
        Uses the TEOS10 algorithm from the `gsw` package by default, unless "alpha"
        and "beta" variables are already provided in `self.grid._ds`.

        Parameters
        ----------
        density_name: str (default: "rho")
            Name of density variable. Supported density variables are:
            "rho" (in-situ), "sigma0", "sigma1", "sigma2", "sigma3", "sigma4"
            (corresponding to functions of the same name in the `gsw` package).

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

        if (
            "alpha" not in self.grid._ds or "beta" not in self.grid._ds or self.teos10
        ) and "p" not in self.grid._ds.data_vars:
            self.grid._ds["p"] = xr.apply_ufunc(
                gsw.p_from_z,
                self.grid._ds.z,
                self.grid._ds.lat,
                0,
                0,
                dask="parallelized",
            )

        if "sigma" in density_name:
            z_ref = density_name.replace("sigma", "")
            try:
                z_ref = -float(z_ref) * 1000
            except ValueError as e:
                raise ValueError(
                    f"`density_name = {density_name}` is not of form 'sigmaX' where 'X' is a number."
                ) from e

            p_ref = xr.apply_ufunc(
                gsw.p_from_z, z_ref, self.grid._ds.lat, 0, 0, dask="parallelized"
            )
        elif density_name == "rho":
            z_ref = self.grid._ds.z
            p_ref = self.grid._ds.p
        else:
            raise NameError(
                f"`density_name = {density_name}` is not a supported option."
            )

        # Prognostic temperature and salinity are, by default, interpreted as
        # conservative temperature and absolute salinity (following McDougall et al. 2021).
        if self.teos10 and "sa" not in self.grid._ds:
            if self.s_var == "absolute":
                self.grid._ds["sa"] = self.grid._ds[self.s_name]
            elif self.s_var == "practical":
                self.grid._ds["sa"] = xr.apply_ufunc(
                    gsw.SA_from_SP,
                    self.grid._ds[self.s_name],
                    self.grid._ds.p,
                    self.grid._ds.lon,
                    self.grid._ds.lat,
                    dask="parallelized",
                )
        if self.teos10 and "ct" not in self.grid._ds:
            if self.t_var == "conservative":
                self.grid._ds["ct"] = self.grid._ds[self.t_name]
            elif self.t_var == "potential":
                self.grid._ds["ct"] = xr.apply_ufunc(
                    gsw.CT_from_pt,
                    self.grid._ds.sa,
                    self.grid._ds[self.t_name],
                    dask="parallelized",
                )
            elif self.t_var == "in-situ":
                self.grid._ds["ct"] = xr.apply_ufunc(
                    gsw.CT_from_t,
                    self.grid._ds.sa,
                    self.grid._ds[self.t_name],
                    self.grid._ds.p,
                    dask="parallelized",
                )
        if not self.teos10 and ("sa" not in self.grid._ds or "ct" not in self.grid._ds):
            self.grid._ds["sa"] = self.grid._ds[self.s_name]
            self.grid._ds["ct"] = self.grid._ds[self.t_name]

        # Calculate thermal expansion coefficient alpha (1/K) at reference pressure
        if "alpha" not in self.grid._ds:
            self.grid._ds["alpha"] = xr.apply_ufunc(
                gsw.alpha,
                self.grid._ds.sa,
                self.grid._ds.ct,
                p_ref,
                dask="parallelized",
            )

        # Calculate the haline contraction coefficient beta (kg/g) at reference pressure
        if "beta" not in self.grid._ds:
            self.grid._ds["beta"] = xr.apply_ufunc(
                gsw.beta, self.grid._ds.sa, self.grid._ds.ct, p_ref, dask="parallelized"
            )

        # Calculate potential density (kg/m^3)
        if density_name not in self.grid._ds:
            if density_name == "rho":
                self.grid._ds[density_name] = xr.apply_ufunc(
                    getattr(gsw, density_name),
                    self.grid._ds.sa,
                    self.grid._ds.ct,
                    self.grid._ds.p,
                    dask="parallelized",
                ).rename(density_name)

            elif "sigma" in density_name:
                self.grid._ds[density_name] = xr.apply_ufunc(
                    getattr(gsw, density_name),
                    self.grid._ds.sa,
                    self.grid._ds.ct,
                    dask="parallelized",
                ).rename(density_name)

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
    )


def add_gridcoords(grid, coords, padding):
    new_grid = _rebuild_grid(grid, extra_coords=coords, extra_padding=padding)
    # Preserve a Z_metrics attribute if a caller attached one to the raw grid.
    if "Z_metrics" in vars(grid):
        new_grid.Z_metrics = grid.Z_metrics

    return new_grid
