import copy
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram
import warnings

from xwmt.wm import WaterMass, _EOS_UNSET
from xwmt.compute import calc_hlamdot_tendency


def flatten_lol(lol):
    """Flatten a (possibly nested) list of lists into a single flat list.

    Previously imported from `xbudget`, which dropped it from its public API in
    0.7.0. Kept local since it is trivial and the only use is flattening
    `lambdas_dict.values()` (a mix of scalar variable names and the list of
    density names). Strings are treated as atoms, not iterated character-by-character.
    """

    def _flatten(container):
        for item in container:
            if isinstance(item, (list, tuple)):
                yield from _flatten(item)
            else:
                yield item

    return list(_flatten(lol))


def _resolve_recipe(recipe, xbudget_dict, func):
    """Return the recipe, honoring the deprecated ``xbudget_dict`` alias.

    ``xbudget_dict`` was the historical name for the xbudget recipe dict (renamed
    to ``recipe`` in xbudget 0.7.0). It is still accepted as a keyword so existing
    callers keep working, but it warns and will be removed in a future version.
    """
    if xbudget_dict is not None:
        if recipe is not None:
            raise TypeError(
                f"{func}() received both `recipe` and the deprecated "
                f"`xbudget_dict`; pass only `recipe`."
            )
        warnings.warn(
            f"The `xbudget_dict` argument of {func}() is deprecated and will be "
            f"removed in a future version; pass the recipe as `recipe` instead.",
            FutureWarning,
            stacklevel=3,
        )
        return xbudget_dict
    if recipe is None:
        raise TypeError(f"{func}() missing required argument: 'recipe'")
    return recipe


class WaterMassTransformations(WaterMass):
    """
    An extension of the WaterMass class that includes methods for WaterMass transformation analysis.
    """

    # Density transformations are decomposed into heat- and salt-driven components,
    # tracked throughout by the variable-name suffixes "_heat"/"_salt".
    _COMPONENTS = ("heat", "salt")

    @staticmethod
    def _component_suffix(component):
        """Variable-name suffix for a density component, e.g. 'heat' -> '_heat'."""
        return f"_{component}"

    @classmethod
    def _component_name(cls, base, component):
        """Component-specific variable name, e.g. ('diffusion', 'heat') -> 'diffusion_heat'."""
        return f"{base}{cls._component_suffix(component)}"

    # Sentinel distinguishing "metadata key absent" from "present but None"
    # (a tracer may legitimately declare `lambda: None`).
    _UNSET = object()

    @classmethod
    def _budget_metadata(cls, recipe, budget, keys):
        """Value of the first *present* key among `keys` in a budget's metadata.

        Budget metadata (`lambda`, `surface_lambda`, `thickness`) sits at the top
        level of each budget in both the raw recipe and the aggregated dict, so
        this one accessor centralizes reading it (mirrors
        `xbudget.BudgetQuery.metadata`). `keys` is tried in order, so a tracer
        that declares only `surface_lambda` still resolves. Presence is what
        matters: a key present with value `None` returns `None`; returns the
        `_UNSET` sentinel only when none of `keys` is present.
        """
        body = recipe.get(budget, {})
        for key in keys:
            if key in body:
                return body[key]
        return cls._UNSET

    def __init__(
        self,
        grid,
        recipe=None,
        mask=None,
        eos=_EOS_UNSET,
        cp=3992.0,
        rho_ref=1035.0,
        gravity=9.81,
        t_var="conservative",
        s_var="absolute",
        method="default",
        rebin=False,
        teos10=None,
        *,
        xbudget_dict=None,
    ):
        """
        Create a new WaterMassTransformation object from an input xgcm.Grid and xbudget recipe.

        Parameters
        ----------
        grid : xgcm.Grid
            Contains information about ocean model grid coordinates, metrics, and data variables.
        recipe : dict
            Nested recipe containing information about lambda and tendency variable names.
            See the `xbudget` package documentation (https://github.com/hdrake/xbudget) for how
            this recipe should be structured. In particular, the `xbudget/recipes`
            directory contains example `.yaml` files that can be read in as preset recipes.
            The `MOM6.yaml` file provides a comprehensive description of the mass, heat, and salt
            budgets in MOM6 (github.com/hdrake/xbudget/blob/main/xbudget/recipes/MOM6.yaml).
        mask : xr.DataArray (default: None)
            Boolean region mask (with same X and Y grid dimensions as `grid._ds` variables).
            If None, generate an all-True mask for domain-wide calculations.
        eos : str, xeos.EquationOfState, or None (default: "teos10")
            Equation of state used to derive density and the expansion/contraction
            coefficients. A canonical `xeos` EOS id (see `xwmt.eos.list_eos()`),
            an `xeos.EquationOfState`, or None (alpha/beta/density provided in
            `grid._ds`). See `WaterMass` for details.
        cp : float (default: 3992.0, the MOM6 default value)
            Value of specific heat capacity.
        rho_ref : float (default: 1035.0, the MOM6 default value)
            Value of reference potential density. Note: WaterMass is assumed to be Boussinesq.
        gravity : float or "gsw" (default: 9.81)
            Constant gravitational acceleration [m s-2] used to convert depth to
            pressure, or "gsw" for `gsw.p_from_z`'s latitude-dependent gravity
            (which requires a `lat` coordinate). See `WaterMass` for details.
        t_var: str ("conservative", "potential", or "in-situ")
            Does variable `t_name` represent "conservative", "potential", or "in-situ" temperature?
        s_var: str ("absolute" or "practical")
            Does variable `s_name` represent "absolute" or "practical" salinity?
        method : str (default: "default")
            Method used for vertical transformations.
            Supported options: "default", "xhistogram", "xgcm".
            If "default", use "xhistogram" for area-integrated calculations (`integrate=True`)
            or "xgcm" for column-wise calculations (`integrate=False`) for efficiency.
            The other options force the use of a specific method, perhaps at the cost of efficiency.
            Note that `method` is ignored on the prebinned path described under `rebin`,
            which always uses "xgcm". That resolution is per call: it does not mutate
            this attribute, so a later call with a non-prebinned lambda still uses the
            method you asked for.
        rebin : bool (default: False)
            Set to True to force a transformation into the target coordinates, even if these
            coordinates already exist in the `grid` data structure.

            When the requested lambda is *already* a coordinate of the grid's `Z` axis
            -- both `{lambda}_l` and `{lambda}_i` are present, as in an isopycnal-coordinate
            model whose vertical coordinate is sigma2 -- the data are already binned in
            lambda before xwmt sees them. xwmt detects this "prebinned" case and uses the
            model's own `{lambda}_l` as the lambda field, rather than recomputing it from
            temperature and salinity through the equation of state.

            `rebin=True` overrides that detection and recomputes regardless. Prefer the
            default when the model's vertical coordinate already is your target lambda:
            a layer's density is then a property of the coordinate and is known exactly,
            whereas recomputing it from layer-mean temperature and salinity is lossy --
            the equation of state is nonlinear, so the density of the mean T and S is not
            the mean of the density. `rebin=True` is for target coordinates the grid does
            not already provide.

            Unlike `mask`, `rebin` is a constructor argument only; it cannot be varied
            per call.
        teos10 : bool, optional
            Deprecated. Use `eos` instead. `teos10=True` selects `eos="teos10"`;
            `teos10=False` maps to `eos=None` (which requires alpha/beta/density to
            be present in `grid._ds`). An explicit `eos=` always takes precedence.
        """
        recipe = _resolve_recipe(recipe, xbudget_dict, "WaterMassTransformations")

        valid_methods = ("default", "xhistogram", "xgcm")
        if method not in valid_methods:
            raise ValueError(
                f"`method` must be one of {valid_methods}, got {method!r}."
            )
        self.method = method
        self.mask = mask
        self.rebin = rebin

        # Read the budget metadata that maps tracers to their lambda coordinate
        # and the mass budget to its layer-thickness variable. These live at the
        # top level of each budget (carried through by xbudget's aggregate /
        # BudgetQuery), so this single accessor works whether `recipe` is a
        # raw recipe or an aggregated dict.
        self.tracer_dict = {}
        tracers = [k for k in recipe.keys() if k != "mass"]
        for tracer in tracers:
            lam = self._budget_metadata(recipe, tracer, ("lambda", "surface_lambda"))
            if lam is self._UNSET:
                raise ValueError(
                    f"""recipe must contain element `["{tracer}"]["lambda"]` `["{tracer}"]["surface_lambda"]`"""
                )
            self.tracer_dict[tracer] = lam

        # `mass` must be present, but `thickness` within it is optional: when it is
        # omitted, `h_name` keeps its default. This supports surface-only water-mass
        # transformations (e.g. `{"mass": {}}`), where no layer thickness is needed.
        # The mass budget's `thickness` is its prognostic layer-thickness variable
        # and the core input WaterMass needs to build its vertical metrics.
        kwargs = {}
        if "mass" not in recipe:
            raise ValueError("""recipe must contain a `["mass"]` entry.""")
        thickness = self._budget_metadata(recipe, "mass", ("thickness",))
        if thickness is not self._UNSET:
            kwargs["h_name"] = thickness

        super().__init__(
            grid,
            t_name=self.tracer_dict.get("heat"),  # defaults to None if not available
            s_name=self.tracer_dict.get("salt"),  # defaults to None if not available
            eos=eos,
            cp=cp,
            rho_ref=rho_ref,
            gravity=gravity,
            t_var=t_var,
            s_var=s_var,
            teos10=teos10,
            **kwargs,
        )

        self.lambdas_dict = self.tracer_dict
        if all(k in self.tracer_dict for k in ["heat", "salt"]):
            self.lambdas_dict = {
                **self.lambdas_dict,
                **{"density": ["sigma0", "sigma1", "sigma2", "sigma3", "sigma4"]},
            }

        self.recipe = copy.deepcopy(recipe)
        for term, bdict in self.recipe.items():
            setattr(self, f"processes_{term}_dict", {})
            for ptype, _processes in bdict.items():
                if ptype in ["lhs", "rhs"]:
                    getattr(self, f"processes_{term}_dict").update(_processes)

    @property
    def xbudget_dict(self):
        """Deprecated alias for :attr:`recipe` (renamed in step with xbudget 0.7.0)."""
        warnings.warn(
            "`WaterMassTransformations.xbudget_dict` is deprecated and will be "
            "removed in a future version; use `.recipe` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.recipe

    @xbudget_dict.setter
    def xbudget_dict(self, value):
        # It was a plain instance attribute before the rename, so assignment has
        # to keep working -- a read-only property would turn a deprecation into a
        # hard AttributeError for existing callers.
        warnings.warn(
            "`WaterMassTransformations.xbudget_dict` is deprecated and will be "
            "removed in a future version; assign to `.recipe` instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.recipe = value

    def lambdas(self, lambda_key=None):
        """
        Return dictionary of desired lambdas.

        Parameters
        ----------
        lambda_key : str or list of str (default: None)
            Lambda(s) of interest. If None, return a list of all available lambdas.

        Returns
        -------
        list

        Example
        -------
        >>> wmt = WaterMassTransformations(grid, recipe)
        >>> wmt.lambdas()
        ['thetao', 'so', 'sigma0', 'sigma1', 'sigma2', 'sigma3', 'sigma4']
        """
        if lambda_key is None:
            return flatten_lol(self.lambdas_dict.values())
        else:
            return self.lambdas_dict.get(lambda_key, None)

    def get_lambda_var(self, lambda_name=None):
        """
        Search lambda dictionary for variable name corresponding to specified lambda

        Parameters
        ----------
        lambda_name : str (default: None)
            Name of lambda variable. Supported options: ["heat", "salt", "density"]
            If None, return None

        Returns
        -------
        str or list

        Example
        -------
        >>> wmt = WaterMassTransformations(grid, recipe)
        >>> wmt.get_lambda_var("heat")
        'thetao'

        >>> wmt.get_lambda_var("density")
        ['sigma0', 'sigma1', 'sigma2', 'sigma3', 'sigma4']

        See also
        --------
        self.lambdas, self.get_lambda_key
        """
        density_lambdas = (
            self.lambdas_dict["density"] if "density" in self.lambdas_dict else []
        )
        if lambda_name in self.lambdas_dict:
            return self.lambdas_dict[lambda_name]
        elif lambda_name in density_lambdas:
            return lambda_name
        else:
            return None

    def get_lambda_key(self, lambda_var):
        """
        Search lambda dictionary for lambda corresponding to variable name

        Parameters
        ----------
        lambda_var : str
            Variable name of lambda.

        Returns
        -------
        str

        Example
        -------
        >>> wmt = WaterMassTransformations(grid, recipe)
        >>> wmt.get_lambda_key("thetao")
        'heat'

        >>> wmt.get_lambda_var("sigma2")
        'density'

        See also
        --------
        self.lambdas, self.get_lambda_var
        """
        for k, v in self.lambdas_dict.items():
            if type(v) is str:
                if lambda_var == v:
                    return k
            elif type(v) is list:
                if lambda_var in v:
                    return k

    def process_names(self, tracer, term):
        """
        Get a tuple containing the names of variables for the 'tracer' and the 'process'-specific
        tendency corresponding to a general 'term' in the tendency equation.

        Parameters
        ----------
        tracer : str
            Supported options: ["heat", "salt"]
        term : str
            key for tendency variable in the recipe

        Returns
        -------
        names : tuple
            `(tracer_name, process)`

        Example
        -------
        >>> wmt = WaterMassTransformations(grid, recipe)
        >>> wmt.process_names("heat", "diffusion")
        ('thetao', 'opottempdiff')
        """
        if tracer in self.tracer_dict.keys():
            process = getattr(self, f"processes_{tracer}_dict").get(term, None)
        else:
            warnings.warn(f"Tracer {tracer} is not defined")
            return (None, None)
        tracer_name = self.tracer_dict.get(tracer, None)
        return (tracer_name, process)

    def available_processes(self, available=True):
        """
        Get a list of all tendency processes that are both specified by `recipe` and available in
        the dataset.

        Parameters
        ----------
        available: bool
            Default True. If False, include processes that are not available in the dataset.

        Returns
        -------
        processes : list of str

        >>> wmt = WaterMassTransformations(grid, recipe)
        >>> processes = wmt.available_processes()
        ['diffusion']
        """
        processes = set()
        for tracer in self.recipe.keys():
            processes |= getattr(self, f"processes_{tracer}_dict").keys()
        if available:
            _processes = []
            for process in processes:
                if all(
                    [
                        getattr(self, f"processes_{tracer}_dict").get(process, None)
                        in self.grid._ds
                        for tracer in self.recipe.keys()
                        if getattr(self, f"processes_{tracer}_dict").get(process, None)
                        is not None
                    ]
                ):
                    _processes.append(process)
            return _processes
        else:
            return processes

    def datadict(self, tracer, term):
        """
        Get a dictionary that organizes the variables and metadata necessary to evaluate tendency terms.

        Parameters
        ----------
        component : str
            Supported options: ["heat", "salt"]
        term : str
            key for tendency variable in the recipe

        Returns
        -------
        ddict : dict

        Example
        --------
        >>> wmt = WaterMassTransformations(grid, recipe)
        >>> ddict = wmt.datadict("heat", "diffusion")
        """
        tracer_name, process = self.process_names(tracer, term)

        if process is None or process not in self.grid._ds:
            return

        tend_arr = self.grid._ds[process]

        # Multiply salt tendency by 1000 to convert to g/m^2/s
        if tracer == "salt":
            tend_arr = tend_arr * 1000.0

        scalar = self.grid._ds[tracer_name]
        if self.grid.axes["Z"].coords["center"] not in scalar.dims:
            scalar = self.expand_surface_array_vertically(
                scalar, target_position="center"
            )

        n_zcoords = len(
            [
                c
                for c in self.grid.axes["Z"].coords.values()
                if c in self.grid._ds[process].dims
            ]
        )

        if n_zcoords == 0:
            tend_arr = self.expand_surface_array_vertically(
                tend_arr, target_position="center"
            )
            tend_dict = {"layer_integrated_tendency": tend_arr}
        elif n_zcoords > 0:
            if self.grid.axes["Z"].coords["outer"] in self.grid._ds[process].dims:
                tend_dict = {"interfacial_flux": tend_arr}
            elif self.grid.axes["Z"].coords["center"] in self.grid._ds[process].dims:
                tend_dict = {"layer_integrated_tendency": tend_arr}

        return {"scalar": scalar, **tend_dict}

    def rho_tend(self, term):
        """
        Get density tendency 'term' from underlying heat and salt tendencies.

        Parameters
        ----------
        term : str
            key for tendency variable in the recipe

        Returns
        -------
        rho_tend_heat, rho_tend_salt
            The two distinct components contributing to the overall density tendency.

        Example
        --------
        >>> wmt = WaterMassTransformations(grid, recipe)
        >>> wmt.get_density()
        >>> (rho_tend_heat, rho_tend_salt) = wmt.rho_tend("diffusion")
        """

        # Either heat or salt tendency/flux may not be used
        rho_tend_heat, rho_tend_salt = None, None

        datadict = self.datadict("heat", term)
        if datadict is not None:
            heat_tend = calc_hlamdot_tendency(
                self.grid, datadict, h=self.Z_metrics["center"]
            )
            # Density tendency due to heat flux (kg/s/m^2)
            rho_tend_heat = -(self.grid._ds.alpha / self.cp) * heat_tend

        datadict = self.datadict("salt", term)
        if datadict is not None:
            salt_tend = calc_hlamdot_tendency(
                self.grid, datadict, h=self.Z_metrics["center"]
            )
            # Density tendency due to salt/salinity (kg/s/m^2)
            rho_tend_salt = self.grid._ds.beta * salt_tend

        return rho_tend_heat, rho_tend_salt

    def calc_hlamdot_and_lambda(self, lambda_name, term):
        """
        Get layer-integrated extensive tracer tendencies (* kg/m^2/s) and corresponding scalar field of lambda

        Parameters
        ----------
        lambda_name : str
            Specifies lambda
        term : str
            key for tendency variable in the recipe

        Returns
        ----------
        hlamdot, lam : xr.DataArray, xr.DataArray

        Example
        --------
        >>> wmt = WaterMassTransformations(grid, recipe)
        >>> (hlamdot, lam) = wmt.calc_hlamdot_and_lambda("heat", "diffusion")
        """

        lam_var = self.get_lambda_var(lambda_name)
        prebinned = all(
            [
                (c in self.grid.axes["Z"].coords.values())
                for c in [f"{lam_var}_l", f"{lam_var}_i"]
            ]
        )

        # Get layer-integrated potential temperature tendency
        # from tendency of heat (in W/m^2)
        if lambda_name == "heat":
            datadict = self.datadict("heat", term)
            if datadict is not None:
                hlamdot = (
                    calc_hlamdot_tendency(
                        self.grid, datadict, h=self.Z_metrics["center"]
                    )
                    / self.cp
                )
                lam = (
                    datadict["scalar"]
                    if not prebinned
                    else self.grid._ds[f"{lam_var}_l"]
                )

        # Get layer-integrated practical salinity tendency
        # from tendency of salt (in g/s/m^2)
        elif lambda_name == "salt":
            datadict = self.datadict("salt", term)
            if datadict is not None:
                hlamdot = calc_hlamdot_tendency(
                    self.grid, datadict, h=self.Z_metrics["center"]
                )
                lam = (
                    datadict["scalar"]
                    if not prebinned
                    else self.grid._ds[f"{lam_var}_l"]
                )

        # Get layer-integrated potential density tendencies (in kg/s/m^2)
        # from heat and salt, lambda = density
        # Here we want to output 2 separate components of the transformation rates:
        # (1) transformation due to heat tend, (2) transformation due to salt tend
        elif lambda_name in (
            [] if self.lambdas("density") is None else self.lambdas("density")
        ):
            lam = self.get_density(lambda_name)
            rhos = self.rho_tend(term)
            hlamdot = {}
            for idx, tend in enumerate(self._COMPONENTS):
                if rhos[idx] is not None:
                    hlamdot[tend] = (
                        rhos[idx] * self.rho_ref
                    )  # Is this correct for non-boussinesq case?
                elif rhos[idx] is None:
                    hlamdot[tend] = rhos[idx]

            if prebinned and not (self.rebin):
                lam = self.grid._ds[f"{lam_var}_l"]
            elif lam_var in self.grid._ds:
                lam = self.grid._ds[lam_var]

        elif lambda_name in self.tracer_dict.keys():
            datadict = self.datadict(lambda_name, term)
            if datadict is not None:
                hlamdot = calc_hlamdot_tendency(
                    self.grid, datadict, h=self.Z_metrics["center"]
                )
                lam = (
                    datadict["scalar"]
                    if not prebinned
                    else self.grid._ds[f"{lam_var}_l"]
                )

        else:
            raise ValueError(f"{lambda_name} is not available in the recipe.")

        try:
            return hlamdot, lam

        except NameError:
            return None, None

    def transform_hlamdot_term(
        self, lambda_name, term, bins=None, mask=None, integrate=True
    ):
        """
        Lazily compute extensive tendencies and transform them into lambda space
        along the vertical ("Z") dimension.

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'heat', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        term : str
            Specifies process term (e.g., 'boundary_forcing', 'vertical_diffusion', etc.). Use `self.available_processes()`
            to see a list of all available terms.
        bins : array like (default None)
            np.array with lambda values specifying the edges for each bin. If not specified, array will be automatically
            derived from the scalar field of lambda (e.g., temperature).
        mask : xr.DataArray (default: None)
            Boolean region mask (with same X and Y grid dimensions as `grid._ds` variables).
            If None, generate an all-True mask for domain-wide calculations.
        integrate : bool (default: True)
            Whether to integrate the result over ("X", "Y") dimensions.

        Returns
        -------
        transformed_term : xarray.DataArray
            Data array containing transformed tendencies.

        See also
        --------
        self.transformations_from_hlamdot, self.integrate_transformations
        """

        hlamdot, lam = self.calc_hlamdot_and_lambda(lambda_name, term)
        if hlamdot is None:
            return

        # Resolve the transformation method for *this call only*. A prebinned
        # target forces "xgcm" below; keeping it local avoids corrupting
        # `self.method` for subsequent calls (e.g. across terms or lambdas).
        method = self.method

        if method in ["default", "xhistogram"]:
            dim = [*self._horizontal_dims, self._zc] if integrate else [self._zc]
        else:
            dim = None

        # Fall back to the constructor-level mask when no per-call mask is given.
        mask = mask if mask is not None else self.mask
        if mask is not None:
            if type(hlamdot) is dict:
                hlamdot = {
                    k: v.where(mask, 0.0) if v is not None else None
                    for k, v in hlamdot.items()
                }
            else:
                hlamdot = hlamdot.where(mask, 0.0)

        if bins is None:
            bins = self.infer_bins(lam)
        bin_bounds = bins.values if isinstance(bins, xr.DataArray) else bins

        # If lambda is already a vertical coordinate, no need to use the 3D lambda for transformations
        lam_var = self.get_lambda_var(lambda_name)
        prebinned = all(
            [
                (c in self.grid.axes["Z"].coords.values())
                for c in [f"{lam_var}_l", f"{lam_var}_i"]
            ]
        )
        if prebinned and not (self.rebin):
            lam = lam.rename({lam.name: lam_var}).rename(lam_var)
            lam_i = self.grid._ds[f"{lam_var}_i"]
            method = "xgcm"
        else:
            if self._zc in lam.dims:
                lam_i = self.grid.interp(lam, "Z", padding="extend").rename(
                    f"{lam.name}_i"
                )
            else:
                lam_i = lam.broadcast_like(self.grid._ds[self._zc])

        if lambda_name in (
            [] if self.lambdas("density") is None else self.lambdas("density")
        ):
            hlamdot_transformed = xr.merge(
                [
                    self._transform_one(
                        hlamdot[component],
                        lam,
                        lam_i,
                        bin_bounds,
                        dim,
                        method,
                        integrate,
                        output_name=self._component_name(term, component),
                    )
                    for component in self._COMPONENTS
                    if hlamdot[component] is not None
                ]
            )
        else:
            hlamdot_transformed = self._transform_one(
                hlamdot,
                lam,
                lam_i,
                bin_bounds,
                dim,
                method,
                integrate,
                output_name=f"{term}",
            )
        return hlamdot_transformed

    def _transform_one(
        self, weights, lam, lam_i, bin_bounds, dim, method, integrate, output_name
    ):
        """
        Transform a single extensive-tendency field `weights` into lambda space along
        "Z" and normalize by bin width. Dispatches between the `xhistogram` (binning)
        and `xgcm` (conservative remapping) backends and, for `xgcm` with
        `integrate=True`, sums over the horizontal dimensions.

        This is the shared kernel for both the scalar-lambda path and each heat/salt
        component of the density-lambda path in `transform_hlamdot_term`.
        """
        if ((method == "default") and integrate) or (method == "xhistogram"):
            transformed = histogram(
                lam,
                bins=[bin_bounds],
                dim=dim,
                weights=weights.fillna(0.0),
                bin_dim_suffix="",
                # TEMPORARY FIX FOR https://github.com/xgcm/xhistogram/issues/16
                block_size=None,
            ).rename({lam.name: f"{lam.name}_l_target"})
        else:  # xgcm conservative remapping
            transformed = (
                self.grid.transform(
                    weights.fillna(0.0),
                    "Z",
                    target=bin_bounds,
                    target_data=lam_i,
                    method="conservative",
                )
                .fillna(0.0)
                .rename({lam_i.name: f"{lam.name}_l_target"})
            )
            if integrate:
                transformed = transformed.sum(self._horizontal_dims)
        return (transformed / np.diff(bin_bounds)).rename(output_name)

    def transformations_from_hlamdot(
        self, lambda_name, term=None, bins=None, mask=None, integrate=True
    ):
        """
        Lazily compute extensive tendencies, transform them into lambda space
        along the vertical ("Z") dimension, and optionally integrate along the
        horizontal dimensions ("X", "Y").

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'heat', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        term : str or list of str (default: None)
            Specifies process term (e.g., 'boundary_forcing', 'vertical_diffusion', etc.). Use `self.available_processes()`
            to list all terms. If None, defaults to list of all terms.
        bins : array like (default None)
            np.array with lambda values specifying the edges for each bin. If not specified, array will be automatically
            derived from the scalar field of lambda (e.g., temperature).
        mask : xr.DataArray (default: None)
            Boolean region mask (with same X and Y grid dimensions as `grid._ds` variables).
            If None, generate an all-True mask for domain-wide calculations.
        integrate : bool (default: True)
            Whether to integrate the result over ("X", "Y") dimensions.

        Returns
        -------
        transformations : xarray.Dataset
            Dataset containing components of water mass transformations, possibly grouped as
            specified by the arguments.

        See also
        --------
        self.integrate_transformations, self.map_transformations
        """

        if isinstance(term, str):
            terms = [term]
        elif isinstance(term, list):
            terms = term
        elif term is None:
            terms = self.available_processes()
        else:
            return

        wmts = []
        for term in terms:
            transformed_hlamdot = self.transform_hlamdot_term(
                lambda_name, term=term, bins=bins, mask=mask, integrate=integrate
            )
            if transformed_hlamdot is not None:
                wmts.append(-transformed_hlamdot)
            else:
                warnings.warn(
                    f"Process '{term}' for component {lambda_name} is unavailable."
                )
        return xr.merge(wmts)

    ### Helper function to groups terms based on density components (sum_components)
    ### and physical processes (group_processes)
    # Calculate the sum of grouped terms
    def _sum_terms(self, ds_terms, newterm, terms):
        # `ds_terms` is always the merged transformation Dataset produced upstream;
        # sum the subset of `terms` present in it into a new `newterm` variable.
        if not isinstance(ds_terms, xr.Dataset):
            return
        das = [ds_terms[term] for term in terms if term in ds_terms]
        if das:
            ds_terms[newterm] = sum(das)

    def _group_processes(self, hlamdot):
        if hlamdot is None:
            return
        for c in hlamdot.coords:
            lambda_key = self.get_lambda_key(c.split("_")[0])
            if lambda_key is not None:
                if lambda_key == "density":
                    suffixes = [self._component_suffix(c) for c in self._COMPONENTS] + [
                        ""
                    ]
                    budget = self.recipe["heat"]
                else:
                    suffixes = [""]
                    budget = self.recipe[lambda_key]
                for suffix in suffixes:
                    if "lhs" in budget:
                        self._sum_terms(
                            hlamdot,
                            f"kinematic_transformation{suffix}",
                            [f"{term}{suffix}" for term in budget["lhs"].keys()],
                        )
                    if "rhs" in budget:
                        self._sum_terms(
                            hlamdot,
                            f"material_transformation{suffix}",
                            [f"{term}{suffix}" for term in budget["rhs"].keys()],
                        )
        return hlamdot

    def _sum_components(self, hlamdot, group_processes=False):
        if hlamdot is None:
            return

        for proc in self.available_processes():
            proc_list = [
                self._component_name(proc, component) for component in self._COMPONENTS
            ]
            self._sum_terms(hlamdot, proc, proc_list)
        if group_processes:
            for proc in ["kinematic_transformation", "material_transformation"]:
                self._sum_terms(
                    hlamdot,
                    proc,
                    [
                        self._component_name(proc, component)
                        for component in self._COMPONENTS
                    ],
                )
        return hlamdot

    def map_transformations(
        self,
        lambda_name,
        term=None,
        group_processes=False,
        sum_components=True,
        bins=None,
        mask=None,
    ):
        """
        Lazily compute column-wise water mass transformations.

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'heat', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        term : str or list of str (default: None)
            Specifies process term (e.g., 'boundary_forcing', 'vertical_diffusion', etc.). Use `self.available_processes()`
            to list all terms. If None, defaults to list of all terms.
        group_processes: bool (default: False)
            Specify whether process terms are summed to categories forcing and diffusion.
        sum_components : bool (default: True)
            Specify whether heat and salt tendencies are summed together (True) or kept separated (False).
        bins : array like (default None)
            np.array with lambda values specifying the edges for each bin. If not specified, array will be automatically
            derived from the scalar field of lambda (e.g., temperature).
        mask : xr.DataArray (default: None)
            Boolean region mask (with same X and Y grid dimensions as `grid._ds` variables).
            If None, generate an all-True mask for domain-wide calculations.

        Returns
        -------
        transformations : xarray.Dataset
            Dataset containing components of water mass transformations, possibly grouped as
            specified by the arguments.

        Example
        --------
        >>> wmt = WaterMassTransformations(grid, recipe)
        >>> wmt_rates = wmt.map_transformations("heat", term=["diffusion"])

        See also
        --------
        self.transformations_from_hlamdot, self.integrate_transformations
        """

        # call the base function
        transformations = self.transformations_from_hlamdot(
            lambda_name,
            term=term,
            bins=bins,
            mask=mask,
            integrate=False,
        )

        # process this function arguments
        if sum_components:
            transformations = self._sum_components(
                transformations, group_processes=group_processes
            )
        if group_processes:
            transformations = self._group_processes(transformations)

        return transformations

    def integrate_transformations(
        self,
        lambda_name,
        term=None,
        group_processes=False,
        sum_components=True,
        bins=None,
        mask=None,
    ):
        """
        Lazily compute horizontally-integrated water mass transformations.

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'heat', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        term : str or list of str (default: None)
            Specifies process term (e.g., 'boundary_forcing', 'vertical_diffusion', etc.). Use `self.available_processes()`
            to list all terms. If None, defaults to list of all terms.
        group_processes: bool (default: False)
            Specify whether process terms are summed to categories forcing and diffusion.
        sum_components : bool (default: True)
            Specify whether heat and salt tendencies are summed together (True) or kept separated (False).
        bins : array like (default None)
            np.array with lambda values specifying the edges for each bin. If not specified, array will be automatically
            derived from the scalar field of lambda (e.g., temperature).
        mask : xr.DataArray (default: None)
            Boolean region mask (with same X and Y grid dimensions as `grid._ds` variables).
            If None, generate an all-True mask for domain-wide calculations.

        Returns
        -------
        transformations : xarray.Dataset
            Dataset containing components of water mass transformations, possibly grouped as
            specified by the arguments.

        Example
        --------
        >>> wmt = WaterMassTransformations(grid, recipe)
        >>> wmt_rates = wmt.map_transformations("heat", term=["diffusion"])

        See also
        --------
        self.transformations_from_hlamdot, self.map_transformations
        """

        # call the base function
        transformations = self.transformations_from_hlamdot(
            lambda_name,
            term=term,
            bins=bins,
            mask=mask,
            integrate=True,
        )

        # process this function arguments
        if sum_components:
            transformations = self._sum_components(
                transformations, group_processes=group_processes
            )
        if group_processes:
            transformations = self._group_processes(transformations)

        return transformations
