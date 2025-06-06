Introduction
===========================

The Water Mass Transformation (WMT) framework was initially put forward by `Walin (1982) <https://doi.org/10.1111/j.2153-3490.1982.tb01806.x>`_ to describe the relationship between surface heat fluxes and interior ocean circulation. A series of studies further refined the framework to include the effect of haline-driven buoyancy forcing (e.g., `Tziperman 1986 <https://doi.org/10.1175/1520-0485(1986)016%3C0680:OTROIM%3E2.0.CO;2>`_; `Speer and Tziperman, 1992 <https://doi.org/10.1175/1520-0485(1992)022%3C0093:ROWMFI%3E2.0.CO;2>`_) and account for the role of interior mixing (e.g., ` Nurser et al., 1999 <https://doi.org/10.1175/1520-0485(1999)029%3C1468:DWMFFA%3E2.0.CO;2>`_; `Iudicone et al., 2008 <https://journals.ametsoc.org/view/journals/phoc/38/7/2007jpo3464.1.xml>`_). A comprehensive overview of past studies in WMT and details of how WMT is derived from diapycnal processes can be found in `Groeskamp et al. 2019 <https://doi.org/10.1146/annurev-marine-010318-095421>`_).


Package Objectives
===========================
The goal of this package is to provide various WMT routines to derive key metrics related to water masses in the ocean.


Required Input
===========================
The required input for the package is a xarray dataset including the necessary scalar field (e.g., ocean temperature, salinity or density) used for water mass classification and surface fluxes of heat, freshwater and salt. Varibale naming follows CMOR convention.
- `tos`: Sea Surface Temperature (units: degC)
> Temperature of upper boundary of the liquid ocean, including temperatures below sea-ice and floating ice shelves.
- `sos`: Sea Surface Salinity (units: 0.001) 
> Sea water salinity is the salt content of sea water, often on the Practical Salinity Scale of 1978. However, the unqualified term 'salinity' is generic and does not necessarily imply any particular method of calculation. The units of salinity are dimensionless and the units attribute should normally be given as 1e-3 or 0.001 i.e. parts per thousand. 
- `hfds`: Downward Heat Flux at Sea Water Surface (units: W m-2)
> This is the net flux of heat entering the liquid water column through its upper surface (excluding any "flux adjustment").
- `wfo`: Water Flux into Sea Water (units: kg m-2 s-1)
> Computed as the water flux into the ocean divided by the area of the ocean portion of the grid cell. This is the sum *wfonocorr* and *wfcorr*.
- `sfdsi`: Downward Sea Ice Basal Salt Flux (units: kg m-2 s-1)
> This field is physical, and it arises since sea ice has a nonzero salt content, so it exchanges salt with the liquid ocean upon melting and freezing.

Surface WMT using `swmt` class
==============================
The first step is to initialize the class by creating an object. This object includes all the calculations for surface WMT (`swmt`) and full 3D WMT (`wmt`). Since we use here surface fluxes, we will use the `swmt` class.
.. math::
  swmt_class = swmt(ds)

The `swmt` class object includes multiple functions to look at the relevant data. The most common function is `.G()` which is the WMT along $\lambda$ (i.e., $G(\lambda)$). Here, we need to define $\lambda$. For example, 'theta' for potential temperature ($\theta$) or 'sigma0' for potential density referenced at 0 dbar ($\sigma_0$). You can use command `.lambdas()` for a list of available $\lambda$'s. Here, we will go with $\sigma_0$. This is all you need, but if you want to define the size of the bins you can do that with the argument `bin`.

Disclaimer
===========================
`xwmt` does not employ any checks to ensure conservation of heat and salt. It is the user's responsibility to ensure that budgets for heat and salt are properly closed in the datasets. Improperly conserved fields will yield incorrect results.

