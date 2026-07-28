Introduction
===========================

The Water Mass Transformation (WMT) framework was initially put forward by `Walin (1982) <https://doi.org/10.1111/j.2153-3490.1982.tb01806.x>`_ to describe the relationship between surface heat fluxes and interior ocean circulation (building on his earlier paper on salinity transformations in estuaries). A series of studies further refined the framework to include the effect of haline-driven buoyancy forcing (e.g., `Tziperman 1986 <https://doi.org/10.1175/1520-0485(1986)016%3C0680:OTROIM%3E2.0.CO;2>`_; `Speer and Tziperman, 1992 <https://doi.org/10.1175/1520-0485(1992)022%3C0093:ROWMFI%3E2.0.CO;2>`_) and account for the role of interior mixing (e.g., `Nurser et al., 1999 <https://doi.org/10.1175/1520-0485(1999)029%3C1468:DWMFFA%3E2.0.CO;2>`_; `Iudicone et al., 2008 <https://journals.ametsoc.org/view/journals/phoc/38/7/2007jpo3464.1.xml>`_). A comprehensive overview of past studies in WMT and details of how WMT is derived from diapycnal processes can be found in `Groeskamp et al. 2019 <https://doi.org/10.1146/annurev-marine-010318-095421>`_.

A detailed account of best practices for computing full water mass budgets in finite-volume ocean models is given by `Drake et al. 2025 <https://doi.org/10.1029/2024MS004383>`_, which was written alongside this package. This package handles the calculation of water mass transformations, generalizing an earlier version of the package that was limited to surface-forced transformations and developed alongside `Tesdal et al. (2023) <https://doi.org/10.1029/2022JC019105>`_. A companion python package, `xwmb`, handles the other terms in the water mass budget: water mass tendencies, lateral overturning transports, boundary mass fluxes, and spurious numerical mixing.

Package Objectives
===========================
The goal of this package is to provide various WMT routines to derive key metrics related to water masses in the ocean and the rates at which they are transformed.

Disclaimer
===========================
`xwmt` does not employ any checks to ensure conservation of heat and salt. It is the user's responsibility to ensure that budgets for heat and salt are properly closed in the datasets. Improperly conserved fields will yield incorrect results.

Nor does `xwmt` check that the transformation rate in a given :math:`\lambda` bin is adequately sampled. In the tails of the water mass distribution, a bin may be populated by only a handful of grid cells; the resulting rate is not a converged estimate of the transformation, but it plots as a perfectly respectable-looking curve and is a common source of confusion. The optional ``N_min`` argument masks bins sampled by fewer than a given number of grid cells, and ``WaterMassTransformations.count_cells_per_bin`` reports the counts so that a sensible threshold can be chosen; see the :doc:`Masking under-resolved bins <examples/masking_undersampled_bins>` example.

