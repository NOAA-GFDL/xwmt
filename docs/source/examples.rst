Examples
========

These notebooks are meant to be read in order. Each builds on the ones before it,
and they get progressively more realistic — and more expensive to run.

If you are new to ``xwmt``, start with the **quickstart**: it needs no data and runs
in seconds.

.. toctree::
   :maxdepth: 1

   Quickstart <examples/quickstart>
   Closed transformation budgets in MOM6 <examples/closed_transformation_budget>
   Case study: North Atlantic Deep Water <examples/north_atlantic_deep_water>
   Surface transformation from surface fields <examples/swmt_decomposition>
   Bring your own model <examples/bring_your_own_model>
   Surface transformation from CMIP6 <examples/swmt_from_cmip>

What each one is for
--------------------

.. list-table::
   :header-rows: 1
   :widths: 32 42 13 13

   * - Notebook
     - Who it's for
     - Data
     - Network
   * - :doc:`Quickstart <examples/quickstart>`
     - Your first transformation curve, on a synthetic basin
     - none
     - no
   * - :doc:`Closed transformation budgets <examples/closed_transformation_budget>`
     - Closed, comprehensive MOM6 budgets and the ``xbudget`` recipe
     - 1.08 GB
     - first run
   * - :doc:`North Atlantic Deep Water <examples/north_atlantic_deep_water>`
     - A case study: where, and by what process, deep water forms
     - shared with above
     - first run
   * - :doc:`Surface transformation <examples/swmt_decomposition>`
     - When you have surface fields and no interior information
     - 1.20 GB
     - first run
   * - :doc:`Bring your own model <examples/bring_your_own_model>`
     - Using ``xwmt`` with a model that isn't MOM6
     - none
     - no
   * - :doc:`Surface transformation from CMIP6 <examples/swmt_from_cmip>`
     - Reading the CMIP6 archive directly
     - streamed
     - every run

The MOM6 example data are downloaded on demand from
`Zenodo record 15420739 <https://zenodo.org/record/15420739>`_ into the repository's
``data/`` directory — once, and verified against the checksums published in the
record. The two MOM6 notebooks in the middle of the sequence share a single file, so
running both costs one download rather than two.
