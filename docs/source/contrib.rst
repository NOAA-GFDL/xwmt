Contributing to the Package
===========================

Reporting Issues
----------------
If you uncover a bug or issue with the code, please open an issue through the GitHub site:  https://github.com/NOAA-GFDL/xwmt

Developing New Routines
-----------------------
Pull requests for new routines and code are welcome. Please note that **xwmt** is in the public domain and licensed and/or copyrighted material is not acceptable.

Creating a development environment
----------------------------------

.. code-block:: bash

    conda env update -f docs/environment.yml
    conda activate docs_env_xwmt
    pip install -e ".[dev]"

The ``[dev]`` extra installs the testing and linting tools (``pytest``,
``black``, ``pylint``, ``netcdf4``); use ``[test]`` for just the testing tools.

Running the tests
-----------------

.. code-block:: bash

    pytest -v

The functional tests download a small MOM6 test dataset on first run (cached
locally and integrity-checked); they skip cleanly when offline.

Code style
----------

Code is formatted with `black <https://black.readthedocs.io>`_, which the CI
enforces. Before committing:

.. code-block:: bash

    black xwmt/

Locally building the documentation
----------------------------------

.. code-block:: bash

    conda activate docs_env_xwmt
    rm -rf docs/_build
    sphinx-build -W -b html docs/source docs/_build/html