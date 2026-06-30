import pytest
import numpy as np


def test_analytical_errors(helpers):
    passing = []
    for lam, tend, loc, lay in zip(
        helpers.exps["lam_profile"],
        helpers.exps["extensive_tendency"],
        helpers.exps["local_wmt"],
        helpers.exps["layer_wmt"],
    ):
        # pick discretizations with no shared multiples so levels do not coincide
        wmt = helpers.idealized_transformations(tend, lam, Nz=5**6, Nlam=2**5)
        err = helpers.mean_absolute_relative_errors(wmt, loc, lay)[0]
        passing.append(err < 1.0e-3)
    assert np.all(passing)


def test_analytical_error_convergence(helpers):
    passing = []
    for lam, tend, loc, lay in zip(
        helpers.exps["lam_profile"],
        helpers.exps["extensive_tendency"],
        helpers.exps["local_wmt"],
        helpers.exps["layer_wmt"],
    ):
        wmt_lo = helpers.idealized_transformations(tend, lam, Nz=5**4, Nlam=2**4)
        wmt_hi_z = helpers.idealized_transformations(tend, lam, Nz=5**6, Nlam=2**4)
        wmt_hi_lam = helpers.idealized_transformations(tend, lam, Nz=5**4, Nlam=2**6)
        err_lo = helpers.mean_absolute_relative_errors(wmt_lo, loc, lay)
        err_hi_z = helpers.mean_absolute_relative_errors(wmt_hi_z, loc, lay)
        err_hi_lam = helpers.mean_absolute_relative_errors(wmt_hi_lam, loc, lay)
        passing.append(
            (err_hi_z[0] < err_lo[0])
            and (err_hi_z[1] < err_lo[1])
            and (err_hi_lam[1] < err_lo[1])
        )
    assert np.all(passing)
