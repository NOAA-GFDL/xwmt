"""
Real-data validation of multi-tile ("broadcast across tiles") water-mass
transformations on the ECCOv4r4 native 13-tile lat-lon-cap (LLC90) grid — the
motivating case for GitHub issue #59.

This test is **excluded from CI by default**: it downloads a multi-hundred-MB
subset of the ECCO example dataset and runs the full pipeline on a ~5M-cell grid,
which is far too heavy for the normal suite. Enable it explicitly with:

    XWMT_TEST_ECCO=1 pytest xwmt/tests/test_multitile_ecco.py -v

The data (record 21479854, v3.0.0 of the ECCO example dataset) is fetched from
Zenodo and checksum-pinned, mirroring `conftest.py`'s Baltic download. If the
files already exist locally, point the test at them to skip the download:

    XWMT_ECCO_DIR=/path/to/files XWMT_TEST_ECCO=1 pytest ...

What it asserts (the issue-#59 invariant): integrating transformations over the
full 13-tile grid equals the sum of the 13 single-tile integrations. Binning
into lambda space and summing over the horizontal are both linear in the tiles,
so the two must agree to machine precision — proving the `face` (tile) dimension
is correctly included in the horizontal reduction.
"""

import os
import hashlib
import urllib.request
import urllib.error

import numpy as np
import pytest
import xarray as xr
import xgcm

import xwmt

pytestmark = pytest.mark.skipif(
    not os.environ.get("XWMT_TEST_ECCO"),
    reason="set XWMT_TEST_ECCO=1 to run the large real-ECCO multi-tile test",
)

ZENODO_RECORD = "21479854"  # ECCO example dataset v3.0.0
ECCO_FILES = {
    "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc": "2663a7e86d7a0e6f7ddf84124c8376a6",
    "OCEAN_TEMPERATURE_SALINITY_mon_mean_2010_ECCO_V4r4_native_llc0090.nc": "5d7a4f0f8ff05d319e1b0f4a0fffadf1",
    "OCEAN_AND_ICE_SURFACE_HEAT_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc": "248b53ad82422aed022bc9fe24a72cce",
}

# Canonical xgcm face_connections for the 13-tile LLC90 grid (from the xgcm
# ECCOv4 example / xbudget's load_example_ecco_grid), keyed by the 'tile' dim.
LLC90_FACE_CONNECTIONS = {
    "tile": {
        0: {"X": ((12, "Y", False), (3, "X", False)), "Y": (None, (1, "Y", False))},
        1: {
            "X": ((11, "Y", False), (4, "X", False)),
            "Y": ((0, "Y", False), (2, "Y", False)),
        },
        2: {
            "X": ((10, "Y", False), (5, "X", False)),
            "Y": ((1, "Y", False), (6, "X", False)),
        },
        3: {"X": ((0, "X", False), (9, "Y", False)), "Y": (None, (4, "Y", False))},
        4: {
            "X": ((1, "X", False), (8, "Y", False)),
            "Y": ((3, "Y", False), (5, "Y", False)),
        },
        5: {
            "X": ((2, "X", False), (7, "Y", False)),
            "Y": ((4, "Y", False), (6, "Y", False)),
        },
        6: {
            "X": ((2, "Y", False), (7, "X", False)),
            "Y": ((5, "Y", False), (10, "X", False)),
        },
        7: {
            "X": ((6, "X", False), (8, "X", False)),
            "Y": ((5, "X", False), (10, "Y", False)),
        },
        8: {
            "X": ((7, "X", False), (9, "X", False)),
            "Y": ((4, "X", False), (11, "Y", False)),
        },
        9: {"X": ((8, "X", False), None), "Y": ((3, "X", False), (12, "Y", False))},
        10: {
            "X": ((6, "Y", False), (11, "X", False)),
            "Y": ((7, "Y", False), (2, "X", False)),
        },
        11: {
            "X": ((10, "X", False), (12, "X", False)),
            "Y": ((8, "Y", False), (1, "X", False)),
        },
        12: {"X": ((11, "X", False), None), "Y": ((9, "Y", False), (0, "X", False))},
    }
}

# One process term is enough to exercise tile broadcasting: the surface heat flux
# (a 2D boundary forcing that xwmt expands into the outcropping layer).
BUDGET = {
    "mass": {"lambda": None, "thickness": "thkcello", "lhs": {}, "rhs": {}},
    "heat": {"lambda": "THETA", "lhs": {}, "rhs": {"boundary_forcing": "TFLUX"}},
    "salt": {"lambda": None, "lhs": {}, "rhs": {}},
}
THETA_BINS = np.linspace(
    -2.0, 32.0, 69
)  # degC bin edges (fixed => full == sum-of-tiles)


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _resolve(name, md5):
    """Return a local path to `name`, using XWMT_ECCO_DIR if set, else downloading
    from Zenodo. Skips (not fails) if the file is unavailable offline."""
    local_dir = os.environ.get("XWMT_ECCO_DIR")
    if local_dir:
        p = os.path.join(local_dir, name)
        if os.path.isfile(p) and _md5(p) == md5:
            return p
    p = name  # download into the working directory (gitignored *.nc)
    if not (os.path.isfile(p) and _md5(p) == md5):
        url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"
        try:
            urllib.request.urlretrieve(url, p)
        except (urllib.error.URLError, OSError) as e:
            pytest.skip(f"could not download {name}: {e}")
    if _md5(p) != md5:
        pytest.fail(f"checksum mismatch for {name}")
    return p


@pytest.fixture(scope="module")
def ecco_dataset():
    """Merged ECCO dataset (one month) with the thickness metric xwmt needs."""
    paths = {n: _resolve(n, m) for n, m in ECCO_FILES.items()}
    geom = xr.open_dataset(paths["GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"])
    ts = xr.open_dataset(
        paths["OCEAN_TEMPERATURE_SALINITY_mon_mean_2010_ECCO_V4r4_native_llc0090.nc"]
    ).isel(time=0)
    hf = xr.open_dataset(
        paths[
            "OCEAN_AND_ICE_SURFACE_HEAT_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc"
        ]
    ).isel(time=0)
    ds = xr.merge([geom, ts[["THETA", "SALT"]], hf[["TFLUX"]]], compat="override")
    # Cell-center thickness (m): layer thickness x open-cell fraction.
    ds["thkcello"] = (ds["drF"] * ds["hFacC"]).transpose("k", "tile", "j", "i")
    # xwmt uses the Z center/outer *coordinate values* as the vertical remap target,
    # so they must be the (strictly monotonic) physical depths. ECCO's native k/k_p1
    # are integer indices (k[0]==k_p1[0]==0), which are non-monotonic once extended;
    # give those dims their depth coordinates (Z, Zp1) instead.
    ds = ds.assign_coords(k=("k", ds["Z"].values), k_p1=("k_p1", ds["Zp1"].values))
    return ds


def _build_grid(ds, multitile):
    coords = {
        "X": {"center": "i"},
        "Y": {"center": "j"},
        "Z": {"center": "k", "outer": "k_p1"},
    }
    kw = {"face_connections": LLC90_FACE_CONNECTIONS} if multitile else {}
    return xgcm.Grid(
        ds,
        coords=coords,
        metrics={("X", "Y"): ["rA"]},
        padding={"Z": "extend"},
        autoparse_metadata=False,
        **kw,
    )


def _integrate(grid, method):
    wmt = xwmt.WaterMassTransformations(grid, BUDGET, eos=None, method=method)
    return wmt.integrate_transformations("heat", bins=THETA_BINS, sum_components=False)[
        "boundary_forcing"
    ]


@pytest.mark.parametrize("method", ["xhistogram", "xgcm"])
def test_ecco_llc90_broadcast_equivalence(ecco_dataset, method):
    ds = ecco_dataset

    full = _integrate(_build_grid(ds, multitile=True), method)
    assert full.dims == ("THETA_l_target",), full.dims  # tiles reduced away

    per_tile = sum(
        _integrate(_build_grid(ds.isel(tile=n), multitile=False), method)
        for n in range(ds.sizes["tile"])
    )

    # Non-trivial, and full-grid == sum over the 13 tiles to machine precision.
    assert np.any(np.abs(np.nan_to_num(full.values)) > 0.0)
    assert np.allclose(
        np.nan_to_num(full.values),
        np.nan_to_num(per_tile.values),
        rtol=1e-9,
        atol=1e-9,
    )
