"""Loaders for the MOM6 example datasets used by the example notebooks.

The data live in Zenodo record 15420739 and are downloaded on demand into the
repository's ``data/`` directory. They are large (~1 GB each), so the download
happens once and is reused.
"""

import hashlib
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import xarray as xr
import xgcm

ZENODO_URL = "https://zenodo.org/record/15420739/files/"

# Resolve against this file rather than the working directory: notebooks are run
# from `examples/`, but the docs build, pytest and ad-hoc scripts are not, and a
# relative "../data" silently writes to the wrong place (or fails) for them.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# MD5s as published in the Zenodo record's file metadata. These guard against a
# truncated or corrupted download, which otherwise surfaces much later as an
# opaque decoding error from deep inside xarray/netCDF4. Update these only
# alongside a deliberate bump of the record version in the filenames below.
CHECKSUMS_MD5 = {
    "MOM6_global_example_sigma2_budgets_v0_0_6.nc": "12e4e552ecf8dcc7fe26db50d4df71dd",
    "MOM6_global_example_surface_fluxes_v0_0_7.nc": "ae10ebaff09788ac5be410550ef696fd",
}


def _md5(path, chunk_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path, file_name):
    """Check `path` against its Zenodo-published MD5, if one is known."""
    expected = CHECKSUMS_MD5.get(file_name)
    if expected is None:
        return
    actual = _md5(path)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path}: expected md5 {expected}, got {actual}. "
            "Delete the file to re-download, or update CHECKSUMS_MD5 if the data "
            "changed intentionally."
        )


def download_MOM6_example_data(file_name):
    """Download `file_name` from Zenodo into `data/`, unless already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination_path = DATA_DIR / file_name

    if destination_path.exists():
        print(
            f"File '{file_name}' already exists at {destination_path}. Skipping download."
        )
        _verify(destination_path, file_name)
        return str(destination_path)

    # Download to a temporary name so an interrupted transfer cannot leave a
    # truncated file behind that later runs would mistake for a complete one.
    tmp_path = destination_path.with_suffix(destination_path.suffix + ".part")
    print(f"File '{file_name}' being downloaded to {destination_path}.")
    try:
        with (
            urllib.request.urlopen(ZENODO_URL + file_name) as response,
            open(tmp_path, "wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)
    except (urllib.error.URLError, OSError) as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download '{file_name}' from {ZENODO_URL}: {e}"
        ) from e

    _verify(tmp_path, file_name)
    tmp_path.rename(destination_path)
    print(f"File '{file_name}' has completed download to {destination_path}.")
    return str(destination_path)


def load_MOM6_example_grid(file_name):
    destination_path = download_MOM6_example_data(file_name)
    ds = xr.open_dataset(destination_path, chunks=-1).fillna(0.0)
    return construct_grid(ds)


def load_MOM6_coarsened_diagnostics():
    file_name = "MOM6_global_example_sigma2_budgets_v0_0_6.nc"
    return load_MOM6_example_grid(file_name)


def load_MOM6_surface_diagnostics():
    file_name = "MOM6_global_example_surface_fluxes_v0_0_7.nc"
    return load_MOM6_example_grid(file_name)


def construct_grid(ds):
    if "sigma2_l" in ds:
        coords = {
            "X": {"center": "xh", "outer": "xq"},
            "Y": {"center": "yh", "outer": "yq"},
            "Z": {"center": "sigma2_l", "outer": "sigma2_i"},
        }
        padding = {"X": "periodic", "Y": "extend", "Z": "extend"}
    else:
        coords = {
            "X": {"center": "xh", "outer": "xq"},
            "Y": {"center": "yh", "outer": "yq"},
        }
        padding = {"X": "periodic", "Y": "extend"}

    metrics = {("X", "Y"): "areacello"}
    grid = xgcm.Grid(
        ds, coords=coords, metrics=metrics, padding=padding, autoparse_metadata=False
    )
    return grid
