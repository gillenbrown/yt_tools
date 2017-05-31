from yt_tools import galaxy

import pytest
import yt
# -----------------------------------------------------------------------------

# test initialization of galaxy class

# -----------------------------------------------------------------------------

kpc = yt.units.kpc
file_loc = "../../../google_drive/research/simulation_outputs/" \
           "fiducial_destroy/continuous_a0.2406.art"
ds = yt.load(file_loc)

def test_gal_dataset_typing():
    with pytest.raises(TypeError):
        galaxy.Galaxy(None, [1, 2, 3] * kpc, 10 * kpc)
    galaxy.Galaxy(ds, [1, 2, 3] * kpc, 10 * kpc) # will not raise error

def test_gal_center_unit_check():
    with pytest.raises(TypeError):
        galaxy.Galaxy(ds, [1, 2, 3], 10 * kpc)

def test_gal_center_length_check():
    with pytest.raises(ValueError):
        galaxy.Galaxy(ds, [1, 2] * kpc, 10*kpc)
    with pytest.raises(ValueError):
        galaxy.Galaxy(ds, [1, 2, 3, 4] * kpc, 10*kpc)

def test_gal_radius_unit_check():
    with pytest.raises(TypeError):
        galaxy.Galaxy(ds, [1, 2, 3]*kpc, 10)