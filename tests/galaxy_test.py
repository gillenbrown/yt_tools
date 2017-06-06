from yt_tools import galaxy
from yt_tools import kde
from yt_tools import utils

import pytest
import yt
import numpy as np
# -----------------------------------------------------------------------------

# test initialization of galaxy class

# -----------------------------------------------------------------------------

kpc = yt.units.kpc
pc = yt.units.pc
file_loc = "../../../google_drive/research/simulation_outputs/" \
           "fiducial_destroy/continuous_a0.2406.art"
ds = yt.load(file_loc)
# then find the densest region, just to have a real galaxy somewhere
best_loc = yt.YTArray(ds.all_data().argmax("density"))

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

# -----------------------------------------------------------------------------

# Fixtures

# -----------------------------------------------------------------------------

@pytest.fixture
def gal():
    return galaxy.Galaxy(ds, best_loc, 17 * pc)

@pytest.fixture
def real_gal():  # has larger radius to actually include everythign we need to
    return galaxy.Galaxy(ds, best_loc, 1000 * pc, j_radius=30 * pc)

# -----------------------------------------------------------------------------

# test the KDE process

# -----------------------------------------------------------------------------

def test_kde_creation_simple_error_checking(gal):
    with pytest.raises(ValueError):
        gal._create_kde_object(dimension=4)
    with pytest.raises(ValueError):
        gal._create_kde_object(dimension=1)
    with pytest.raises(ValueError):
        gal._create_kde_object(quantity="test", dimension=3)
    # then test what should work. We need to add a disk to make cylindrical work
    gal.add_disk()
    gal._create_kde_object()
    gal._create_kde_object(dimension=2)
    gal._create_kde_object(dimension=3)
    gal._create_kde_object(quantity="mass")
    gal._create_kde_object(quantity="Z")

def test_kde_creation_no_disk_error_checking(gal):
    with pytest.raises(RuntimeError):
        gal._create_kde_object(dimension=2)
    gal.add_disk()
    gal._create_kde_object(dimension=2)  # will work now

def test_kde_creation_dimensions(gal):
    assert gal._create_kde_object(dimension=3).dimension == 3
    gal.add_disk()  # so we can get cartesian
    assert gal._create_kde_object(dimension=2).dimension == 2

def test_kde_profile_units(gal):
    with pytest.raises(TypeError):
        gal.kde_profile(spacing=0.5, outer_radius=100 * kpc, dimension=3)
    with pytest.raises(TypeError):
        gal.kde_profile(spacing=0.5 * kpc, outer_radius=100, dimension=3)
    gal.kde_profile(spacing=0.5 * kpc, outer_radius=100 * kpc,
                    dimension=3)  # no error

def test_kde_profile_coords_error_checking(gal):
    with pytest.raises(ValueError):
        gal.kde_profile(dimension="sdf")
    with pytest.raises(RuntimeError):  # no disk
        gal.kde_profile(dimension=2)
    gal.add_disk()
    gal.kde_profile(dimension=2)  # no error now.
    gal.kde_profile(dimension=3)  # no error
    gal.kde_profile()  # no error

def test_kde_profile_quantity_check(gal):
    # we need spherical on all of this to avoid runtime errors from no disk
    gal.kde_profile(dimension=3)
    gal.kde_profile("MASS", dimension=3)
    gal.kde_profile("Mass", dimension=3)
    gal.kde_profile("mass", dimension=3)
    gal.kde_profile("Z", dimension=3)
    with pytest.raises(ValueError):
        gal.kde_profile("test", dimension=3)

def test_kde_profile_results_exist_and_right_length(gal):
    """Check whether the results exist where they should and have right size."""
    # we need spherical on all of this to avoid runtime errors from no disk.
    gal.kde_profile("MASS", dimension=3,
                    spacing=1.0 * pc, outer_radius=2.0 * pc)
    gal.kde_profile("Z", dimension=3,
                    spacing=1.0 * pc, outer_radius=2.0 * pc)
    gal.add_disk()
    gal.kde_profile("MASS", dimension=2,
                    spacing=1.0 * pc, outer_radius=2.0 * pc)
    gal.kde_profile("Z", dimension=2,
                    spacing=1.0 * pc, outer_radius=2.0 * pc)
    assert len(gal.densities["mass_kde_3D"]) == 2
    assert len(gal.densities["mass_kde_2D"]) == 2
    assert len(gal.densities["z_kde_3D"]) == 2
    assert len(gal.densities["z_kde_2D"]) == 2

def test_kde_profile_results_reasonable(gal):
    """Check whether the results have roughtly correct values."""
    # we need spherical on all of this to avoid runtime errors from no disk.
    gal.kde_profile("MASS", dimension=3,
                    spacing=40.0 * pc, outer_radius=50.0 * pc)
    gal.kde_profile("Z", dimension=3,
                    spacing=40.0 * pc, outer_radius=50.0 * pc)
    gal.add_disk()
    gal.kde_profile("MASS", dimension=2,
                    spacing=40.0 * pc, outer_radius=50.0 * pc)
    gal.kde_profile("Z", dimension=2,
                    spacing=40.0 * pc, outer_radius=50.0 * pc)
    # there should be high density at the center here
    assert gal.densities["mass_kde_3D"][0] > 10**3
    assert gal.densities["mass_kde_2D"][0] > 10**5
    # the cylindrical should be a higher value than the spherical, since it
    # only is in 2D, not three
    assert gal.densities["mass_kde_2D"][0] > \
           gal.densities["mass_kde_3D"][0]
    # the stellar density should decrease outwards
    assert gal.densities["mass_kde_3D"][0] > \
           gal.densities["mass_kde_3D"][1]
    assert gal.densities["mass_kde_2D"][0] > \
           gal.densities["mass_kde_2D"][1]
    # metallicity should be between zero and one
    assert 0 < gal.densities["z_kde_3D"][0] < 1
    assert 0 < gal.densities["z_kde_2D"][0] < 1

# -----------------------------------------------------------------------------

# test the disk operations

# -----------------------------------------------------------------------------

def test_add_disk_units(gal):
    gal.add_disk()  # default parameters should work
    gal.add_disk(20 * pc, 20 * pc, 20 * pc)
    with pytest.raises(TypeError):
        gal.add_disk(j_radius=10)
    with pytest.raises(TypeError):
        gal.add_disk(disk_radius=10)
    with pytest.raises(TypeError):
        gal.add_disk(disk_height=10)

def test_add_disk_result_type(gal):
    """We should have an actual disk when done. """
    assert gal.disk is None
    gal.add_disk()
    assert isinstance(gal.disk, yt.data_objects.selection_data_containers.YTDisk)

def test_add_disk_properties(gal):
    """Check that the disk has the properties we want. """
    disk_height = 29.3 * pc
    disk_radius = 22.5 * pc
    gal.add_disk(disk_radius=disk_radius, disk_height=disk_height)
    assert gal.disk.height == disk_height
    assert gal.disk.radius == disk_radius
    # test that it's not in the default orientation
    assert not np.array_equal(gal.disk.get_field_parameter("normal"), [0, 0, 1])

def test_add_disk_kde_creation(gal):
    assert gal._star_kde_mass_2d is None
    assert gal._star_kde_metals_2d is None
    gal.add_disk()
    assert isinstance(gal._star_kde_mass_2d, kde.KDE)
    assert isinstance(gal._star_kde_metals_2d, kde.KDE)

# -----------------------------------------------------------------------------
#
# test the inclusion of the structural properties
#
# -----------------------------------------------------------------------------

def test_nsc_radius_units_and_mass_and_axis_ratios(real_gal):
    """Test several things about the NSC. I am combining a lot of things into
    one test since the real_gal takes a long time to initialize, since it has
    to do the KDE process. """
    utils.test_for_units(real_gal.nsc_radius, "NSC radius")
    assert 0 * pc < real_gal.nsc_radius < real_gal.sphere.radius

    # Test that the NSC mass is less than the total galaxy mass
    assert real_gal.stellar_mass(nsc=False) > real_gal.stellar_mass(nsc=True)

    # Test that the axis ratios for the NSC look reasonable.
    assert real_gal.nsc_axis_ratios.b_over_a < 1.0




