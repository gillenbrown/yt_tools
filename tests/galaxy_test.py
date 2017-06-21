from yt_tools import galaxy
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

def test_gal_id_generator():
    """This has to be run first, since it depends on the initial state of
    the id counter. """
    gal_0 = galaxy.Galaxy(ds, best_loc, 10*kpc)
    assert gal_0.id == 101
    gal_1 = galaxy.Galaxy(ds, best_loc, 10 * kpc)
    assert gal_1.id == 102
    gal_id = galaxy.Galaxy(ds, best_loc, 10 * kpc, 314)
    assert gal_id.id == 314

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

@pytest.fixture
def read_in_gal():
    file = open("./real_gal_save.txt", "r")
    gal =  galaxy.read_gal(ds, file)
    file.close()
    return gal

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
    assert gal.disk_kde is None
    gal.add_disk()
    assert isinstance(gal.disk_kde,
                      yt.data_objects.selection_data_containers.YTDisk)

    assert gal.disk_nsc is None
    gal.add_disk(disk_type="nsc")
    assert isinstance(gal.disk_nsc,
                      yt.data_objects.selection_data_containers.YTDisk)

    with pytest.raises(ValueError):
        gal.add_disk(disk_type="wer")

def test_add_disk_properties(gal):
    """Check that the disk has the properties we want. """
    disk_height = 29.3 * pc
    disk_radius = 22.5 * pc
    gal.add_disk(disk_radius=disk_radius, disk_height=disk_height)
    assert gal.disk_kde.height == disk_height
    assert gal.disk_kde.radius == disk_radius
    # test that it's not in the default orientation
    assert not np.allclose(gal.disk_kde.get_field_parameter("normal"),
                           [0, 0, 1])

# def test_add_disk_kde_creation(gal):
#     assert gal._star_kde_mass_2d is None
#     assert gal._star_kde_metals_2d is None
#     gal.add_disk()
#     assert isinstance(gal._star_kde_mass_2d, kde.KDE)
#     assert isinstance(gal._star_kde_metals_2d, kde.KDE)

# -----------------------------------------------------------------------------
#
# test the inclusion of the structural properties
#
# -----------------------------------------------------------------------------

def test_real_nsc_existence(read_in_gal):
    utils.test_for_units(read_in_gal.nsc_radius, "NSC radius")
    assert 0 * pc < read_in_gal.nsc_radius < read_in_gal.sphere.radius

def test_real_nsc_disk_attributes(read_in_gal):
    assert np.isclose(read_in_gal.disk_nsc.radius.in_units("pc").value,
                      2 * read_in_gal.nsc_radius.in_units("pc").value)
    assert np.isclose(read_in_gal.disk_nsc.height.in_units("pc").value,
                      2 * read_in_gal.nsc_radius.in_units("pc").value)
    assert np.allclose(read_in_gal.disk_nsc.center.in_units("pc").value,
                       read_in_gal.center.in_units("pc").value)

def test_real_nsc_stellar_mass(read_in_gal):
    # Test that the NSC mass is less than the total galaxy mass
    assert read_in_gal.stellar_mass(nsc=False) > \
           read_in_gal.stellar_mass(nsc=True)

def test_real_nsc_radius_cut(read_in_gal):
    # test that the NSC indices actually pick up the right objects
    radius_key = ('STAR', 'particle_position_spherical_radius')
    for container, idx in zip([read_in_gal.sphere, read_in_gal.disk_kde,
                               read_in_gal.disk_nsc],
                              [read_in_gal.nsc_idx_sphere,
                               read_in_gal.nsc_idx_disk_kde,
                               read_in_gal.nsc_idx_disk_nsc]):
        radii = container[radius_key]
        assert np.max(radii[idx]) < read_in_gal.nsc_radius
        # then get the indices not in the nsc
        all_idx_set = set(range(len(radii)))
        nsc_idx_set = set(idx)
        non_nsc_idx = list(all_idx_set.difference(nsc_idx_set))
        assert np.min(radii[non_nsc_idx]) > read_in_gal.nsc_radius

def test_real_nsc_axis_ratios(read_in_gal):
    # Test that the axis ratios for the NSC look reasonable.
    assert read_in_gal.nsc_axis_ratios.b_over_a < 1.0

def test_nsc_rotation_units(read_in_gal):
    # test that the rotation on the NSC has the right units
    utils.test_for_units(read_in_gal.mean_rot_vel, "rotational velocity")
    utils.test_for_units(read_in_gal.nsc_3d_sigma, "sigma")
    read_in_gal.mean_rot_vel.in_units("km/s")  # throw error if not compatible
    read_in_gal.nsc_3d_sigma.in_units("km/s")  # throw error if not compatible

def test_nsc_abundances(read_in_gal):
    # then check that the abundances exist and are not identical with each other
    assert -5 < read_in_gal.nsc_abundances.z_on_h_total() < 5
    assert -5 < read_in_gal.gal_abundances.z_on_h_total() < 5
    assert -5 < read_in_gal.nsc_abundances.x_on_h_total("Fe") < 5
    assert -5 < read_in_gal.gal_abundances.x_on_h_total("Fe") < 5
    assert np.isclose(read_in_gal.nsc_abundances.x_on_fe_total("Fe"), 0)
    assert np.isclose(read_in_gal.gal_abundances.x_on_fe_total("Fe"), 0)
    assert -5 < read_in_gal.nsc_abundances.x_on_fe_total("Na") < 5
    assert -5 < read_in_gal.gal_abundances.x_on_fe_total("Na") < 5
    assert not np.isclose(read_in_gal.nsc_abundances.z_on_h_total(),
                          read_in_gal.gal_abundances.z_on_h_total())
    assert not np.isclose(read_in_gal.nsc_abundances.x_on_h_total("Na"),
                          read_in_gal.gal_abundances.x_on_h_total("Na"))
    assert not np.isclose(read_in_gal.nsc_abundances.x_on_fe_total("Na"),
                          read_in_gal.gal_abundances.x_on_fe_total("Na"))

# -----------------------------------------------------------------------------
#
# test the reading and writing
#
# -----------------------------------------------------------------------------

def test_reading_writing(read_in_gal):
    """The only thing we need is that the object needs to be the same after
    we write then read it in. There is a lot of checking here, though."""
    file = open("./real_gal_save.txt", "w")
    read_in_gal.write(file)
    file.close()

    file = open("./real_gal_save.txt", "r")
    new_gal = galaxy.read_gal(ds, file)

    # then compare things. First basic stuff:
    assert read_in_gal.id == new_gal.id
    assert np.allclose(read_in_gal.center.in_units("pc").value,
                       new_gal.center.in_units("pc").value)
    # ^ the .value is needed to make yt arrays play nice with allclose
    assert read_in_gal.radius == new_gal.radius
    assert read_in_gal.ds == new_gal.ds

    # spheres should be the same
    assert np.allclose(read_in_gal.sphere.center.in_units("pc").value,
                       new_gal.sphere.center.in_units("pc").value)
    assert read_in_gal.sphere.radius == new_gal.sphere.radius

    # disk stuff
    assert read_in_gal.disk_kde.radius == new_gal.disk_kde.radius
    assert np.allclose(read_in_gal.disk_kde._norm_vec,
                       new_gal.disk_kde._norm_vec)
    assert read_in_gal.disk_kde.height == new_gal.disk_kde.height
    assert read_in_gal.disk_nsc.radius == new_gal.disk_nsc.radius
    assert np.allclose(read_in_gal.disk_nsc._norm_vec,
                       new_gal.disk_nsc._norm_vec)
    assert read_in_gal.disk_nsc.height == new_gal.disk_nsc.height

    # NSC indexes should be the same
    assert np.array_equal(read_in_gal.nsc_idx_sphere, new_gal.nsc_idx_sphere)
    assert np.array_equal(read_in_gal.nsc_idx_disk_nsc,
                          new_gal.nsc_idx_disk_nsc)
    assert np.array_equal(read_in_gal.nsc_idx_disk_kde,
                          new_gal.nsc_idx_disk_kde)

    # KDE profiles should be the same too.
    assert len(read_in_gal.radii) > 0  # should have multiple keys
    assert len(new_gal.radii) > 0  # should have multiple keys
    for key in read_in_gal.binned_radii:
        assert len(read_in_gal.radii[key]) > 0
        assert len(read_in_gal.densities[key]) > 0
        assert len(read_in_gal.binned_radii[key]) > 0
        assert len(read_in_gal.binned_densities[key]) > 0
        # by asserting they match below, we also check new gal length

        assert np.allclose(read_in_gal.radii[key], new_gal.radii[key])
        assert np.allclose(read_in_gal.densities[key], new_gal.densities[key])
        assert np.allclose(read_in_gal.binned_radii[key],
                           new_gal.binned_radii[key])
        assert np.allclose(read_in_gal.binned_densities[key],
                           new_gal.binned_densities[key])

    # then check some of the derived parameters
    assert read_in_gal.nsc_axis_ratios.b_over_a == new_gal.nsc_axis_ratios.b_over_a
    assert read_in_gal.nsc_axis_ratios.c_over_a == new_gal.nsc_axis_ratios.c_over_a
    assert read_in_gal.nsc_axis_ratios.c_over_b == new_gal.nsc_axis_ratios.c_over_b
    assert read_in_gal.nsc_radius == new_gal.nsc_radius
    assert np.array_equal(read_in_gal.nsc_idx_sphere, new_gal.nsc_idx_sphere)
    assert np.array_equal(read_in_gal.nsc_idx_disk_kde,
                          new_gal.nsc_idx_disk_kde)
    assert np.array_equal(read_in_gal.nsc_idx_disk_nsc,
                          new_gal.nsc_idx_disk_nsc)
    assert np.isclose(read_in_gal.mean_rot_vel.in_units("km/s").value,
                      new_gal.mean_rot_vel.in_units("km/s").value)
    assert np.isclose(read_in_gal.nsc_3d_sigma.in_units("km/s").value,
                      new_gal.nsc_3d_sigma.in_units("km/s").value)
    assert np.isclose(read_in_gal.nsc_abundances.x_on_fe_total("N"),
                      new_gal.nsc_abundances.x_on_fe_total("N"))
    assert np.isclose(read_in_gal.nsc_abundances.x_on_h_total("Ca"),
                      new_gal.nsc_abundances.x_on_h_total("Ca"))
    assert np.isclose(read_in_gal.nsc_abundances.log_z_over_z_sun_total(),
                      new_gal.nsc_abundances.log_z_over_z_sun_total())
    assert np.isclose(read_in_gal.nsc_abundances.z_on_h_total(),
                      new_gal.nsc_abundances.z_on_h_total())
    assert np.isclose(read_in_gal.gal_abundances.x_on_fe_total("N"),
                      new_gal.gal_abundances.x_on_fe_total("N"))
    assert np.isclose(read_in_gal.gal_abundances.x_on_h_total("Ca"),
                      new_gal.gal_abundances.x_on_h_total("Ca"))
    assert np.isclose(read_in_gal.gal_abundances.z_on_h_total(),
                      new_gal.gal_abundances.z_on_h_total())
    assert np.isclose(read_in_gal.gal_abundances.log_z_over_z_sun_total(),
                      new_gal.gal_abundances.log_z_over_z_sun_total())


# -----------------------------------------------------------------------------
#
# test containment
#
# -----------------------------------------------------------------------------

def test_containment(read_in_gal, gal):
    """Both galaxies are at the same spot, but read_in_gal has a much larger
    radius, so gal should be contained in read_in_gal."""
    assert read_in_gal.contains(gal)
    assert not gal.contains(read_in_gal)
    # galaxy can't contain itself.
    assert not gal.contains(gal)
    assert not read_in_gal.contains(read_in_gal)