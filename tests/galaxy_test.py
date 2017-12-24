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
# best_loc = yt.YTArray(ds.all_data().argmax("density"))
best_loc = [848.18993318, 249.52074043, 180.26988714] * kpc
# to speed things up I ran the line that does this and copied the result

def test_gal_id_generator():
    """This has to be run first, since it depends on the initial state of
    the id counter. """
    gal_0 = galaxy.Galaxy(ds, best_loc, 10*kpc, 1*kpc)
    assert gal_0.id == 101
    gal_1 = galaxy.Galaxy(ds, best_loc, 10 * kpc, 1*kpc)
    assert gal_1.id == 102
    gal_id = galaxy.Galaxy(ds, best_loc, 10 * kpc, 1*kpc, 314)
    assert gal_id.id == 314

def test_gal_dataset_typing():
    with pytest.raises(TypeError):
        galaxy.Galaxy(None, [1, 2, 3] * kpc, 10 * kpc, 1 * kpc)
    galaxy.Galaxy(ds, [1, 2, 3] * kpc, 10 * kpc, 1 * kpc) # will not raise error

def test_gal_center_unit_check():
    with pytest.raises(TypeError):
        galaxy.Galaxy(ds, [1, 2, 3], 10 * kpc, 1 * kpc)

def test_gal_center_length_check():
    with pytest.raises(ValueError):
        galaxy.Galaxy(ds, [1, 2] * kpc, 10*kpc, 1 * kpc)
    with pytest.raises(ValueError):
        galaxy.Galaxy(ds, [1, 2, 3, 4] * kpc, 10*kpc, 1 * kpc)

def test_gal_radius_unit_check():
    with pytest.raises(TypeError):
        galaxy.Galaxy(ds, [1, 2, 3]*kpc, 10, 1 * kpc)

def test_gal_j_radius_unit_check():
    with pytest.raises(TypeError):
        galaxy.Galaxy(ds, [1, 2, 3] * kpc, 10 * kpc, 1)

# -----------------------------------------------------------------------------

# Fixtures

# -----------------------------------------------------------------------------

@pytest.fixture
def gal():
    return galaxy.Galaxy(ds, best_loc, 17 * pc, 17*pc)

@pytest.fixture
def real_gal():  # has larger radius to actually include everythign we need to
    gal =  galaxy.Galaxy(ds, best_loc, 1000 * pc, 1000*pc)
    gal.add_disk(disk_radius=150 * pc, disk_height=100*pc)
    gal.find_nsc_radius()
    gal.nsc_half_mass_radius()
    gal.create_axis_ratios_nsc()
    gal.create_axis_ratios_gal()
    gal.nsc_rotation()
    gal.nsc_dispersion_eigenvectors()
    gal.create_abundances()
    return gal

@pytest.fixture
def read_in_gal():
    file = open("./real_gal_save.txt", "r")
    gal =  galaxy.read_gal(ds, file)
    file.close()
    return gal

# -----------------------------------------------------------------------------

# test the profiles process

# -----------------------------------------------------------------------------

def test_kde_creation_everything(gal):
    """This is in one mega test because the KDE creation process takes a while,
    and I don't want to have to wait. It is the access operations, which
    speed up after the first time. """
    with pytest.raises(ValueError):
        gal._create_kde_object(dimension=4)
    with pytest.raises(ValueError):
        gal._create_kde_object(dimension=0)
    with pytest.raises(ValueError):
        gal._create_kde_object(quantity="test", dimension=3)

    # check that a 2D one won't work without adding a disk
    with pytest.raises(RuntimeError):
        gal._create_kde_object(dimension=2)

    # then add the disk so everything should work.
    gal.add_disk(disk_radius=30*pc, disk_height=30*pc)
    # nothing below should raise an error, and the assert should work too.
    gal._create_kde_object()
    # kde objects should have the right dimensions
    assert gal._create_kde_object(dimension=1).dimension == 1
    assert gal._create_kde_object(dimension=2).dimension == 2
    assert gal._create_kde_object(dimension=3).dimension == 3
    # mass and metallicity should work
    gal._create_kde_object(quantity="mass")
    gal._create_kde_object(quantity="Z")

def test_kde_profile_everything(gal):
    """This is in one mega test because the KDE creation process takes a while,
    and I don't want to have to wait. This makes running tests easier, even
    though I know it's bad practice. """
    # first check units.
    with pytest.raises(TypeError):
        gal.kde_profile(outer_radius=10, dimension=2)
    # dimension has to be an integer
    with pytest.raises(ValueError):
        gal.kde_profile(dimension="sdf", outer_radius=10*pc)
    with pytest.raises(RuntimeError):  # no disk
        gal.kde_profile(dimension=2, outer_radius=10*pc)
    with pytest.raises(ValueError):  # 3D won't work
        gal.kde_profile(dimension=3, outer_radius=10*pc)

    # have to add disk to get this in 2D
    gal.add_disk(disk_radius=30*pc, disk_height=30*pc)
    gal.kde_profile(outer_radius=10 * pc, dimension=2)  # no error
    gal.kde_profile(outer_radius=10 * pc)  # defaults, no error
    with pytest.raises(ValueError):  # 1D won't work even after adding disk.
        gal.kde_profile(dimension=1, outer_radius=10*pc)

    # then check which keys work.
    gal.kde_profile("MASS", dimension=2, outer_radius=10 * pc)
    gal.kde_profile("Mass", dimension=2, outer_radius=10 * pc)
    gal.kde_profile("mass", dimension=2, outer_radius=10 * pc)
    gal.kde_profile("Z", dimension=2, outer_radius=10 * pc)
    with pytest.raises(ValueError):
        gal.kde_profile("test", dimension=2, outer_radius=10 * pc)

    # then check that the results have the right number of points
    assert len(gal.kde_densities["mass_kde_2D"]) == 10 * 100
    assert len(gal.kde_densities["z_kde_2D"]) == 10 * 100
    assert len(gal.kde_densities_smoothed["mass_kde_2D"]) == 10
    assert len(gal.kde_densities_smoothed["z_kde_2D"]) == 10

    # then check that the binning worked properly
    assert np.allclose(gal.kde_radii["mass_kde_2D"][::100],
                       gal.kde_radii_smoothed["mass_kde_2D"])
    assert np.isclose(gal.kde_radii_smoothed["mass_kde_2D"][0], 0.0)
    assert np.isclose(gal.kde_radii_smoothed["mass_kde_2D"][1], 1.0)
    assert len(gal.kde_radii["mass_kde_2D"]) == \
           100 * len(gal.kde_radii_smoothed["mass_kde_2D"])
    assert len(gal.kde_densities["mass_kde_2D"]) == \
           100 * len(gal.kde_densities_smoothed["mass_kde_2D"])

    # then see if the values are reasonable
    # there should be high density at the center here
    assert gal.kde_densities_smoothed["mass_kde_2D"][0] > 10**5
    # # the cylindrical should be a higher value than the spherical, since it
    # # only is in 2D, not three
    # assert gal.densities["mass_kde_2D"][0] > \
    #        gal.densities["mass_kde_3D"][0]
    # the stellar density should decrease outwards
    assert gal.kde_densities_smoothed["mass_kde_2D"][0] > \
           gal.kde_densities_smoothed["mass_kde_2D"][9]
    # metallicity should be between zero and one
    assert 0 < gal.kde_densities_smoothed["z_kde_2D"][0] < 1

def test_histogram_profile_everything(gal):
    """This is in one mega test because the histogram process takes a while,
    and I don't want to have to wait. This makes running tests easier, even
    though I know it's bad practice. """
    bin_num = 5
    bin_edges = np.arange(0, bin_num + 1)
    # first check whether a disk has been added
    with pytest.raises(RuntimeError):
        gal.histogram_profile(bin_edges)
    gal.add_disk(disk_radius=30*pc, disk_height=30*pc)
    gal.histogram_profile(bin_edges)  # no error

    # bin edges needs to be iterable
    with pytest.raises(TypeError):
        gal.histogram_profile(1)
    # all have to be positive.
    with pytest.raises(ValueError):
        gal.histogram_profile([-1, 0, 1])

    # then check that the results have the right number of points
    assert len(gal.binned_radii) == bin_num
    assert len(gal.binned_densities) == bin_num

def test_integrated_kde_profile_everything(gal):
    """This is in one mega test because the profile process takes a while,
    and I don't want to have to wait. This makes running tests easier, even
    though I know it's bad practice. """
    bin_num = 5
    bin_edges = np.arange(0, bin_num + 1)
    # first check whether a disk has been added
    with pytest.raises(RuntimeError):
        gal.integrated_kde_profile(bin_edges)
    gal.add_disk(disk_radius=30*pc, disk_height=30*pc)

    # bin_radii has to be non_negative
    with pytest.raises(ValueError):
        gal.integrated_kde_profile([-1, 0, 1])
    # and the bins have to be iterable
    with pytest.raises(TypeError):
        gal.integrated_kde_profile(0)

    # then do one that will work, with no error.
    gal.integrated_kde_profile(bin_edges)

    # then check that the results have the right number of points
    assert len(gal.integrated_kde_radii) == bin_num
    assert len(gal.integrated_kde_densities) == bin_num

# -----------------------------------------------------------------------------

# test the disk operations

# -----------------------------------------------------------------------------

def test_add_disk_units(gal):
    gal.add_disk()  # default parameters should work
    gal.add_disk(20 * pc, 20 * pc)
    with pytest.raises(TypeError):
        gal.add_disk(disk_radius=10)
    with pytest.raises(TypeError):
        gal.add_disk(disk_height=10)

def test_add_disk_methods(gal):
    gal.add_disk(method="axis_ratios")
    gal.add_disk(method="angular_momentum")
    with pytest.raises(ValueError):
        gal.add_disk(method="sdf")

def test_add_disk_result_type(gal):
    """We should have an actual disk when done. """
    assert gal.disk_kde is None
    gal.add_disk(disk_radius=30*pc, disk_height=30*pc)
    assert isinstance(gal.disk_kde,
                      yt.data_objects.selection_data_containers.YTDisk)

    assert gal.disk_nsc is None
    gal.add_disk(disk_radius=30*pc, disk_height=30*pc,
                 disk_type="nsc")
    assert isinstance(gal.disk_nsc,
                      yt.data_objects.selection_data_containers.YTDisk)

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

# -----------------------------------------------------------------------------
#
# test the inclusion of the structural properties
#
# -----------------------------------------------------------------------------

def test_real_nsc_existence(read_in_gal):
    utils.test_for_units(read_in_gal.nsc_radius, "NSC radius")
    assert 0 * pc < read_in_gal.nsc_radius < \
           read_in_gal.sphere.radius

def test_real_nsc_disk_attributes(read_in_gal):
    assert np.isclose(read_in_gal.disk_nsc.radius.in_units("pc").value,
                      10 * read_in_gal.nsc_radius.in_units("pc").value)
    assert np.isclose(read_in_gal.disk_nsc.height.in_units("pc").value,
                      10 * read_in_gal.nsc_radius.in_units("pc").value)
    assert np.allclose(read_in_gal.disk_nsc.center.in_units("pc").value,
                       read_in_gal.center.in_units("pc").value)

def test_real_nsc_stellar_mass(read_in_gal):
    # Test that the NSC mass is less than the total galaxy mass
    nsc_rad = read_in_gal.nsc_radius
    assert read_in_gal.stellar_mass(radius_cut=None) > \
           read_in_gal.stellar_mass(radius_cut=nsc_rad)

def test_real_nsc_radius_cut(read_in_gal):
    # test that the NSC indices actually pick up the right objects
    radius_key = ('STAR', 'particle_position_spherical_radius')
    for container, idx in zip([read_in_gal.j_sphere,
                               read_in_gal.disk_nsc],
                              [read_in_gal.nsc_idx_j_sphere,
                               read_in_gal.nsc_idx_disk_nsc]):
        radii = container[radius_key]
        assert np.max(radii[idx]) < read_in_gal.nsc_radius
        # then get the indices not in the nsc
        all_idx_set = set(range(len(radii)))
        nsc_idx_set = set(idx)
        non_nsc_idx = list(all_idx_set.difference(nsc_idx_set))
        assert np.min(radii[non_nsc_idx]) > read_in_gal.nsc_radius

def test_real_nsc_indices(read_in_gal):
    # The same number of stars should be in the NSC to matter what
    # container is being used.
    assert len(read_in_gal.nsc_idx_j_sphere) ==\
           len(read_in_gal.nsc_idx_disk_nsc)

    # then check that there are actually stars in the NSC
    assert len(read_in_gal.nsc_idx_disk_nsc) > 0

def test_real_nsc_axis_ratios(read_in_gal):
    # Test that the axis ratios for the NSC look reasonable.
    assert read_in_gal.nsc_axis_ratios.b_over_a < 1.0

def test_nsc_rotation_units(read_in_gal):
    # test that the rotation on the NSC has the right units
    utils.test_for_units(read_in_gal.mean_rot_vel,
                         "rotational velocity")
    utils.test_for_units(read_in_gal.nsc_3d_sigma,
                         "sigma")
    # throw error if not compatible
    read_in_gal.mean_rot_vel.in_units("km/s")
    read_in_gal.nsc_3d_sigma.in_units("km/s")

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
# test other stuff
#
# -----------------------------------------------------------------------------

def test_containment(gal, read_in_gal):
    """Both galaxies are at the same spot, but read_in_gal has a much larger
    radius, so gal should be contained in read_in_gal."""
    assert read_in_gal.contains(gal)
    assert not gal.contains(read_in_gal)
    # galaxy can't contain itself.
    assert not gal.contains(gal)
    assert not read_in_gal.contains(read_in_gal)

def test_half_mass_radius_units(read_in_gal):
    half_mass_radius = read_in_gal.galaxy_half_mass_radius()
    utils.test_for_units(half_mass_radius, "")  # no error

def test_half_mass_radius_actually_worked(read_in_gal):
    half_mass_radius = read_in_gal.galaxy_half_mass_radius()
    total_mass = read_in_gal.stellar_mass(radius_cut=None)
    half_mass = read_in_gal.stellar_mass(radius_cut=half_mass_radius)
    assert 0.5 < half_mass / total_mass
    assert 0 * pc < half_mass_radius < read_in_gal.radius

# -----------------------------------------------------------------------------
#
# test the reading and writing
#
# -----------------------------------------------------------------------------

def test_reading_writing(read_in_gal):
    """The only thing we need is that the object needs to be the same after
    we write then read it in. There is a lot of checking here, though."""

    old_gal = read_in_gal  # needed to easily switch from original to read in
    file = open("./real_gal_save.txt", "w")
    old_gal.write(file)
    file.close()

    file = open("./real_gal_save.txt", "r")
    new_gal = galaxy.read_gal(ds, file)

    # then compare things. First basic stuff:
    assert old_gal.id == new_gal.id
    assert np.allclose(old_gal.center.in_units("pc").value,
                       new_gal.center.in_units("pc").value)
    # ^ the .value is needed to make yt arrays play nice with allclose
    assert old_gal.radius == new_gal.radius
    assert old_gal.ds == new_gal.ds

    # spheres should be the same
    assert np.allclose(old_gal.sphere.center.in_units("pc").value,
                       new_gal.sphere.center.in_units("pc").value)
    assert old_gal.sphere.radius == new_gal.sphere.radius

    # disk stuff
    assert old_gal.disk_kde.radius == new_gal.disk_kde.radius
    assert np.allclose(old_gal.disk_kde._norm_vec,
                       new_gal.disk_kde._norm_vec)
    assert old_gal.disk_kde.height == new_gal.disk_kde.height
    assert old_gal.disk_nsc.radius == new_gal.disk_nsc.radius
    assert np.allclose(old_gal.disk_nsc._norm_vec,
                       new_gal.disk_nsc._norm_vec)
    assert old_gal.disk_nsc.height == new_gal.disk_nsc.height

    # eigenvectors of disk
    assert np.allclose(old_gal.nsc_axis_ratios.a_vec,
                       new_gal.nsc_axis_ratios.a_vec)
    assert np.allclose(old_gal.nsc_axis_ratios.b_vec,
                       new_gal.nsc_axis_ratios.b_vec)
    assert np.allclose(old_gal.nsc_axis_ratios.c_vec,
                       new_gal.nsc_axis_ratios.c_vec)
    assert np.allclose(old_gal.gal_axis_ratios.a_vec,
                       new_gal.gal_axis_ratios.a_vec)
    assert np.allclose(old_gal.gal_axis_ratios.b_vec,
                       new_gal.gal_axis_ratios.b_vec)
    assert np.allclose(old_gal.gal_axis_ratios.c_vec,
                       new_gal.gal_axis_ratios.c_vec)

    # NSC radii should be the same
    assert old_gal.nsc_radius == new_gal.nsc_radius
    assert np.allclose(old_gal.nsc_radius_err.to("pc").value,
                       new_gal.nsc_radius_err.to("pc").value)
    assert old_gal.half_mass_radius == new_gal.half_mass_radius
    assert np.allclose(old_gal.half_mass_radius_errs,
                       new_gal.half_mass_radius_errs)

    # NSC indexes should be the same
    assert np.array_equal(old_gal.nsc_idx_j_sphere,
                          new_gal.nsc_idx_j_sphere)
    assert np.array_equal(old_gal.nsc_idx_disk_nsc,
                          new_gal.nsc_idx_disk_nsc)

    # KDE profiles should be the same too.
    assert len(new_gal.kde_radii) == 0  # should have multiple keys
    for key in new_gal.kde_radii:   # shouldn't happen, but I'll keep this
        assert len(new_gal.kde_radii[key]) > 0
        assert len(new_gal.kde_densities[key]) > 0
        # by asserting they match below, we also check new gal length
        assert np.allclose(old_gal.kde_radii[key],
                           new_gal.kde_radii[key])
        assert np.allclose(old_gal.kde_densities[key],
                           new_gal.kde_densities[key])
    # then do the same for the smoothed values.
    for key in new_gal.kde_radii_smoothed:
        assert len(new_gal.kde_radii_smoothed[key]) > 0
        assert len(new_gal.kde_densities_smoothed[key]) > 0
        assert np.allclose(old_gal.kde_radii_smoothed[key],
                           new_gal.kde_radii_smoothed[key])
        assert np.allclose(old_gal.kde_densities_smoothed[key],
                           new_gal.kde_densities_smoothed[key])

    # and the binned radii
    assert np.allclose(old_gal.binned_radii,     new_gal.binned_radii)
    assert np.allclose(old_gal.binned_densities, new_gal.binned_densities)
    assert np.allclose(old_gal.integrated_kde_radii,
                       new_gal.integrated_kde_radii)
    assert np.allclose(old_gal.integrated_kde_densities,
                       new_gal.integrated_kde_densities)

    # then check some of the derived parameters
    assert np.isclose(old_gal.nsc_axis_ratios.b_over_a,
                      new_gal.nsc_axis_ratios.b_over_a)
    assert np.isclose(old_gal.nsc_axis_ratios.c_over_a,
                      new_gal.nsc_axis_ratios.c_over_a)
    assert np.isclose(old_gal.nsc_axis_ratios.c_over_b,
                      new_gal.nsc_axis_ratios.c_over_b)
    assert np.isclose(old_gal.nsc_axis_ratios.ellipticity,
                      new_gal.nsc_axis_ratios.ellipticity)
    assert np.isclose(old_gal.mean_rot_vel.in_units("km/s").value,
                      new_gal.mean_rot_vel.in_units("km/s").value)
    assert np.isclose(old_gal.nsc_3d_sigma.in_units("km/s").value,
                      new_gal.nsc_3d_sigma.in_units("km/s").value)
    assert np.isclose(old_gal.anisotropy_parameter,
                      new_gal.anisotropy_parameter)
    assert np.isclose(old_gal.nsc_disp_along_a.in_units("km/s").value,
                      new_gal.nsc_disp_along_a.in_units("km/s").value)
    assert np.isclose(old_gal.nsc_disp_along_b.in_units("km/s").value,
                      new_gal.nsc_disp_along_b.in_units("km/s").value)
    assert np.isclose(old_gal.nsc_disp_along_c.in_units("km/s").value,
                      new_gal.nsc_disp_along_c.in_units("km/s").value)
    assert np.isclose(old_gal.nsc_abundances.x_on_fe_total("N"),
                      new_gal.nsc_abundances.x_on_fe_total("N"))
    assert np.isclose(old_gal.nsc_abundances.x_on_h_total("Ca"),
                      new_gal.nsc_abundances.x_on_h_total("Ca"))
    assert np.isclose(old_gal.nsc_abundances.log_z_over_z_sun_total(),
                      new_gal.nsc_abundances.log_z_over_z_sun_total())
    assert np.isclose(old_gal.nsc_abundances.z_on_h_total(),
                      new_gal.nsc_abundances.z_on_h_total())
    assert np.isclose(old_gal.gal_abundances.x_on_fe_total("N"),
                      new_gal.gal_abundances.x_on_fe_total("N"))
    assert np.isclose(old_gal.gal_abundances.x_on_h_total("Ca"),
                      new_gal.gal_abundances.x_on_h_total("Ca"))
    assert np.isclose(old_gal.gal_abundances.z_on_h_total(),
                      new_gal.gal_abundances.z_on_h_total())
    assert np.isclose(old_gal.gal_abundances.log_z_over_z_sun_total(),
                      new_gal.gal_abundances.log_z_over_z_sun_total())
