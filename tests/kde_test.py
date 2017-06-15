from yt_tools import kde
from yt_tools import utils

import pytest
import numpy as np

# -----------------------------------------------------------

#  Test the initialization of the KDE class

# -----------------------------------------------------------

def test_kde_init_results_3d():
    """Tests that a KDE object in 3D has the correct values"""
    kde_obj = kde.KDE([[1, 2, 3], [2, 3, 4], [3, 4, 5]], [4, 5, 6])
    assert kde_obj.dimension == 3
    assert kde_obj.x == [1, 2, 3]
    assert kde_obj.y == [2, 3, 4]
    assert kde_obj.z == [3, 4, 5]
    assert kde_obj.values == [4, 5, 6]
    assert kde_obj.smoothing_function == utils.gaussian_3d_radial

def test_kde_init_results_2d():
    """Tests that a KDE object in 2D has the correct values, including it
    not having a z coordinate. """
    kde_obj = kde.KDE([[1, 2, 3], [2, 3, 4]], [3, 4, 5])
    assert kde_obj.dimension == 2
    assert kde_obj.x == [1, 2, 3]
    assert kde_obj.y == [2, 3, 4]
    assert kde_obj.values == [3, 4, 5]
    assert kde_obj.smoothing_function == utils.gaussian_2d_radial
    with pytest.raises(AttributeError):
        kde_obj.z

def test_kde_init_no_weights():
    """Tests that if you don't pass in weights, all are weighted equally."""
    kde_obj = kde.KDE([[1, 2, 3], [2, 3, 4]])
    assert np.array_equal(kde_obj.values, np.ones(3))

def test_kde_init_checking():
    """Test the various error checking that happens"""
    with pytest.raises(ValueError):
        # values has to have the same length as x and y
        kde_obj = kde.KDE([[1, 2, 3], [2, 3, 4]], [3, 4, 5, 6])
    with pytest.raises(ValueError):
        # values has to have the same length as x and y
        kde_obj = kde.KDE([[1, 2, 3], [2, 3, 4], [3, 4, 5]], [1, 2, 3, 4])
    with pytest.raises(ValueError):
        # x, y, and z have to have the same length
        kde_obj = kde.KDE([[1, 2, 3, 4], [2, 3, 4], [3, 4, 5]], [1, 2, 3])
    with pytest.raises(TypeError):
        # x, y, and z have to be arrays
        kde.KDE([1, 2, 3], [1])
    with pytest.raises(TypeError):
        # values has to be an array
        kde.KDE([[0], [0], [0]], 1)
    with pytest.raises(TypeError):
        # x, y, and z have to be arrays, even if we don't pass in any values
        kde.KDE([0, 0, 0])

# -----------------------------------------------------------

#  Fixtures for types of KDE objects. Will be used to make
#  the creation of KDE objects easier.

# -----------------------------------------------------------

@pytest.fixture
def single_point_at_zero_2d():
    return kde.KDE([np.array([0]), np.array([0])])

@pytest.fixture
def single_point_at_one_2d():
    return kde.KDE([np.array([1]), np.array([1])])

@pytest.fixture
def single_point_at_zero_3d():
    return kde.KDE([np.array([0]), np.array([0]), np.array([0])])

@pytest.fixture
def single_point_at_one_3d():
    return kde.KDE([np.array([1]), np.array([1]), np.array([1])])

@pytest.fixture
def single_point_at_zero_2d_weighted():
    return kde.KDE([np.array([0]), np.array([0])],
                   np.array([np.random.uniform(1, 5)]))

@pytest.fixture
def single_point_at_zero_3d_weighted():
    return kde.KDE([np.array([0]), np.array([0]), np.array([0])],
                   np.array([np.random.uniform(1, 5)]))

# -----------------------------------------------------------

#  Testing KDE density 

# -----------------------------------------------------------

def test_density_error_checking_points_2d_loc_3d(single_point_at_zero_2d):
    """Test that a 2D point needs 2D coordinates"""
    with pytest.raises(ValueError):
        single_point_at_zero_2d.density(1, 1, 1, 1)

def test_density_error_checking_points_3d_loc_2d(single_point_at_zero_3d):
    """Test that a 3D point needs 3D coordinates"""
    with pytest.raises(ValueError):
        single_point_at_zero_3d.density(1, 1, 1)

def test_density_single_point_2d_zero_zero_dist(single_point_at_zero_2d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_2d_radial(0, sigma)
    assert np.isclose(density, single_point_at_zero_2d.density(sigma, 0, 0))

def test_density_single_point_2d_one_zero_dist(single_point_at_one_2d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_2d_radial(0, sigma)
    assert np.isclose(density, single_point_at_one_2d.density(sigma, 1, 1))

def test_density_single_point_3d_zero_zero_dist(single_point_at_zero_3d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_3d_radial(0, sigma)
    assert np.isclose(density, single_point_at_zero_3d.density(sigma, 0, 0, 0))

def test_density_single_point_3d_one_zero_dist(single_point_at_one_3d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_3d_radial(0, sigma)
    assert np.isclose(density, single_point_at_one_3d.density(sigma, 1, 1, 1))

def test_density_single_point_2d_zero_nonzero_dist(single_point_at_zero_2d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_2d_radial(1, sigma)
    assert np.isclose(density, single_point_at_zero_2d.density(sigma, 0, 1))

def test_density_single_point_2d_one_nonzero_dist(single_point_at_one_2d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_2d_radial(2, sigma)
    assert np.isclose(density, single_point_at_one_2d.density(sigma, 3, 1))

def test_density_single_point_3d_zero_nonzero_dist(single_point_at_zero_3d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_3d_radial(5, sigma)
    assert np.isclose(density, single_point_at_zero_3d.density(sigma, 0, 5, 0))

def test_density_single_point_3d_one_nonzero_dist(single_point_at_one_3d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_3d_radial(3, sigma)
    assert np.isclose(density, single_point_at_one_3d.density(sigma, 1, 1, -2))

def test_density_single_point_2d_weighted(single_point_at_zero_2d_weighted):
    """Test a simple density calculation in 2D, but with weights"""
    sigma = 5.357
    density = utils.gaussian_2d_radial(0, sigma)

    kde_density = single_point_at_zero_2d_weighted.density(sigma, 0, 0)
    weight = np.sum(single_point_at_zero_2d_weighted.values)
    assert np.isclose(density * weight, kde_density)

def test_density_single_point_3d_weighted(single_point_at_zero_3d_weighted):
    """Test a simple density calculation in 3D, but with weights"""
    sigma = 5.357
    density = utils.gaussian_3d_radial(0, sigma)

    kde_density = single_point_at_zero_3d_weighted.density(sigma, 0, 0, 0)
    weight = np.sum(single_point_at_zero_3d_weighted.values)
    assert np.isclose(density * weight, kde_density)

def test_density_4_points_2d():
    """Test a density calculation with multiple points in 2D. I made a plus
    sign around the origin, where each point is weighted differently."""
    weights = [2, 3, 4, 5]
    # create points on the axes 1 unit away from the center
    kde_obj = kde.KDE([np.array([1, -1, 0, 0]),
                       np.array([0, 0, 1, -1])], 
                      weights)
    sigma = 4.987
    # get the density of a gaussian at 1 unit away
    density = utils.gaussian_2d_radial(1, sigma)
    # then multiply by the sum of the weights. This works since they are all 
    # the same distance away, so should have the same density.
    total_density = density * np.sum(weights)
    # then compare that to the real result
    assert np.isclose(total_density, kde_obj.density(sigma, 0, 0))

def test_density_6_points_3d():
    """Test a density calculation in 3D with multiple points. Each point is 
    1 unit away from the origin, but has different weights"""
    weights = [2, 3, 4, 5, 6, 7]
    # create points on the axes 1 unit away from the center
    kde_obj = kde.KDE([np.array([1, -1,  0,  0,  0,  0]),
                       np.array([0,  0,  1, -1,  0,  0]),
                       np.array([0,  0,  0,  0,  1, -1])], 
                      weights)
    sigma = 4.987
    # get the density of a gaussian at 1 unit away
    density = utils.gaussian_3d_radial(1, sigma)
    # then multiply by the sum of the weights. This works since they are all the
    # same distance away from the center, so they should have the same density.
    total_density = density * np.sum(weights)
    # then compare that to the real result
    assert np.isclose(total_density, kde_obj.density(sigma, 0, 0, 0))


# -----------------------------------------------------------

#  Test the centering setup functions

# -----------------------------------------------------------

def test_initial_cell_size_3d():
    """Test the process that creates the size of the initial kernel"""
    kde_obj = kde.KDE([np.array([10, 110]),
                       np.array([0,  1]),
                       np.array([0,  50])])
    assert kde_obj._initial_cell_size() == 10

    kde_obj = kde.KDE([np.array([10, 11]),
                       np.array([0,  1]),
                       np.array([0,  49])])
    assert kde_obj._initial_cell_size() == 4.9

def test_initial_cell_size_2d():
    """Test the process that creates the size of the initial kernel"""
    kde_obj = kde.KDE([np.array([10, 110]),
                       np.array([0,  50])])
    assert kde_obj._initial_cell_size() == 10

    kde_obj = kde.KDE([np.array([10, 11]),
                       np.array([0,  49])])
    assert kde_obj._initial_cell_size() == 4.9


def test_grid_resolution_steps():
    """Test the process that creates the grid resolution. I will only test
    the simple cases where I know what I want."""
    assert np.allclose(kde.grid_resolution_steps(100, 1),
                       np.array([100, 10, 1]))
    assert np.allclose(kde.grid_resolution_steps(2, 1),
                       np.array([2, 1]))
    assert np.allclose(kde.grid_resolution_steps(1.0001, 1),
                       np.array([1.0001, 1]))
    assert np.allclose(kde.grid_resolution_steps(5000, 50),
                       np.array([5000., 500., 50.]))
    assert np.allclose(kde.grid_resolution_steps(50, 1),
                       np.logspace(np.log10(50), 0, 3))

def test_constructing_grid_3d():
    """ See if the creation of the grid is correct"""
    # I will manually create the simplest grid to make sure it's working
    result_grid = [[-1, -1, -1], [-1, -1,  0], [-1, -1,  1],
                   [-1,  0, -1], [-1,  0,  0], [-1,  0,  1],
                   [-1,  1, -1], [-1,  1,  0], [-1,  1,  1],
                   [ 0, -1, -1], [ 0, -1,  0], [ 0, -1,  1],
                   [ 0,  0, -1], [ 0,  0,  0], [ 0,  0,  1],
                   [ 0,  1, -1], [ 0,  1,  0], [ 0,  1,  1], 
                   [ 1, -1, -1], [ 1, -1,  0], [ 1, -1,  1],
                   [ 1,  0, -1], [ 1,  0,  0], [ 1,  0,  1],
                   [ 1,  1, -1], [ 1,  1,  0], [ 1,  1,  1]]
    kde_grid = kde.construct_grid(1, 0, 0, 0, dimensions=3, points_per_side=1)
    assert np.array_equal(np.array(kde_grid), np.array(result_grid))

def test_constructing_grid_2d():
    """ See if the creation of the grid is correct"""
    # I will manually create the simplest grid to make sure it's working
    result_grid = [[-1, -1], [-1, 0], [-1, 1],
                   [ 0, -1], [ 0, 0], [ 0, 1],
                   [ 1, -1], [ 1, 0], [ 1, 1]]
    kde_grid = kde.construct_grid(1, 0, 0, 0, dimensions=2, points_per_side=1)
    assert np.array_equal(np.array(kde_grid), np.array(result_grid))
                            
def test_max_in_pairs():
    values = [ 1,   2,   3,   4,   5,   4,   2]
    labels = ["a", "b", "c", "d", "e", "f", "g"]
    assert kde.max_in_pairs(values, labels) == "e"

def test_error_checking_max_in_pairs():
    with pytest.raises(ValueError):
        kde.max_in_pairs([1], [1, 2])

# -----------------------------------------------------------

#  Test the centering function itself

# -----------------------------------------------------------

def test_centering_single_point_2d(single_point_at_zero_2d_weighted):
    """Make sure it works for a single point in 2d"""
    accuracy = 0.001
    single_point_at_zero_2d_weighted.centering(0.5, accuracy)
    assert np.isclose(single_point_at_zero_2d_weighted.location_max_x, 0, 
                      rtol=0, atol=accuracy)
    assert np.isclose(single_point_at_zero_2d_weighted.location_max_y, 0, 
                      rtol=0, atol=accuracy)

def test_centering_single_point_3d(single_point_at_zero_3d_weighted):
    """make sure it works for a single point in 3d"""
    accuracy = 0.001
    single_point_at_zero_3d_weighted.centering(0.5, accuracy)
    assert np.isclose(single_point_at_zero_3d_weighted.location_max_x, 0, 
                      rtol=0, atol=accuracy)
    assert np.isclose(single_point_at_zero_3d_weighted.location_max_y, 0, 
                      rtol=0, atol=accuracy)
    assert np.isclose(single_point_at_zero_3d_weighted.location_max_z, 0, 
                      rtol=0, atol=accuracy)

    # test not at zero
    kde_obj = kde.KDE([np.array([10]), np.array([1]), np.array([5])],
                      np.array([4]))
    kde_obj.centering(0.5, accuracy)
    assert np.isclose(kde_obj.location_max_x, 10, 
                      rtol=0, atol=accuracy)
    assert np.isclose(kde_obj.location_max_y, 1, 
                      rtol=0, atol=accuracy)
    assert np.isclose(kde_obj.location_max_z, 5, 
                      rtol=0, atol=accuracy)

def test_centering_many_points_3d():
    """Creates a Gaussian distribution of many points, and this should
    recover the center"""
    xs = np.random.normal(3, 1, 10000)
    ys = np.random.normal(4, 1, 10000)
    zs = np.random.normal(-2, 1, 10000)

    this_kde = kde.KDE([xs, ys, zs])
    this_kde.centering(0.2, 0.01)
    assert np.isclose(this_kde.location_max_x, 3.0, atol=0.2)
    assert np.isclose(this_kde.location_max_y, 4.0, atol=0.2)
    assert np.isclose(this_kde.location_max_z, -2.0, atol=0.2)


@pytest.fixture
def two_points_3d():
    """ Returns two points, one at (0, 0, 0), the other at (1, 1, 1) with 
     equal weights"""
    return kde.KDE([np.array([0, 1]), np.array([0, 1]), np.array([0, 1])])

@pytest.fixture
def two_points_2d():
    """ Returns two points, one at (0, 0), the other at (1, 1) with 
     equal weights"""
    return kde.KDE([np.array([0, 1]), np.array([0, 1])])

@pytest.fixture
def two_points_3d_weighted():
    """ Returns two points, one at (0, 0, 0), the other at (1, 1, 1) with 
     weights of (10, 1)"""
    return kde.KDE([np.array([0, 1]), np.array([0, 1]), np.array([0, 1])],
                   np.array([10, 1]))

@pytest.fixture
def two_points_2d_weighted():
    """ Returns two points, one at (0, 0), the other at (1, 1) with 
     weights of (10, 1)"""
    return kde.KDE([np.array([0, 1]), np.array([0, 1])],
                   np.array([10, 1]))

def test_two_points_large_kernel_3d(two_points_3d):
    """Check that with two points and an extremely large kernel, the center will
    be right in between the two points."""
    accuracy = 0.001
    two_points_3d.centering(20, accuracy)
    assert np.isclose(two_points_3d.location_max_x, 0.5, rtol=0, atol=accuracy)
    assert np.isclose(two_points_3d.location_max_y, 0.5, rtol=0, atol=accuracy)
    assert np.isclose(two_points_3d.location_max_z, 0.5, rtol=0, atol=accuracy)

def test_two_points_large_kernel_2d(two_points_2d):
    """Check that with two points and an extremely large kernel, the center will
    be right in between the two points"""
    accuracy = 0.001
    two_points_2d.centering(20, accuracy)
    assert np.isclose(two_points_2d.location_max_x, 0.5, rtol=0, atol=accuracy)
    assert np.isclose(two_points_2d.location_max_y, 0.5, rtol=0, atol=accuracy)
    assert two_points_2d.location_max_z == None

def test_two_points_small_kernel_3d(two_points_3d_weighted):
    """Check that with two points with uneven weights, and a very small kernel,
    the center will be at the location of the one of the larger weight."""
    accuracy = 0.001
    two_points_3d_weighted.centering(0.01, accuracy)
    assert np.isclose(two_points_3d_weighted.location_max_x, 0, 
                      rtol=0, atol=accuracy)
    assert np.isclose(two_points_3d_weighted.location_max_y, 0, 
                      rtol=0, atol=accuracy)
    assert np.isclose(two_points_3d_weighted.location_max_z, 0, 
                      rtol=0, atol=accuracy)

def test_two_points_small_kernel_2d(two_points_2d_weighted):
    """Check that with two points with uneven weights, and a very small kernel,
    the center will be at the location of the one of the larger weight."""
    accuracy = 0.001
    two_points_2d_weighted.centering(0.01, accuracy)
    assert np.isclose(two_points_2d_weighted.location_max_x, 0, 
                      rtol=0, atol=accuracy)
    assert np.isclose(two_points_2d_weighted.location_max_y, 0, 
                      rtol=0, atol=accuracy)
    assert two_points_2d_weighted.location_max_z == None

def test_radial_profile_3d_single_point(single_point_at_zero_3d):
    radii = np.arange(0, 100, 0.5)
    sigma = 2.0
    true_densities = utils.gaussian_3d_radial(radii, sigma)
    assert np.allclose(single_point_at_zero_3d.radial_profile(sigma, radii,
                                                              [0, 0, 0]),
                       true_densities)

def test_radial_profile_2d_single_point(single_point_at_zero_2d):
    radii = np.arange(0, 100, 0.5)
    sigma = 2.0
    true_densities = utils.gaussian_2d_radial(radii, sigma)
    assert np.allclose(single_point_at_zero_2d.radial_profile(sigma, radii,
                                                              [0, 0]),
                       true_densities)

def test_radial_profile_error_checking_2D(single_point_at_zero_2d):
    with pytest.raises(ValueError):
        single_point_at_zero_2d.radial_profile(1, np.arange(0, 10), [1, 1, 1])

def test_radial_profile_error_checking_3D(single_point_at_zero_3d):
    with pytest.raises(ValueError):
        single_point_at_zero_3d.radial_profile(1, np.arange(0, 10),
                                               [1, 1])
