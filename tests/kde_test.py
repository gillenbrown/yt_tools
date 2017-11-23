from yt_tools import kde
from yt_tools import utils

import pytest
import numpy as np
from scipy import integrate

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

def test_kde_init_results_1d():
    """Tests that a KDE object in 2D has the correct values, including it
    not having a z coordinate. """
    kde_obj = kde.KDE([[1, 2, 3]], [3, 4, 5])
    assert kde_obj.dimension == 1
    assert kde_obj.x == [1, 2, 3]
    assert kde_obj.values == [3, 4, 5]
    assert kde_obj.smoothing_function == utils.gaussian_1d
    with pytest.raises(AttributeError):
        kde_obj.y
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
def single_point_at_zero_1d():
    return kde.KDE([np.array([0])])

@pytest.fixture
def single_point_at_one_1d():
    return kde.KDE([np.array([1])])

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
def single_point_at_zero_1d_weighted():
    return kde.KDE([np.array([0])],
                   np.array([np.random.uniform(1, 5)]))

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

def test_density_error_checking_points_1d(single_point_at_zero_1d):
    """Test that a 1D point needs 1D coordinates"""
    with pytest.raises(ValueError):
        single_point_at_zero_1d.density(kernel_size=1, x=1, y=1)
    with pytest.raises(ValueError):
        single_point_at_zero_1d.density(kernel_size=1, x=1, z=1)
    with pytest.raises(ValueError):
        single_point_at_zero_1d.density(kernel_size=1, x=1, y=1, z=1)
    with pytest.raises(ValueError):
        single_point_at_zero_1d.density(kernel_size=1, x=None)
    # no error
    single_point_at_zero_1d.density(kernel_size=1, x=1)

def test_density_error_checking_points_2d(single_point_at_zero_2d):
    """Test that a 2D point needs 2D coordinates"""
    with pytest.raises(ValueError):
        single_point_at_zero_2d.density(kernel_size=1, x=1, y=1, z=1)
    with pytest.raises(ValueError):
        single_point_at_zero_2d.density(kernel_size=1, x=None, y=1)
    with pytest.raises(ValueError):
        single_point_at_zero_2d.density(kernel_size=1, x=1, y=None)
    with pytest.raises(ValueError):
        single_point_at_zero_2d.density(kernel_size=1, x=None, y=None)

    # no error
    single_point_at_zero_2d.density(kernel_size=1, x=1, y=1)

def test_density_error_checking_points_3d(single_point_at_zero_3d):
    """Test that a 3D point needs 3D coordinates"""
    with pytest.raises(ValueError):
        single_point_at_zero_3d.density(kernel_size=1, x=1, y=1, z=None)
    with pytest.raises(ValueError):
        single_point_at_zero_3d.density(kernel_size=1, x=1, y=None, z=1)
    with pytest.raises(ValueError):
        single_point_at_zero_3d.density(kernel_size=1, x=None, y=1, z=1)
    with pytest.raises(ValueError):
        single_point_at_zero_3d.density(kernel_size=1, x=None, y=None, z=1)
    with pytest.raises(ValueError):
        single_point_at_zero_3d.density(kernel_size=1, x=None, y=1, z=None)
    with pytest.raises(ValueError):
        single_point_at_zero_3d.density(kernel_size=1, x=1, y=None, z=None)
    with pytest.raises(ValueError):
        single_point_at_zero_3d.density(kernel_size=1, x=None, y=None, z=None)

    # no error
    single_point_at_zero_3d.density(kernel_size=1, x=1, y=1, z=1)

def test_density_single_point_1d_zero_zero_dist(single_point_at_zero_1d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_1d(0, sigma)
    assert np.isclose(density, single_point_at_zero_1d.density(sigma, 0))

def test_density_single_point_1d_one_zero_dist(single_point_at_one_1d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_1d(0, sigma)
    assert np.isclose(density, single_point_at_one_1d.density(sigma, 1))

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


# Now with nonzero distance, at any location.

def test_density_single_point_1d_zero_nonzero_dist(single_point_at_zero_1d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_1d(1, sigma)
    assert np.isclose(density, single_point_at_zero_1d.density(sigma, 1))

def test_density_single_point_1d_one_nonzero_dist(single_point_at_one_1d):
    """Test a simple density calculation in 2D"""
    sigma = 5
    density = utils.gaussian_1d(2, sigma)
    assert np.isclose(density, single_point_at_one_1d.density(sigma, 3))

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

# now with weighting

def test_density_single_point_1d_weighted(single_point_at_zero_1d_weighted):
    """Test a simple density calculation in 2D, but with weights"""
    sigma = 5.357
    density = utils.gaussian_1d(0, sigma)

    kde_density = single_point_at_zero_1d_weighted.density(sigma, 0)
    weight = np.sum(single_point_at_zero_1d_weighted.values)
    assert np.isclose(density * weight, kde_density)

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

def test_density_2_points_1d():
    """Test a density distribution around a single point in 1D.
    Each point is weighted differently."""
    weights = np.array([4, 7])
    # create points different distances from the center.
    kde_obj = kde.KDE([np.array([-1, 2])], weights)
    sigma = 2.3678
    # get the Gaussian density at the appropriate distances. We will do this
    # at x=0.
    density = utils.gaussian_1d(np.array([1, 2]), sigma)
    # then multiply by the sum of the weights
    true_density = np.sum(weights * density)

    # then compare that to the real result
    assert np.isclose(true_density, kde_obj.density(sigma, 0))

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


def test_grid_resolution_steps_examples():
    """Test the process that creates the grid resolution. I will only test
    the simple cases where I know what I want."""
    assert np.allclose(kde.grid_resolution_steps(81, 1),
                       np.array([81, 27, 9, 3, 1]))
    assert np.allclose(kde.grid_resolution_steps(2, 1),
                       np.array([2, 1]))
    assert np.allclose(kde.grid_resolution_steps(1.0001, 1),
                       np.array([1.0001, 1]))
    assert np.allclose(kde.grid_resolution_steps(4050, 50),
                       np.array([4050, 1350, 450, 150, 50]))


def test_grid_resolution_steps_lengths():
    # then check the length of a couple of examples.
    l_82 = len(kde.grid_resolution_steps(82, 1))
    l_81 = len(kde.grid_resolution_steps(81, 1))
    l_80 = len(kde.grid_resolution_steps(80, 1))
    l_28 = len(kde.grid_resolution_steps(28, 1))
    l_27 = len(kde.grid_resolution_steps(27, 1))
    assert l_82 != l_81
    assert l_81 == l_80
    assert l_80 == l_28
    assert l_28 != l_27


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
    """I originally wanted to create a Gaussian distrubition of many points,
    then find the center of that. The randomness in creating that distribution
    make the process not work well. So instead I create a grid of points that
    are weighted by the appropriate value for a 3D Gaussian. Then the code
    can find the weighted center of those points."""
    cen_x = 3.0
    cen_y = 4.0
    cen_z = -2.0
    # use the centering code's guts to create a grid of points
    locations = kde.construct_grid(cell_size=0.2, center_x=cen_x,
                                   center_y=cen_y, center_z=cen_z,
                                   points_per_side=10)
    # then weight them according to their Gaussian distance from the center.
    weights = []
    xs, ys, zs = [], [], []
    for loc in locations:
        x, y, z = loc
        distance = utils.distance(x, cen_x, y, cen_y, z, cen_z)
        weights.append(utils.gaussian_3d_radial(distance, 1.0))
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # then we can do the centering process, and it should work great.
    this_kde = kde.KDE([xs, ys, zs], values=weights)
    this_kde.centering(0.4, 0.01)
    assert np.isclose(this_kde.location_max_x, cen_x, atol=0.01)
    assert np.isclose(this_kde.location_max_y, cen_y, atol=0.01)
    assert np.isclose(this_kde.location_max_z, cen_z, atol=0.01)

def test_centering_3d_few():
    """Create a situation with a few points all at one spot."""
    xs = [0, 1, 1, 1, 2]
    ys = xs
    zs = xs
    this_kde = kde.KDE([xs, ys, zs])
    this_kde.centering(0.2, 0.01)
    assert np.isclose(this_kde.location_max_x, 1, atol=0.01)
    assert np.isclose(this_kde.location_max_y, 1, atol=0.01)
    assert np.isclose(this_kde.location_max_z, 1, atol=0.01)

def test_centering_bimodal_2d():
    xs = [0, 1, 1, 1, 2, 3, 3, 4]
    ys = xs
    this_kde = kde.KDE([xs, ys])
    this_kde.centering(0.2, 0.01)
    assert np.isclose(this_kde.location_max_x, 1, atol=0.01)
    assert np.isclose(this_kde.location_max_y, 1, atol=0.01)

def test_centering_bimodal_2d_large_range():
    xs = [-100, 0, 1, 1, 1, 2, 3, 3, 4, 100]
    ys = xs
    this_kde = kde.KDE([xs, ys])
    this_kde.centering(0.2, 0.01)
    assert np.isclose(this_kde.location_max_x, 1, atol=0.01)
    assert np.isclose(this_kde.location_max_y, 1, atol=0.01)

def test_centering_bimodal_different_sizes_large_range():
    """This test shows the bug/feature of my code. It will ignore peaks that are
    "higher", but that have less mass. The advantage of doing this is that the
    center it finds is likely to be near the center of mass of the galaxy."""
    xs = [-100, 100, # for edes
          1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  # false peak
          20, 20, 21, 21, 22, 22, 23, 23, 23, 24, 24, 25, 25] # real peak
    ys = xs
    this_kde = kde.KDE([xs, ys])
    this_kde.centering(0.2, 0.01)
    assert np.isclose(this_kde.location_max_x, 23, atol=0.01)
    assert np.isclose(this_kde.location_max_y, 23, atol=0.01)

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

# def test_radial_profile_3d_single_point(single_point_at_zero_3d):
    # radii = np.arange(0, 100, 0.5)
    # sigma = 2.0
    # true_densities = utils.gaussian_3d_radial(radii, sigma)
    # assert np.allclose(single_point_at_zero_3d.radial_profile(sigma, radii, 1,
    #                                                           [0, 0, 0]),
    #                    true_densities)


# -----------------------------------------------------------

#  Radial profiles

# -----------------------------------------------------------

def test_radial_profile_error_checking_kernel_length(single_point_at_zero_2d):
    radii = np.linspace(0, 1, 5)
    kernels = np.linspace(1, 2, 4)  # not same length as radii
    with pytest.raises(ValueError):
        single_point_at_zero_2d.radial_profile(kernels, radii, 1, [0, 0])

def test_radial_profile_error_checking_1d_center(single_point_at_zero_1d):
    """The center should have the right dimension."""
    radii = np.linspace(0, 1, 5)
    kernels = np.linspace(1, 2, 5)
    with pytest.raises(ValueError):
        single_point_at_zero_1d.radial_profile(kernels, radii, 1, [0, 0])

def test_radial_profile_error_checking_2d_center(single_point_at_zero_2d):
    """The center should have the right dimension."""
    radii = np.linspace(0, 1, 5)
    kernels = np.linspace(1, 2, 5)
    with pytest.raises(ValueError):
        single_point_at_zero_2d.radial_profile(kernels, radii, 1, [0, 0, 0])

def test_radial_profile_error_checking_3d_center(single_point_at_zero_3d):
    """The center should have the right dimension."""
    radii = np.linspace(0, 1, 5)
    kernels = np.linspace(1, 2, 5)
    with pytest.raises(ValueError):
        single_point_at_zero_3d.radial_profile(kernels, radii, 1, [0, 0])

def test_radial_profile_error_check_positive_radii_1D(single_point_at_zero_1d):
    for radius in [-2, -1, 0, 1, 2]:
        if radius < 0:
            with pytest.raises(ValueError):
                single_point_at_zero_1d.radial_profile([1], [radius], 1, [0])
        else:  # positive radius, won't get an error.
            single_point_at_zero_1d.radial_profile([1], [radius], 1, [0])

def test_radial_profile_error_check_positive_radii_2D(single_point_at_zero_2d):
    for radius in [-2, -1, 0, 1, 2]:
        if radius < 0:
            with pytest.raises(ValueError):
                single_point_at_zero_2d.radial_profile([1], [radius], 1, [0, 0])
        else:  # positive radius, won't get an error.
            single_point_at_zero_2d.radial_profile([1], [radius], 1, [0, 0])

def test_radial_profile_1d_num_each_error_checking(single_point_at_zero_1d):
    """num_each can't be anything other than 1 in 1D."""
    num_radii = 123
    radii = np.linspace(0, 100, num_radii)
    num_each = 23
    with pytest.raises(ValueError):
        single_point_at_zero_1d.radial_profile(1, radii, num_each, [0])

def test_radial_profile_1d_center_error_checking(single_point_at_zero_1d):
    """Center can't be anything other than zero in 1D. """
    radii = [1, 2, 3]
    with pytest.raises(ValueError):
        single_point_at_zero_1d.radial_profile(1, radii, 1, [10])

def test_radial_profile_1d_no_neg_x_error_checking():
    """All x values have to be positive when doing a radial profile in 1D.
    See the documentation for the radial profile function for more info."""
    kde_obj = kde.KDE([[-1]])
    with pytest.raises(ValueError):
        kde_obj.radial_profile([1], [1], 1, [0])

#def test_radial_profile_error_check_positive_radii_3D(single_point_at_zero_3d):
#     for radius in [-2, -1, 0, 1, 2]:
#         if radius < 0:
#             with pytest.raises(ValueError):
#                 single_point_at_zero_3d.radial_profile([1], [radius], 1,
#                                                        [0, 0, 0])
#         else:  # positive radius, won't get an error.
#             single_point_at_zero_3d.radial_profile([1], [radius], 1,
#                                                    [0, 0, 0])

def test_radial_profile_1d_single_point(single_point_at_zero_1d):
    radii = np.arange(0, 100, 0.5)
    sigma = 2.0
    # the true density in the profile is a little trickier than it is in 2 or
    # 3D, since we have to account for the portion that goes below zero. We
    # handle that here by multiplying by 2, since it's symmetric.
    true_densities = utils.gaussian_1d(radii, sigma) * 2.0
    test_out = single_point_at_zero_1d.radial_profile(sigma, radii, 1, [0])
    new_radii, new_densities = test_out
    assert np.allclose(true_densities, new_densities)
    assert np.allclose(radii, new_radii)

def test_radial_profile_2d_single_point(single_point_at_zero_2d):
    radii = np.arange(0, 100, 0.5)
    sigma = 2.0
    true_densities = utils.gaussian_2d_radial(radii, sigma)
    func_out = single_point_at_zero_2d.radial_profile(sigma, radii, 1, [0, 0])
    new_radii, new_densities = func_out
    assert np.allclose(true_densities, new_densities)
    assert np.allclose(radii, new_radii)

def test_radial_profile_2d_length_of_arrays(single_point_at_zero_2d):
    num_radii = 123
    radii = np.linspace(0, 100,num_radii)
    num_each = 23
    new_rad, dens = single_point_at_zero_2d.radial_profile(1, radii, num_each,
                                                           [0, 0])
    assert len(new_rad) == len(dens)
    assert len(new_rad) == num_radii * num_each

def test_radial_profile_2d_multiple_at_each(single_point_at_zero_2d):
    sigma = 2.0
    num_each = 2
    true_densities = [utils.gaussian_2d_radial(0, sigma),
                      utils.gaussian_2d_radial(0, sigma),
                      utils.gaussian_2d_radial(1, sigma),
                      utils.gaussian_2d_radial(1, sigma),
                      utils.gaussian_2d_radial(2, sigma),
                      utils.gaussian_2d_radial(2, sigma)]
    func_out = single_point_at_zero_2d.radial_profile(sigma, [0, 1, 2],
                                                      num_each, [0, 0])
    new_radii, new_densities = func_out
    assert np.allclose(true_densities, new_densities)
    assert np.allclose([0, 0, 1, 1, 2, 2], new_radii)

        
def test_radial_profile_multiple_kernel_sizes_1d(single_point_at_zero_1d):
    radii = [1, 2, 3]
    kernels = [0.5, 0.6, 0.7]
    true_densities = [utils.gaussian_1d(r, s) * 2 # same reason as above
                      for r, s in zip(radii, kernels)]
    _, test_densities = single_point_at_zero_1d.radial_profile(kernels, radii,
                                                               1, [0])
    assert np.allclose(true_densities, test_densities)

def test_radial_profile_multiple_kernel_sizes_2d(single_point_at_zero_2d):
    radii = [1, 2, 3]
    kernels = [0.5, 0.6, 0.7]
    true_densities = [utils.gaussian_2d_radial(r, s)
                      for r, s in zip(radii, kernels)]
    _, test_densities = single_point_at_zero_2d.radial_profile(kernels, radii,
                                                               1, [0, 0])
    assert np.allclose(true_densities, test_densities)

@pytest.mark.parametrize("loc", [0, 1, 2, 10, 70])
def test_integrate_over_kde_with_weights_1D_single(loc):
    """Test that when we integrate over a weighted KDE profile, we do indeed
    get the total mass back, like we should. This is true no matter what
    radius the particle is at. """
    # setup
    total_mass = np.random.uniform(1, 5)  # doesn't matter
    kde_obj = kde.KDE([[loc]], values=[total_mass])

    # make the profile
    x_values = np.arange(0, 100, 0.01)
    kernel_size = np.random.uniform(1, 5)  # doesn't matter
    _, profile = kde_obj.radial_profile(kernel_size, x_values, num_each=1,
                                        center=[0])
    # then integrate it
    integral = integrate.simps(y=profile, x=x_values)

    assert np.isclose(integral, total_mass)

@pytest.mark.parametrize("loc", [[0, 3],
                                 [1, 2],
                                 [2, 10],
                                 [10, 20],
                                 [70, 50]])
def test_integrate_over_kde_with_weights_1D_multiple(loc):
    """Test that when we integrate over a weighted KDE profile, we do indeed
    get the total mass back, like we should. This is true no matter what
    radius the particle is at. """
    # setup
    masses = np.random.uniform(1, 5, 2)  # doesn't matter
    total_mass = np.sum(masses)
    kde_obj = kde.KDE([loc], values=masses)

    # make the profile
    x_values = np.arange(0, 100, 0.005)
    kernel_size = np.random.uniform(1, 5)  # doesn't matter
    _, profile = kde_obj.radial_profile(kernel_size, x_values, num_each=1,
                                        center=[0])
    # then integrate it
    integral = integrate.simps(y=profile, x=x_values)

    assert np.isclose(integral, total_mass)


