from yt_tools import utils
from yt.units import pc
import numpy as np

from scipy.integrate import quad

import pytest

# -----------------------------------------------------------

#  Test the distance calculation

# -----------------------------------------------------------

def test_distance_zero_dist():
    """Tests the distance calculation for the trivial case"""
    assert utils.distance(0, 0, 0, 0) == 0

def test_distance_2D():
    """Tests the distance calculation in 2D in a couple ways"""
    assert np.isclose(utils.distance(2, 1, 2, 1), np.sqrt(2))
    # make sure the squaring is working properly
    assert np.isclose(utils.distance(3, 1, 3, 1), np.sqrt(8))

def test_distance_3D():
    """Tests the distance calculation in 3D in a couple ways"""
    assert np.isclose(utils.distance(2, 1, 2, 1, 2, 1), np.sqrt(3))
    # make sure order doesn't matter
    assert np.isclose(utils.distance(1, 3, 1, 3, 1, 3), np.sqrt(12))

def test_distance_array():
    """Test the distance calculation when using arrays"""
    kde_dist = utils.distance(np.array([1, 1]), 0, np.array([1, 1]), 0)
    real_dist = np.array([np.sqrt(2), np.sqrt(2)])
    assert np.allclose(kde_dist, real_dist)

    kde_dist = utils.distance(np.array([2, 1, 3]), 1, np.array([1, -3, 5]), 2)
    real_dist = np.array([np.sqrt(2), np.sqrt(25), np.sqrt(13)])
    assert np.allclose(kde_dist, real_dist)

# -----------------------------------------------------------

#  Testing the radial Gaussian functions

# -----------------------------------------------------------

# we will be checking the results of various integrals. Since there is some
# numerical error on these, we have to have some tolerance here. Scipy reports
# errors, but we want to be generous on the size of our tolerance. This factor
# will be multiplied by the given error, and the difference must be within
# this range to be correct.
error_tolerance = 10

# I am going to be integrating in spherical and cylindrical space, so I will
# create these integrands beforehand to make life easier.
def spherical_integrand(r, func, sigma):
    return func(r, sigma) * 4 * np.pi * r**2

def cylindrical_integrad(r, func, sigma):
    return func(r, sigma) * 2 * np.pi * r

def test_3d_radial():
    """Test the integration of the 3D Gaussian function"""
    sigma = np.random.uniform(0.001, 100)
    integral, error = quad(spherical_integrand, 0, np.infty,
                           args=(utils.gaussian_3d_radial, sigma))
    assert(np.isclose(1, integral, atol=error*error_tolerance, rtol=0))

def test_3d_radial():
    """Test the integration of the 3D Gaussian function"""
    sigma = np.random.uniform(0.001, 100)
    integral, error = quad(cylindrical_integrad, 0, np.infty,
                           args=(utils.gaussian_2d_radial, sigma))
    assert(np.isclose(1, integral, atol=error*error_tolerance, rtol=0))

# I also want to test the error checking for these Gaussian functions.
@pytest.mark.parametrize("func", [utils.gaussian_2d_radial,
                                  utils.gaussian_3d_radial])
def test_gaussian_2d_error_checking_radius(func):
    """The radius must be positive, whether we have an array or one value"""
    with pytest.raises(RuntimeError):
        func(-1, 1)
    with pytest.raises(RuntimeError):
        func(np.array([0, 1, 2, 3, -5]), 1)

@pytest.mark.parametrize("func", [utils.gaussian_2d_radial,
                                  utils.gaussian_3d_radial])
def test_gaussian_2d_error_checking_sigma(func):
    """The standard deviation must be positive"""
    with pytest.raises(ValueError):
        func(1, 0)
    with pytest.raises(ValueError):
        func(1, -1)

def test_radial_gaussian_2d_actual_points():
    """Test against points plugged into calculator after deriving formula
    manually again. """
    assert np.isclose(utils.gaussian_2d_radial(2, 5), 0.0058767412)
    assert np.isclose(utils.gaussian_2d_radial(7, 5), 0.0023893047)
    assert np.isclose(utils.gaussian_2d_radial(2.3, 7.4), 0.0027693608)
    assert np.isclose(utils.gaussian_2d_radial(9.5, 7.4), 0.0012749001)

def test_radial_gaussian_3d_actual_points():
    """Test against points plugged into calculator after deriving formula
    manually again. """
    assert np.isclose(utils.gaussian_3d_radial(2, 5), 4.688961058E-4)
    assert np.isclose(utils.gaussian_3d_radial(7, 5), 1.906389302E-4)
    assert np.isclose(utils.gaussian_3d_radial(2.3, 7.4), 1.492993391E-4)
    assert np.isclose(utils.gaussian_3d_radial(9.5, 7.4), 6.873129013E-5)

# test other stuff

def test_unit_checking():
    with pytest.raises(TypeError):
        utils.test_for_units([1], "")
    with pytest.raises(TypeError):
        utils.test_for_units(1, "")
    utils.test_for_units(1*pc, "")  # will pass, return nothing

# create an object with a non-YT units thing, just to check that that part
# works too.
class bad_units(object):
    def __init__(self):
        self.units = "kg"

def test_unit_checking_other_units():
    with pytest.raises(TypeError):
        utils.test_for_units(bad_units(), "")

def test_spherical_to_cartesian():
    assert np.allclose(utils.convert_spherical_to_cartesian(10, 0, 0),
                       (0, 0, 10))
    assert np.allclose(utils.convert_spherical_to_cartesian(10, np.pi/2.0, 0),
                       (10, 0, 0))
    assert np.allclose(utils.convert_spherical_to_cartesian(10, np.pi / 2.0, np.pi / 2.0),
                       (0, 10, 0))
    assert np.allclose(utils.convert_spherical_to_cartesian(10, np.pi / 4.0, - np.pi / 4.0),
                       (5, -5, 10 / np.sqrt(2)))

def test_polar_to_cartesian():
    assert np.allclose(utils.convert_polar_to_cartesian(10, 0),
                       (10, 0))
    assert np.allclose(utils.convert_polar_to_cartesian(10, np.pi/2.0),
                       (0, 10))
    assert np.allclose(utils.convert_polar_to_cartesian(5, np.pi / 4.0),
                       (5.0 / np.sqrt(2), 5.0 / np.sqrt(2)))


# -----------------------------------------------------------------------------

# test the weighted mean

# -----------------------------------------------------------------------------

def test_weighted_mean_error_checking_length():
    """Weights and values have to be the same length."""
    with pytest.raises(ValueError):
        utils.weighted_mean([1, 2, 3], [1, 2])  # different lengths

def test_weighted_mean_error_checking_weights():
    """Weights cannot be negative."""
    with pytest.raises(ValueError):
        utils.weighted_mean([1, 2, 3], [-1, 2, 3])

def test_weighted_mean_all_same_weight():
    values = np.random.uniform(0, 1, 1000)
    weights = np.ones(1000)
    real_mean = np.mean(values)
    weighted_mean_1 = utils.weighted_mean(values, weights)
    weighted_mean_2 = utils.weighted_mean(values, weights * 2)
    assert np.isclose(real_mean, weighted_mean_1)
    assert np.isclose(real_mean, weighted_mean_2)

def test_weighted_mean_with_lists():
    """Lists should work just as well as arrays. """
    values = [0., 1., 2., 3.]
    weights = [4., 5., 6., 7.]
    assert np.isclose(utils.weighted_mean(values, weights),
                      utils.weighted_mean(np.array(values), np.array(weights)))

def test_weighted_mean_simple_calculations():
    """Test a simple version of the weighted mean. """
    values = [1, 2]
    weights = [1, 2]
    assert np.isclose(utils.weighted_mean(values, weights), 5.0 / 3.0)

    weights = [0, 2]
    assert np.isclose(utils.weighted_mean(values, weights), 2)

    values = [1, 2, 3]
    weights = [3, 2, 1]
    assert np.isclose(utils.weighted_mean(values, weights), 5.0 / 3.0)

def test_weighted_mean_example():
    """Shows how a weighted mean is just like a regular mean with repeats."""
    unweighted_values = [2, 2, 3, 3, 3, 4]
    weighted_vales = [2, 3, 4]
    weights = [2, 3, 1]
    assert np.isclose(np.mean(unweighted_values),
                      utils.weighted_mean(weighted_vales, weights))

# -----------------------------------------------------------------------------

# test the weighted variance

# -----------------------------------------------------------------------------

def test_weighted_variance_error_checking_length():
    """Weights and values have to be the same length."""
    with pytest.raises(ValueError):
        utils.weighted_variance([1, 2, 3], [1, 2])  # different lengths

def test_weighted_variance_error_checking_weights():
    """Weights cannot be negative."""
    with pytest.raises(ValueError):
        utils.weighted_variance([1, 2, 3], [-1, 2, 3])

def test_weighted_variance_all_same_weight():
    values = np.random.uniform(0, 1, 1000)
    weights = np.ones(1000)
    real_var = np.var(values, ddof=0)
    weighted_var_1 = utils.weighted_variance(values, weights, ddof=0)
    weighted_var_2 = utils.weighted_variance(values, weights * 2, ddof=0)
    assert np.isclose(real_var, weighted_var_1)
    assert np.isclose(real_var, weighted_var_2)

def test_weighted_variance_with_lists():
    """Lists should work just as well as arrays. """
    values = [0., 1., 2., 3.]
    weights = [4., 5., 6., 7.]
    assert np.isclose(utils.weighted_variance(values, weights),
                      utils.weighted_variance(np.array(values),
                                              np.array(weights)))

def test_weighted_variance_simple_calculations():
    """Test a simple version of the weighted variance. """
    values = [1, 2]
    weights = [1, 2]
    assert np.isclose(utils.weighted_variance(values, weights), 1.0 / 3.0)

    weights = [0, 2]
    assert np.isclose(utils.weighted_variance(values, weights), 0)


def test_weighted_variance_example():
    """Shows how a weighted variance is just like a regular variance with
    repeats."""
    unweighted_values = [2, 2, 2, 3, 3, 3, 3, 4, 4]
    weighted_vales = [2, 3, 4]
    weights = [3, 4, 2]
    assert np.isclose(np.var(unweighted_values, ddof=1),
                      utils.weighted_variance(weighted_vales, weights, ddof=1))
    assert np.isclose(np.var(unweighted_values, ddof=0),
                      utils.weighted_variance(weighted_vales, weights, ddof=0))

# -----------------------------------------------------------------------------

# test sum in quadrature

# -----------------------------------------------------------------------------

def test_sum_in_quadrature_single_values():
    """Should get the value back"""
    assert np.isclose(utils.sum_in_quadrature(5), 5)
    assert np.isclose(utils.sum_in_quadrature(np.array([5])), 5)

def test_sum_in_quadrature_scalars():
    assert np.isclose(utils.sum_in_quadrature(1, 1, 1), np.sqrt(3))
    assert np.isclose(utils.sum_in_quadrature(2, 2, 2), np.sqrt(12))
    assert np.isclose(utils.sum_in_quadrature(1, 2, 3), np.sqrt(14))
    assert np.isclose(utils.sum_in_quadrature(2, 3), np.sqrt(13))

def test_sum_in_quadrature_list():
    assert np.isclose(utils.sum_in_quadrature([1, 1, 1]), np.sqrt(3))
    assert np.isclose(utils.sum_in_quadrature([2, 2, 2]), np.sqrt(12))
    assert np.isclose(utils.sum_in_quadrature([1, 2, 3]), np.sqrt(14))
    assert np.isclose(utils.sum_in_quadrature([2, 3]), np.sqrt(13))

def test_sum_in_quadrature_array():
    assert np.isclose(utils.sum_in_quadrature(np.array([1, 1, 1])), np.sqrt(3))
    assert np.isclose(utils.sum_in_quadrature(np.array([2, 2, 2])), np.sqrt(12))
    assert np.isclose(utils.sum_in_quadrature(np.array([1, 2, 3])), np.sqrt(14))
    assert np.isclose(utils.sum_in_quadrature(np.array([2, 3])), np.sqrt(13))

# -----------------------------------------------------------------------------

# test binning

# -----------------------------------------------------------------------------

def test_binning_single_values():
    assert np.allclose(utils.bin_values([1, 1], bin_size=1), [1, 1])
    assert np.allclose(utils.bin_values([4.5, 2.3], bin_size=1), [4.5, 2.3])

def test_binning_even_bin_size():
    assert np.allclose(utils.bin_values([1, 1, 2, 2, 3, 3], 2), [1, 2, 3])
    assert np.allclose(utils.bin_values([1, 2, 3, 4, 5, 6, 7, 8, 9], 3),
                       [2, 5, 8])
    assert np.allclose(utils.bin_values([1, 2, 3, 4, 5, 6, 7, 8], 2),
                       [1.5, 3.5, 5.5, 7.5])

def test_binning_not_even_bins():
    assert np.allclose(utils.bin_values([1, 2, 3], 2), [2])
    assert np.allclose(utils.bin_values([1, 2, 3, 4, 5], 2), [1.5, 4])
    assert np.allclose(utils.bin_values([1, 2, 3, 4, 5, 6, 7, 8], 3),
                       [2, 6])

def test_binning_length_size():
    assert len(utils.bin_values(np.arange(100), 10)) == 10
    assert len(utils.bin_values(np.arange(101), 10)) == 10
    assert len(utils.bin_values(np.arange(109), 10)) == 10

def test_intersection_error_checking():
    with pytest.raises(ValueError):
        utils.sphere_intersection(-10, 5, 8)
    with pytest.raises(ValueError):
        utils.sphere_intersection(10, -5, 8)
    with pytest.raises(ValueError):
        utils.sphere_intersection(10, 5, -8)

def test_intersection_values():
    assert utils.sphere_intersection(10, 10, 0)
    assert utils.sphere_intersection(10, 5, 8)
    assert utils.sphere_intersection(10, 5, 12)
    assert not utils.sphere_intersection(10, 5, 4.9)
    assert not utils.sphere_intersection(10, 5, 15.1)
    # flip order of radii
    assert utils.sphere_intersection(5, 10, 8)
    assert utils.sphere_intersection(5, 10, 12)
    assert not utils.sphere_intersection(5, 10, 4.9)
    assert not utils.sphere_intersection(5, 10, 15.1)

    # other values
    assert utils.sphere_intersection(15, 23, 20)
    assert not utils.sphere_intersection(15, 23, 0)
    assert not utils.sphere_intersection(15, 23, 7.9)
    assert not utils.sphere_intersection(15, 23, 38.1)
    # swap values
    assert utils.sphere_intersection(23, 15, 20)
    assert not utils.sphere_intersection(23, 15, 0)
    assert not utils.sphere_intersection(23, 15, 7.9)
    assert not utils.sphere_intersection(23, 15, 38.1)
    # edeg cases count
    assert utils.sphere_intersection(10, 5, 5)
    assert utils.sphere_intersection(10, 5, 15)
    assert utils.sphere_intersection(5, 10, 5)
    assert utils.sphere_intersection(5, 10, 15)

def test_sphere_contaiment_error_checking():
    # same as sphere intersection
    # centers need to be a three element thing
    with pytest.raises(TypeError):
        utils.sphere_containment(1, 1, 1, 1)
    with pytest.raises(ValueError):
        utils.sphere_containment([1], [1], 1, 1)
    with pytest.raises(ValueError):
        utils.sphere_containment([1, 2, 3, 4], [1, 2, 3, 4], 1, 1)
    # radius can't be negative.
    with pytest.raises(ValueError):
        utils.sphere_containment([1, 2, 3], [1, 2, 3], 1, -1)

def test_sphere_containment():
    """Tests whether one sphere is contained in the other."""
    cen_1 = [0, 0, 0]
    cen_2 = [1, 1, 1]
    assert utils.sphere_containment(cen_1, cen_1, 1, 2)
    assert not utils.sphere_containment(cen_1, cen_1, 2, 1)  # order matters
    assert not utils.sphere_containment(cen_1, cen_1, 2, 2)  # same size

    # then vary location. After each test here where it should pass, I will
    # flip the order so that it wont'.
    assert utils.sphere_containment(cen_1, cen_2, 1, 5)
    assert not utils.sphere_containment(cen_2, cen_2, 5, 1)

    assert not utils.sphere_containment(cen_1, cen_2, 1, 1)  # they intersect

    assert not utils.sphere_containment(cen_1, cen_2, 0.1, 0.1)  # don't touch
    assert not utils.sphere_containment(cen_1, cen_2, 0.2, 0.1)  # don't touch
    assert not utils.sphere_containment(cen_1, cen_2, 0.1, 0.2)  # don't touch

def test_kernel_sizes():

