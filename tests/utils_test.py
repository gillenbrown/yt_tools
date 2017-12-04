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
    assert utils.distance(0, 0) == 0

def test_distance_1D():
    """Tests the distance calculation in 2D in a couple ways"""
    assert np.isclose(utils.distance(2, 1), 1)
    # make sure the squaring is working properly
    assert np.isclose(utils.distance(3, 1), 2)

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
    kde_dist = utils.distance(np.array([1, 2]), 1)
    real_dist = np.array([0, 1])
    assert np.allclose(kde_dist, real_dist)

    kde_dist = utils.distance(np.array([1, 2]), 0, np.array([1, 2]), 0)
    real_dist = np.array([np.sqrt(2), np.sqrt(8)])
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

def linear_integrand(r, func, sigma):
    return func(r, sigma)

def test_3d_radial_integration():
    """Test the integration of the 3D Gaussian function"""
    sigma = np.random.uniform(0.001, 100)
    integral, error = quad(spherical_integrand, 0, np.infty,
                           args=(utils.gaussian_3d_radial, sigma))
    assert(np.isclose(1, integral, atol=error*error_tolerance, rtol=0))

def test_2d_radial_integration():
    """Test the integration of the 2D Gaussian function"""
    sigma = np.random.uniform(0.001, 100)
    integral, error = quad(cylindrical_integrad, 0, np.infty,
                           args=(utils.gaussian_2d_radial, sigma))
    assert(np.isclose(1, integral, atol=error*error_tolerance, rtol=0))

def test_1d_radial_integration():
    """Test the integration of the 1D Gaussian function"""
    sigma = np.random.uniform(0.001, 100)
    integral, error = quad(linear_integrand, -1 * np.infty, np.infty,
                           args=(utils.gaussian_1d, sigma))
    assert(np.isclose(1, integral, atol=error*error_tolerance, rtol=0))

# I also want to test the error checking for these Gaussian functions.
@pytest.mark.parametrize("func", [utils.gaussian_2d_radial,  # 1D can handle -
                                  utils.gaussian_3d_radial])
def test_gaussian_2d_error_checking_radius(func):
    """The radius must be positive, whether we have an array or one value"""
    with pytest.raises(RuntimeError):
        func(-1, 1)
    with pytest.raises(RuntimeError):
        func(np.array([0, 1, 2, 3, -5]), 1)

@pytest.mark.parametrize("func", [utils.gaussian_1d,
                                  utils.gaussian_2d_radial,
                                  utils.gaussian_3d_radial])
def test_gaussian_error_checking_sigma(func):
    """The standard deviation must be positive"""
    with pytest.raises(RuntimeError):
        func(1, 0)
    with pytest.raises(RuntimeError):
        func(1, -1)

@pytest.mark.parametrize("func", [utils.gaussian_1d,
                                  utils.gaussian_2d_radial,
                                  utils.gaussian_3d_radial])
def test_gaussian_error_checking_lengths(func):
    test_array = np.array([1, 2, 3])
    # if single radii and sigma, that's cool
    func(1, 1)
    # multiple radii and one sigma is cool
    func(test_array, 1)
    # multiple sigma and one radii is not cool
    with pytest.raises(RuntimeError):
        func(1, test_array)
    # multiple sigma and radii is cool if they have the same length...
    func(test_array, test_array)
    # ...but not if they are different length
    with pytest.raises(RuntimeError):
        func(test_array, np.array([1, 2, 3, 4]))


def test_radial_gaussian_1d_actual_points():
    """Test against points plugged into calculator."""
    assert np.isclose(utils.gaussian_1d(2, 5), 0.073654028)
    assert np.isclose(utils.gaussian_1d(7, 5), 0.029945493)
    assert np.isclose(utils.gaussian_1d(2.3, 7.4), 0.05136901)
    assert np.isclose(utils.gaussian_1d(9.5, 7.4), 0.023648184)

def test_radial_gaussian_1d_actual_points_radii_array():
    """Same as previous function, but radii is put in an array. """
    x_values = np.array([2, 7])
    sigma = 5
    test_densities = utils.gaussian_1d(x_values, sigma)
    true_densities = [0.073654028, 0.029945493]
    assert np.allclose(test_densities, true_densities)

    x_values = np.array([2.3, 9.5])
    sigma = 7.4
    test_densities = utils.gaussian_1d(x_values, sigma)
    true_densities = [0.05136901, 0.023648184]
    assert np.allclose(test_densities, true_densities)

def test_radial_gaussian_1d_actual_points_both_array():
    """Same as previous function, but everything is put in an array. """
    x_values = np.array([2, 7, 2.3, 9.5])
    sigmas = np.array([5, 5, 7.4, 7.4])
    test_densities = utils.gaussian_1d(x_values, sigmas)
    true_densities = [0.073654028, 0.029945493, 0.05136901, 0.023648184]
    assert np.allclose(test_densities, true_densities)

def test_radial_gaussian_2d_actual_points():
    """Test against points plugged into calculator after deriving formula
    manually again. """
    assert np.isclose(utils.gaussian_2d_radial(2, 5), 0.0058767412)
    assert np.isclose(utils.gaussian_2d_radial(7, 5), 0.0023893047)
    assert np.isclose(utils.gaussian_2d_radial(2.3, 7.4), 0.0027693608)
    assert np.isclose(utils.gaussian_2d_radial(9.5, 7.4), 0.0012749001)

def test_radial_gaussian_2d_actual_points_radii_array():
    """Same as previous function, but radii is put in an array. """
    radii = np.array([2, 7])
    sigma = 5
    test_densities = utils.gaussian_2d_radial(radii, sigma)
    true_densities = [0.0058767412, 0.0023893047]
    assert np.allclose(test_densities, true_densities)

    radii = np.array([2.3, 9.5])
    sigma = 7.4
    test_densities = utils.gaussian_2d_radial(radii, sigma)
    true_densities = [0.0027693608, 0.0012749001]
    assert np.allclose(test_densities, true_densities)

def test_radial_gaussian_2d_actual_points_both_array():
    """Same as previous function, but everything is put in an array. """
    radii = np.array([2, 7, 2.3, 9.5])
    sigmas = np.array([5, 5, 7.4, 7.4])
    test_densities = utils.gaussian_2d_radial(radii, sigmas)
    true_densities = [0.0058767412, 0.0023893047, 0.0027693608, 0.0012749001]
    assert np.allclose(test_densities, true_densities)

def test_radial_gaussian_3d_actual_points():
    """Test against points plugged into calculator after deriving formula
    manually again. """
    assert np.isclose(utils.gaussian_3d_radial(2, 5), 4.688961058E-4)
    assert np.isclose(utils.gaussian_3d_radial(7, 5), 1.906389302E-4)
    assert np.isclose(utils.gaussian_3d_radial(2.3, 7.4), 1.492993391E-4)
    assert np.isclose(utils.gaussian_3d_radial(9.5, 7.4), 6.873129013E-5)

def test_radial_gaussian_3d_actual_points_radii_array():
    """Same as previous function, but radii is put in an array. """
    radii = np.array([2, 7])
    sigma = 5
    test_densities = utils.gaussian_3d_radial(radii, sigma)
    true_densities = [4.688961058E-4, 1.906389302E-4]
    assert np.allclose(test_densities, true_densities)

    radii = np.array([2.3, 9.5])
    sigma = 7.4
    test_densities = utils.gaussian_3d_radial(radii, sigma)
    true_densities = [1.492993391E-4, 6.873129013E-5]
    assert np.allclose(test_densities, true_densities)

def test_radial_gaussian_3d_actual_points_both_array():
    """Same as previous function, but everything is put in an array. """
    radii = np.array([2, 7, 2.3, 9.5])
    sigmas = np.array([5, 5, 7.4, 7.4])
    test_densities = utils.gaussian_3d_radial(radii, sigmas)
    true_densities = [4.688961058E-4, 1.906389302E-4,
                      1.492993391E-4, 6.873129013E-5]
    assert np.allclose(test_densities, true_densities)



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

# def test_2d_polar_points_single_direction():
#     """Get the polar points for one direction, which will be the x direction"""
#     x_func, y_func = utils.get_2d_polar_points([1, 2], 1)
#     x_real, y_real = [1, 2], [0, 0]
#     assert np.allclose(x_func, x_real)
#     assert np.allclose(y_func, y_real)
#
# def test_2d_polar_points_4_directions():
#     """This will put points on all 4 axis directions.
#     They need to be ordered in terms of increasing radii, with all points with
#     the same radius next to each other."""
#     x_func, y_func = utils.get_2d_polar_points([1, 2], 4)
#     x_real = [1, 0, -1, 0, 2, 0, -2, 0]
#     y_real = [0, 1, 0, -1, 0, 2, 0, -2]
#     assert np.allclose(x_func, x_real)
#     assert np.allclose(y_func, y_real)

def test_2d_polar_points_many_directions_lengths():
    """Sees if the points have the right amount."""
    num_radii = 234
    radii = np.linspace(0, 100, num_radii)
    num_angles = 634
    x, y = utils.get_2d_polar_points(radii, num_angles)
    assert len(x) == num_radii * num_angles
    assert len(y) == num_radii * num_angles


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

def test_binning_single_values_a():
    assert np.allclose(utils.bin_values([1, 1], bin_size=1), [1, 1])

def test_binning_single_values_b():
    means = utils.bin_values([4.5, 2.3], bin_size=1)
    assert np.allclose(means, [4.5, 2.3])

def test_binning_even_bin_size_a():
    means = utils.bin_values([1, 1, 2, 2, 3, 3], 2)
    assert np.allclose(means, [1, 2, 3])

def test_binning_even_bin_size_b():
    means = utils.bin_values([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    assert np.allclose(means, [2, 5, 8])

def test_binning_even_bin_size_c():
    means = utils.bin_values([1, 2, 3, 4, 5, 6, 7, 8], 2)
    assert np.allclose(means, [1.5, 3.5, 5.5, 7.5])

def test_binning_not_even_bins_a():
    means = utils.bin_values([1, 2, 3], 2)
    assert np.allclose(means, [2])

def test_binning_not_even_bins_b():
    means = utils.bin_values([1, 2, 3, 4, 5], 2)
    assert np.allclose(means, [1.5, 4])

def test_binning_not_even_bins_c():
    means = utils.bin_values([1, 2, 3, 4, 5, 6, 7, 8], 3)
    assert np.allclose(means, [2, 6])

@pytest.mark.parametrize("length,bin_size,bins", [
    (99,  10, 9 ),
    (100, 10, 10),
    (101, 10, 10),
    (109, 10, 10),
    (110, 10, 11)
])
def test_binning_length_size(length, bin_size, bins):
    means = utils.bin_values(np.arange(length), bin_size)
    assert len(means) == bins

# -----------------------------------------------------------------------------

# sphere intersection

# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------

# test mean in log space

# -----------------------------------------------------------------------------

def test_log_mean_simple_case_positive_log_a():
    assert np.isclose(utils.log_mean(1, 100), 10)

def test_log_mean_simple_case_positive_log_b():
    assert np.isclose(utils.log_mean(100, 10000), 1000)

def test_log_mean_simple_case_negative_log_a():
    assert np.isclose(utils.log_mean(0.001, 0.1), 0.01)

def test_log_mean_simple_case_negative_log_b():
    assert np.isclose(utils.log_mean(0.00001, 0.001), 0.0001)

def test_log_mean_less_easy_positive_1():
    assert np.isclose(utils.log_mean(1, 10), 3.16227766)  # 10^0.5 in calculator

def test_log_mean_less_easy_positive_b():
    assert np.isclose(utils.log_mean(314, 3268), 1012.991609)  # calculator

def test_log_mean_less_easy_negative_a():
    assert np.isclose(utils.log_mean(0.01, 0.1), 0.0316227766) # calculator

def test_log_mean_less_easy_negative_b():
    assert np.isclose(utils.log_mean(0.003458, 0.00008629), 0.000546252)

# -----------------------------------------------------------------------------

# test half mass upper limits

# -----------------------------------------------------------------------------

def test_more_than_half_false_a():
    assert not utils.max_above_half([1, 2, 1, 1, 1, 1])

def test_more_than_half_false_b():
    assert not utils.max_above_half([2, 2, 2, 2, 2, 10])

def test_more_than_half_true_a():
    assert utils.max_above_half([10, 1])

def test_more_than_half_true_b():
    assert utils.max_above_half([10, 5, 4])

# -----------------------------------------------------------------------------

# test annulus area

# -----------------------------------------------------------------------------

def test_annulus_error_checking():
    # neither radius can be below zero
    with pytest.raises(ValueError):
        utils.annulus_area(-1, 1)
    with pytest.raises(ValueError):
        utils.annulus_area(1, -1)
    with pytest.raises(ValueError):
        utils.annulus_area(-1, -1)

def test_annulus_area_around_zero():
    assert np.isclose(utils.annulus_area(0, 1), np.pi)

def test_annulus_area_same_radius():
    radius = np.random.uniform(1, 10, 1)
    assert np.isclose(utils.annulus_area(radius, radius), 0)

def test_annulus_area_actual_annulus_a():
    test_area = utils.annulus_area(10, 9)
    true_area = 59.69026042  # calculator
    assert np.isclose(test_area, true_area)

def test_annulus_area_actual_annulus_b():
    test_area = utils.annulus_area(5, 2)
    true_area = 65.97344573  # calculator
    assert np.isclose(test_area, true_area)

def test_annulus_area_symmetry():
    """The order of the operands shouldn't matter."""
    radius_a = np.random.uniform(1, 10)
    radius_b = np.random.uniform(1, 10)
    forwards = utils.annulus_area(radius_a, radius_b)
    backwards = utils.annulus_area(radius_b, radius_a)
    assert np.isclose(forwards, backwards)

# -----------------------------------------------------------------------------

# test annulus random points

# -----------------------------------------------------------------------------

def test_annulus_random_distribution_error_checking_positive_radii():
    with pytest.raises(ValueError):
        utils.generate_random_xy_annulus(-1, 1, 100)
    with pytest.raises(ValueError):
        utils.generate_random_xy_annulus(1, -1, 100)
    with pytest.raises(ValueError):
        utils.generate_random_xy_annulus(-1, -2, 100)
    utils.generate_random_xy_annulus(1, 2, 100)  # no error

def test_annulus_random_distibution_postive_number():
    with pytest.raises(ValueError):
        utils.generate_random_xy_annulus(1, 2, -100)
    utils.generate_random_xy_annulus(1, 2, 100)  # no error

def test_annulus_random_distribution_radii():
    inner_radius = np.random.uniform(1, 5, 1)
    outer_radius = np.random.uniform(5, 8, 1)
    x, y = utils.generate_random_xy_annulus(inner_radius, outer_radius, 1000)

    radii = np.sqrt(x**2 + y**2)
    assert np.all(inner_radius < radii)
    assert np.all(radii < outer_radius)

def test_annulus_random_distribution_lengths():
    inner_radius = np.random.uniform(1, 5, 1)
    outer_radius = np.random.uniform(5, 8, 1)
    x, y = utils.generate_random_xy_annulus(inner_radius, outer_radius, 1000)

    assert len(x) == len(y) == 1000

# then checked the evenness by eye.

# -----------------------------------------------------------------------------

# test annulus integration

# -----------------------------------------------------------------------------
def test_surface_density_error_checking_no_function():
    with pytest.raises(TypeError):
        utils.surface_density_annulus(1, 1, 2, 0.1)

def test_surface_density_positive_error():
    with pytest.raises(ValueError):
        utils.surface_density_annulus(lambda x, y:1, 1, 2, -1)

def test_surface_density_positive_radii():
    # should be takes care of by annulus point generation
    with pytest.raises(ValueError):
        utils.surface_density_annulus(lambda x, y:1, -1, 2, 1)
    with pytest.raises(ValueError):
        utils.surface_density_annulus(lambda x, y:1, -1, -2, 1)
    with pytest.raises(ValueError):
        utils.surface_density_annulus(lambda x, y:1, 1, -2, 1)
    utils.surface_density_annulus(lambda x, y:1, 1, 2, 1) # no error

@pytest.mark.parametrize("r_a,r_b,result", [
    (0, 1, 1),
    (1, 10, 1)
])
def test_surface_density_annulus_constant_density(r_a, r_b, result):
    def flat(x, y):
        return 1
    tolerance = 0.01
    integral = utils.surface_density_annulus(flat, r_a, r_b,
                                             error_tolerance=tolerance)
    assert np.isclose(integral, result, rtol=2*tolerance, atol=0)

@pytest.mark.parametrize("r_a,r_b,result", [
    (0, 1, 2.0/3.0),
    (3, 8, 5.87878787)
])
def test_surface_density_annulus_r(r_a, r_b, result):
    def radius(x, y):
        return np.sqrt(x**2 + y**2)

    tolerance = 0.01
    integral = utils.surface_density_annulus(radius, r_a, r_b,
                                             error_tolerance=tolerance)
    assert np.isclose(integral, result, rtol=2*tolerance, atol=0)

@pytest.mark.parametrize("r_a,r_b,result", [
    (0, 1, 0.5),
    (3, 8, 36.5)
])
def test_surface_density_annulus_r_squared(r_a, r_b, result):
    def radius_squared(x, y):
        return x ** 2 + y ** 2

    tolerance = 0.01
    integral = utils.surface_density_annulus(radius_squared, r_a, r_b,
                                             error_tolerance=tolerance)
    assert np.isclose(integral, result, rtol=2*tolerance, atol=0)

@pytest.mark.parametrize("r_a,r_b,dens", [
    (0, 1, 3.14234872),
    (1, 10, 8.64738)
])
def test_surface_density_annulus_constant_density_args(r_a, r_b, dens):
    # make sure this works with an argument.
    def flat(x, y, k):
        return k
    tolerance = 0.01
    integral = utils.surface_density_annulus(flat, r_a, r_b,
                                             error_tolerance=tolerance,
                                             density_func_kwargs={"k": dens})
    assert np.isclose(integral, dens, rtol=2*tolerance, atol=0)
