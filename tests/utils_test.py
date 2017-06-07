from yt_tools import utils
from yt.units import pc
import numpy as np

import pytest

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
