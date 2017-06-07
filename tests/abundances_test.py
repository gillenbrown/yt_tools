from yt_tools import abundances

import pytest
import numpy as np

def test_init_error_checking_length():
    """Lists have to be the same length."""
    with pytest.raises(ValueError):
        abundances.Abundances([1, 2, 3], [0.2, 0.4], [0.5, 0.3])
    with pytest.raises(ValueError):
        abundances.Abundances([1, 3], [0.2, 0.4, 0.4], [0.5, 0.3])
    with pytest.raises(ValueError):
        abundances.Abundances([1, 2], [0.2, 0.4], [0.5, 0.3, 0.3])
    abundances.Abundances([1, 2], [0.2, 0.4], [0.3, 0.3])

def test_init_error_checking_positive_masses():
    """Masses have to be positive"""
    with pytest.raises(ValueError):
        abundances.Abundances([-1, 2], [0.1, 0.1], [0.1, 0.1])

def test_init_error_checking_z_bounds():
    """Metallicities must be between zero and one."""
    with pytest.raises(ValueError):
        abundances.Abundances([1, 2], [-0.1, 0.1], [0.1, 0.1])
    with pytest.raises(ValueError):
        abundances.Abundances([1, 2], [0.1, 0.1], [-0.1, 0.1])
    with pytest.raises(ValueError):
        abundances.Abundances([1, 2], [1.1, 0.1], [0.1, 0.1])
    with pytest.raises(ValueError):
        abundances.Abundances([1, 2], [0.1, 0.1], [1.1, 0.1])
    # the sum of the metallicity can't be larger than one, either.
    with pytest.raises(ValueError):
        abundances.Abundances([1, 2], [0.6, 0.6], [0.6, 0.2])

def test_init_value_parsing():
    """I do a few things with the values in the __init__ to be used later."""
    abund = abundances.Abundances([1, 2], [0.1, 0.2], [0.3, 0.4])
    assert np.array_equal(abund.mass, [1, 2])
    assert np.array_equal(abund.Z_Ia, [0.1, 0.2])
    assert np.array_equal(abund.Z_II, [0.3, 0.4])
    assert np.allclose(abund.Z_tot, [0.4, 0.6])
    assert np.allclose(abund.one_minus_Z_tot, [0.6, 0.4])
    assert abund.yields_Ia is not None
    assert abund.yields_II is not None

def test_solar_fractions():
    """Checks whether the solar abundances are reasonable. """
    abund = abundances.Abundances([1, 2], [0.1, 0.2], [0.3, 0.4])
    assert 0.015 < abund.z_sun < 0.02

    metal_fracs_sum = np.sum(abund.solar_metal_fractions.values())
    metal_fracs_sum -= abund.solar_metal_fractions["H"]
    metal_fracs_sum -= abund.solar_metal_fractions["He"]
    assert np.isclose(metal_fracs_sum, 1)

solar_z = abundances.create_solar_metal_fractions()[0]

@pytest.fixture
def single_mass_zero():
    return abundances.Abundances([3]*4, [0]*4, [0]*4)

@pytest.fixture
def single_mass_one_Ia():
    return abundances.Abundances([3]*4, [1]*4, [0]*4)

@pytest.fixture
def single_mass_one_II():
    return abundances.Abundances([3]*4, [0]*4, [1]*4)

@pytest.fixture
def single_mass_solar_Ia():
    return abundances.Abundances([3]*4, [0]*4, [solar_z]*4)

@pytest.fixture
def single_mass_solar_II():
    return abundances.Abundances([3]*4, [0]*4, [solar_z]*4)

@pytest.fixture
def two_star_different_z():
    return abundances.Abundances([2, 3], [0.1, 0.2], [0.05, 0.15])

def test_z_on_h_calculation_single_zero(single_mass_zero):
    """For z zero metallicity object we should have negative infinity. """
    assert np.isneginf(single_mass_zero.z_on_h())

def test_z_on_h_calculation_single_one_Ia(single_mass_one_Ia):
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(single_mass_one_Ia.z_on_h())

def test_z_on_h_calculation_single_one_II(single_mass_one_II):
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(single_mass_one_II.z_on_h())

def test_z_on_h_calculation_single_solar_Ia(single_mass_solar_Ia):
    """For a solar metallicity object we should get zero. """
    assert np.isclose(single_mass_solar_Ia.z_on_h(), 0)

def test_z_on_h_calculation_single_solar_II(single_mass_solar_II):
    """For a solar metallicity object we should get zero. """
    assert np.isclose(single_mass_solar_II.z_on_h(), 0)

def test_z_on_h_calculation_not_simple(two_star_different_z):
    """This calculation was done by hand. """
    assert np.isclose(two_star_different_z.z_on_h(), 1.3502)
