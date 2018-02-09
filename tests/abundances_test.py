from yt_tools import nsc_abundances

import pytest
import numpy as np

import yields

m_2 = [1, 2]
mzz0_2 = [0, 0]

def test_init_error_checking_length():
    """Lists have to be the same length."""
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances([1, 2, 3], [0.2, 0.4],
                                      [0.5, 0.3], [0, 0], [1, 2])
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances([1, 3], [0.2, 0.4, 0.4],
                                      [0.5, 0.3], [0, 0], [1, 2])
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances([1, 2], [0.2, 0.4],
                                      [0.5, 0.3, 0.3], [0, 0], [1, 2])
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances([1, 2], [0.2, 0.4],
                                      [0.5, 0.3], [0, 0, 0], [1, 2])
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances([1, 2], [0.2, 0.4],
                                      [0.5, 0.3], [0, 0], [1, 2, 2])

    # no error
    nsc_abundances.NSC_Abundances([1, 2], [0.2, 0.4], [0.3, 0.3],
                                  [0, 0], [1, 2])

def test_init_error_checking_positive_masses():
    """Masses have to be positive"""
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances([0, 2], [0.1, 0.1], [0.1, 0.1],
                                      [0, 0], [1, 1])
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances([1, -2], [0.1, 0.1], [0.1, 0.1],
                                      [0, 0], [1, 1])
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances([1, 2], [0.1, 0.1], [0.1, 0.1],
                                      [0, 0], [0, 1])
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances([1, 2], [0.1, 0.1], [0.1, 0.1],
                                      [0, 0], [-2, 1])

def test_init_error_checking_z_bounds():
    """Metallicities must be between zero and one."""
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances(m_2, [-0.1, 0.1], [0.1, 0.1], [0, 0], m_2)
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances(m_2, [0.1, 0.1], [-0.1, 0.1], [0, 0], m_2)
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances(m_2, [1.1, 0.1], [0.1, 0.1], [0, 0], m_2)
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances(m_2, [0.1, 0.1], [1.1, 0.1], [0, 0], m_2)
    # the sum of the metallicity can't be larger than one, either.
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances(m_2, [0.6, 0.6], [0.6, 0.2], [0, 0], m_2)

def test_init_error_checking_spread_bounds():
    with pytest.raises(ValueError):
        nsc_abundances.NSC_Abundances([1], [0.2], [0.2], [-0.2], [1])

def test_init_value_parsing():
    """I do a few things with the values in the __init__ to be used later."""
    abund = nsc_abundances.NSC_Abundances(m_2, [0.1, 0.2],
                                          [0.3, 0.4], [0.2, 0.7], m_2)
    assert np.allclose(abund.mass, [1, 2])
    assert np.allclose(abund.Z_Ia, [0.1, 0.2])
    assert np.allclose(abund.Z_II, [0.3, 0.4])
    assert np.allclose(abund.Z_tot, [0.4, 0.6])
    assert np.allclose(abund.one_minus_Z_tot, [0.6, 0.4])
    assert np.allclose(abund.mZZ_II, [0.2, 0.7])
    assert np.allclose(abund.initial_masses, [1, 2])
    assert abund.yields_Ia is not None
    assert abund.yields_II is not None
    assert abund.abund is not None

def test_sigma_z_value():
    abund = nsc_abundances.NSC_Abundances(m_2, [0.1, 0.2], [0.3, 0.4],
                                          [0.2, 0.8], m_2)
    assert np.allclose(abund.var_z_II_int_ind, [0.11, 0.24])

    abund = nsc_abundances.NSC_Abundances(m_2, [0.1, 0.2], [0.3, 0.4],
                                          [0, 0], m_2)
    assert np.allclose(abund.var_z_II_int_ind, [0, 0])

def test_solar_fractions():
    """Checks whether the solar abundances are reasonable. """
    abund = nsc_abundances.NSC_Abundances(m_2, [0.1, 0.2], [0.3, 0.4],
                                          [0, 0], m_2)
    assert 0.015 < abund.z_sun < 0.02

    metal_fracs_sum = np.sum(abund.solar_metal_fractions.values())
    metal_fracs_sum -= abund.solar_metal_fractions["H"]
    metal_fracs_sum -= abund.solar_metal_fractions["He"]
    assert np.isclose(metal_fracs_sum, 1)

# -----------------------------------------------------------

#  Test abundance calculations

# -----------------------------------------------------------

# setup
solar_z = yields.solar_z

@pytest.fixture
def single_mass_zero():
    return nsc_abundances.NSC_Abundances(masses=[3] * 4,
                                         Z_Ia=[0] * 4,
                                         Z_II=[0] * 4,
                                         mZZ_II=[0] * 4,
                                         m_i=[3] * 4)

@pytest.fixture
def single_mass_one_Ia():
    return nsc_abundances.NSC_Abundances(masses=[3] * 4,
                                         Z_Ia=[1] * 4,
                                         Z_II=[0] * 4,
                                         mZZ_II=[0] * 4,
                                         m_i=[3] * 4)

@pytest.fixture
def single_mass_one_II():
    return nsc_abundances.NSC_Abundances(masses=[3] * 4,
                                         Z_Ia=[0] * 4,
                                         Z_II=[1] * 4,
                                         mZZ_II=[0] * 4,
                                         m_i=[3] * 4)

@pytest.fixture
def single_mass_solar_Ia():
    return nsc_abundances.NSC_Abundances(masses=[3] * 4,
                                         Z_Ia=[solar_z] * 4,
                                         Z_II=[0] * 4,
                                         mZZ_II=[0] * 4,
                                         m_i=[3] * 4)

@pytest.fixture
def single_mass_solar_II():
    return nsc_abundances.NSC_Abundances(masses=[3] * 4,
                                         Z_Ia=[0] * 4,
                                         Z_II=[solar_z] * 4,
                                         mZZ_II=[0] * 4,
                                         m_i=[3] * 4)

@pytest.fixture
def two_star_different_z():
    return nsc_abundances.NSC_Abundances(masses=[2, 3],
                                         Z_Ia=[0.01, 0.02],
                                         Z_II=[0.005, 0.015],
                                         mZZ_II=[0, 0],
                                         m_i=[2, 3])

# -----------------------------------------------------------

#  Test [Z/H]

# -----------------------------------------------------------

def test_z_on_h_calculation_single_zero(single_mass_zero):
    """For z zero metallicity object we should have negative infinity. """
    assert np.isneginf(single_mass_zero.z_on_h_total())

def test_z_on_h_calculation_single_one_Ia(single_mass_one_Ia):
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(single_mass_one_Ia.z_on_h_total())

def test_z_on_h_calculation_single_one_II(single_mass_one_II):
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(single_mass_one_II.z_on_h_total())

def test_z_on_h_calculation_single_solar_Ia(single_mass_solar_Ia):
    """For a solar metallicity object we should get zero. """
    assert np.isclose(single_mass_solar_Ia.z_on_h_total(), 0)

def test_z_on_h_calculation_single_solar_II(single_mass_solar_II):
    """For a solar metallicity object we should get zero. """
    assert np.isclose(single_mass_solar_II.z_on_h_total(), 0)

def test_z_on_h_calculation_not_simple(two_star_different_z):
    """This calculation was done by hand. """
    assert np.isclose(two_star_different_z.z_on_h_total(), 0.225411791, atol=0)

# -----------------------------------------------------------

#  Test [Fe/H]

# -----------------------------------------------------------

def test_fe_on_h_calculation_single_zero_total(single_mass_zero):
    """For z zero metallicity object we should have negative infinity. """
    assert np.isneginf(single_mass_zero.x_on_h_total("Fe"))

# def test_fe_on_h_calculation_single_zero_average(single_mass_zero):
#     """For z zero metallicity object we should have negative infinity. """
#     assert np.isneginf(single_mass_zero.x_on_h_average("Fe")[0])

def test_fe_on_h_calculation_single_one_Ia_total(single_mass_one_Ia):
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(single_mass_one_Ia.x_on_h_total("Fe"))

# def test_fe_on_h_calculation_single_one_Ia_average(single_mass_one_Ia):
#     """For an object of metallicity one, we will get an infinite value, since
#     we will be dividing by zero. """
#     assert np.isposinf(single_mass_one_Ia.x_on_h_average("Fe")[0])

def test_fe_on_h_calculation_single_one_II_total(single_mass_one_II):
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(single_mass_one_II.x_on_h_total("Fe"))

# def test_fe_on_h_calculation_single_one_II_average(single_mass_one_II):
#     """For an object of metallicity one, we will get an infinite value, since
#     we will be dividing by zero. """
#     assert np.isposinf(single_mass_one_II.x_on_h_average("Fe")[0])

def test_fe_on_h_calculation_single_solar_Ia_total(single_mass_solar_Ia):
    """For a solar metallicity object we have to manually calculate it using
     the actual yield object. """
    assert np.isclose(single_mass_solar_Ia.x_on_h_total("Fe"), 0.858336, atol=0)

# def test_fe_on_h_calculation_single_solar_Ia_average(single_mass_solar_Ia):
#     """For a solar metallicity object we have to manually calculate it using
#      the actual yield object. """
#     assert np.allclose(single_mass_solar_Ia.x_on_h_average("Fe"), (0.858336, 0))

def test_fe_on_h_calculation_single_solar_II_total(single_mass_solar_II):
    """For a solar metallicity object we have to manually calculate it using
     the actual yield object. """
    assert np.isclose(single_mass_solar_II.x_on_h_total("Fe"), -0.2740616354)

# def test_fe_on_h_calculation_single_solar_II_average(single_mass_solar_II):
#     """For a solar metallicity object we have to manually calculate it using
#      the actual yield object. """
#     assert np.allclose(single_mass_solar_II.x_on_h_average("Fe"),
#                        (-0.2740616354, 0))

def test_fe_on_h_calculation_not_simple_total(two_star_different_z):
    """This calculation was done by hand. """
    assert np.isclose(two_star_different_z.x_on_h_total("Fe"),
                      0.8775378801, atol=0)


# -----------------------------------------------------------

#  Test [X/H] for different elements

# -----------------------------------------------------------

def test_na_on_h_calculation_single_zero_total(single_mass_zero):
    """For z zero metallicity object we should have negative infinity. """
    assert np.isneginf(single_mass_zero.x_on_h_total("Na"))

# def test_na_on_h_calculation_single_zero_average(single_mass_zero):
#     """For z zero metallicity object we should have negative infinity. """
#     assert np.isneginf(single_mass_zero.x_on_h_average("Na")[0])

def test_na_on_h_calculation_single_one_Ia_total(single_mass_one_Ia):
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(single_mass_one_Ia.x_on_h_total("Na"))

# def test_na_on_h_calculation_single_one_Ia_average(single_mass_one_Ia):
#     """For an object of metallicity one, we will get an infinite value, since
#     we will be dividing by zero. """
#     assert np.isposinf(single_mass_one_Ia.x_on_h_average("Na")[0])

def test_na_on_h_calculation_single_one_II_total(single_mass_one_II):
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(single_mass_one_II.x_on_h_total("Na"))

# def test_na_on_h_calculation_single_one_II_average(single_mass_one_II):
#     """For an object of metallicity one, we will get an infinite value, since
#     we will be dividing by zero. """
#     assert np.isposinf(single_mass_one_II.x_on_h_average("Na")[0])

def test_na_on_h_calculation_single_solar_Ia_total(single_mass_solar_Ia):
    """For a solar metallicity object we have to manually calculate it using
     the actual yield object. """
    assert np.isclose(single_mass_solar_Ia.x_on_h_total("Na"), -1.66083)

# def test_na_on_h_calculation_single_solar_Ia_average(single_mass_solar_Ia):
#     """For a solar metallicity object we have to manually calculate it using
#      the actual yield object. """
#     assert np.allclose(single_mass_solar_Ia.x_on_h_average("Na"),
#                        (-1.66083, 0))

def test_na_on_h_calculation_single_solar_II_total(single_mass_solar_II):
    """For a solar metallicity object we have to manually calculate it using
     the actual yield object. """
    assert np.isclose(single_mass_solar_II.x_on_h_total("Na"),
                      0.3078305, atol=0)

# def test_na_on_h_calculation_single_solar_II_average(single_mass_solar_II):
    # """For a solar metallicity object we have to manually calculate it using
    #  the actual yield object. """
    # assert np.allclose(single_mass_solar_II.x_on_h_average("Na"),
    #                    (0.3078305, 0))

def test_na_on_h_calculation_not_simple_total(two_star_different_z):
    """This calculation was done by hand. """
    assert np.isclose(two_star_different_z.x_on_h_total("Na"),
                      0.0998201197, atol=0)

# -----------------------------------------------------------

#  Test [Fe/Fe]. Should give zero.

# -----------------------------------------------------------

def test_fe_on_fe_calculation_single_zero_total(single_mass_zero):
    """For z zero metallicity object we should get some kind of error.
    We are taking a log of 0/0, which apparently give a nan. """
    assert np.isnan(single_mass_zero.x_on_fe_total("Fe"))

# def test_fe_on_fe_calculation_single_zero_average(single_mass_zero):
#     """For z zero metallicity object we should get some kind of error.
#     We are taking a log of 0/0, which apparently give a nan. """
#     assert np.isnan(single_mass_zero.x_on_fe_average("Fe")[0])

def test_fe_on_fe_calculation_single_one_Ia_total(single_mass_one_Ia):
    """For an object of metallicity one, we will get zero"""
    assert np.isclose(single_mass_one_Ia.x_on_fe_total("Fe"), 0)

# def test_fe_on_fe_calculation_single_one_Ia_average(single_mass_one_Ia):
#     """For an object of metallicity one, we will get zero"""
#     assert np.allclose(single_mass_one_Ia.x_on_fe_average("Fe"), (0, 0))

def test_fe_on_fe_calculation_single_one_II_total(single_mass_one_II):
    """For an object of metallicity one, we will get zero """
    assert np.isclose(single_mass_one_II.x_on_fe_total("Fe"), 0)

# def test_fe_on_fe_calculation_single_one_II_average(single_mass_one_II):
#     """For an object of metallicity one, we will get zero """
#     assert np.allclose(single_mass_one_II.x_on_fe_average("Fe"), (0, 0))

def test_fe_on_fe_calculation_single_solar_Ia_total(single_mass_solar_Ia):
    """For a solar metallicity object we get zero.  """
    assert np.isclose(single_mass_solar_Ia.x_on_fe_total("Fe"), 0)

# def test_fe_on_fe_calculation_single_solar_Ia_average(single_mass_solar_Ia):
#     """For a solar metallicity object we get zero.  """
#     assert np.allclose(single_mass_solar_Ia.x_on_fe_average("Fe"), (0, 0))

def test_fe_on_fe_calculation_single_solar_II_total(single_mass_solar_II):
    """For a solar metallicity object we get zero.  """
    assert np.isclose(single_mass_solar_II.x_on_fe_total("Fe"), 0)

# def test_fe_on_fe_calculation_single_solar_II_average(single_mass_solar_II):
#     """For a solar metallicity object we get zero.  """
#     assert np.allclose(single_mass_solar_II.x_on_fe_average("Fe"), (0, 0))

def test_fe_on_fe_calculation_not_simple_total(two_star_different_z):
    """We always get zero """
    assert np.isclose(two_star_different_z.x_on_fe_total("Fe"), 0)

# def test_fe_on_fe_calculation_not_simple_average(two_star_different_z):
#     """We always get zero """
#     assert np.allclose(two_star_different_z.x_on_fe_average("Fe"), (0, 0))


# -----------------------------------------------------------

#  Test [O/Fe]. Should not give zero. I'll calculate the values. I don't need
# to check the metallicity of one points anymore, since there isn't a 1-Z
# anywhere in this code. The solar metallicity is fine.

# -----------------------------------------------------------

def test_o_on_fe_calculation_single_zero_total(single_mass_zero):
    """For z zero metallicity object we should get some kind of infinity.
    We are taking a log of 0/0, so who knows what that will give. """
    assert np.isnan(single_mass_zero.x_on_fe_total("O"))

# def test_o_on_fe_calculation_single_zero_average(single_mass_zero):
#     """For z zero metallicity object we should get some kind of infinity.
#     We are taking a log of 0/0, so who knows what that will give. """
#     assert np.isnan(single_mass_zero.x_on_fe_average("O")[0])

def test_o_on_fe_calculation_single_solar_Ia_total(single_mass_solar_Ia):
    """Calculated by hand.  """
    assert np.isclose(single_mass_solar_Ia.x_on_fe_total("O"),
                      -1.507779731, atol=0)

# def test_o_on_fe_calculation_single_solar_Ia_average(single_mass_solar_Ia):
#     """Calculated by hand.  """
#     assert np.allclose(single_mass_solar_Ia.x_on_fe_average("O"),
#                        (-1.507779731, 0))

def test_o_on_fe_calculation_single_solar_II_total(single_mass_solar_II):
    """Calculated by hand.  """
    assert np.isclose(single_mass_solar_II.x_on_fe_total("O"),
                      0.3533312216, atol=0)

# def test_o_on_fe_calculation_single_solar_II_average(single_mass_solar_II):
#     """Calculated by hand.  """
#     assert np.allclose(single_mass_solar_II.x_on_fe_average("O"),
#                        (0.3533312216, 0))

def test_o_on_fe_calculation_not_simple_total(two_star_different_z):
    """Calculated by hand.  """
    assert np.isclose(two_star_different_z.x_on_fe_total("O"),
                      -0.8538614906, atol=0)

# -----------------------------------------------------------

#  Test simple metallicity values

# -----------------------------------------------------------

def test_log_z_z_sun_single_zero_total(single_mass_zero):
    """For a zero metallicity object we should get a negative infinity."""
    assert np.isneginf(single_mass_zero.log_z_over_z_sun_total())

# def test_log_z_z_sun_single_zero_average(single_mass_zero):
#     """For a zero metallicity object we should get a negative infinity."""
#     assert np.isneginf(single_mass_zero.log_z_over_z_sun_average()[0])

def test_log_z_z_sun_single_solar_Ia_total(single_mass_solar_Ia):
    """For solar we should get zero."""
    assert np.isclose(single_mass_solar_Ia.log_z_over_z_sun_total(), 0)

# def test_log_z_z_sun_single_solar_Ia_average(single_mass_solar_Ia):
#     """For solar we should get zero."""
#     assert np.allclose(single_mass_solar_Ia.log_z_over_z_sun_average(),
#                        (0, 0))

def test_log_z_z_sun_single_solar_II_total(single_mass_solar_II):
    """For solar we should get zero."""
    assert np.isclose(single_mass_solar_II.log_z_over_z_sun_total(), 0)

# def test_log_z_z_sun_single_solar_II_average(single_mass_solar_II):
#     """For solar we should get zero."""
#     assert np.allclose(single_mass_solar_II.log_z_over_z_sun_average(),
#                        (0, 0))

def test_log_z_z_sun_single_one_Ia_total(single_mass_one_Ia):
    """For solar we should get the log of the ratio of 1 and z_sun.
    This was calculated by hand."""
    assert np.isclose(single_mass_one_Ia.log_z_over_z_sun_total(), 1.789274018)

# def test_log_z_z_sun_single_one_Ia_average(single_mass_one_Ia):
#     """For solar we should get the log of the ratio of 1 and z_sun.
#     This was calculated by hand."""
#     assert np.allclose(single_mass_one_Ia.log_z_over_z_sun_average(),
#                        (1.789274018, 0))

def test_log_z_z_sun_single_one_II_total(single_mass_one_II):
    """For solar we should get the log of the ratio of 1 and z_sun.
    This was calculated by hand."""
    assert np.isclose(single_mass_one_II.log_z_over_z_sun_total(), 1.789274018)

# def test_log_z_z_sun_single_one_II_average(single_mass_one_II):
#     """For solar we should get the log of the ratio of 1 and z_sun.
#     This was calculated by hand."""
#     assert np.allclose(single_mass_one_II.log_z_over_z_sun_average(),
#                        (1.789274018, 0))

def test_log_z_not_simple_total(two_star_different_z):
    """Calculated by hand. """
    assert np.isclose(two_star_different_z.log_z_over_z_sun_total(),
                      0.2206377821)

