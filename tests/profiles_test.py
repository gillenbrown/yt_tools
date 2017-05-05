import yt_tools

import numpy as np
from scipy.integrate import quad
import pytest

# I want to test the profiles for random values of the quantities, so 
# create random values for the mass and scale length of the profiles
num_tests = 1
M_a_combos = zip(10**(np.random.uniform(5, 10, num_tests)),
                 np.random.uniform(1, 100, num_tests))

# we will be checking the results of various integrals. Since there is some
# numerical error on these, we have to have some tolerance here. Scipy reports
# errors, but we want to be generous on the size of our tolerance. This factor
# will be multiplied by the given error, and the difference must be within
# this range to be correct.
error_tolerance = 10  

# I am going to be integrating in spherical and cylindrical space, so I will 
# create these integrands beforehand to make life easier.
def spherical_integrand(r, func, M, a):
    return func(r, M, a) * 4 * np.pi * r**2

def cylindrical_integrad(r, func, M, a):
    return func(r, M, a) * 2 * np.pi * r

# We want to check that the profiles integrate out to be the mass. We have to do
# the 2d and 3d differently, since the integration is different.
profiles = [yt_tools.hernquist_2d, yt_tools.plummer_2d,
            yt_tools.hernquist_3d, yt_tools.plummer_3d,
            yt_tools.exp_disk]
integrands = [cylindrical_integrad, cylindrical_integrad, 
              spherical_integrand, spherical_integrand,
              cylindrical_integrad]
# we then zip these together so we can pass them in to the pytest parametrize
parametrize_input_normalization = zip(profiles, integrands)

# then we can actually run the test
@pytest.mark.parametrize("M,a", M_a_combos)
@pytest.mark.parametrize("profile,integrand", parametrize_input_normalization)
def test_profile_normalization(M, a, profile, integrand):
    integral, error = quad(integrand, 0, np.infty, args=(profile, M, a))
    assert(np.isclose(M, integral, atol=error*error_tolerance, rtol=0))

# Next we want to test that the half mass functions work properly
# I also have to get everything together in order to do the parametrize 
# thing properly for each profile
half_mass = [yt_tools.hernquist_2d_half_mass, yt_tools.plummer_2d_half_mass,
             yt_tools.hernquist_3d_half_mass, yt_tools.plummer_3d_half_mass,
             yt_tools.exp_disk_half_mass]
# we have to do the hernquist 2d profile differently, since it isn't an analytic
# half-mass form, but is only numerical. I have to do the error checking 
# separately for that. Normally we want the relative tolerance to be zero, but
# the absolute tolerance to be error_tolerance*scipy's error. For the 
# Hernquist 2d it's different.
r_tol = [1e-4, 0, 0, 0, 1e-5]
error_tol= [0, error_tolerance, error_tolerance, error_tolerance, 0]
# then we can combine everything together
parametrize_input_half_mass = zip(profiles, integrands, half_mass, 
                                  r_tol, error_tol) 

@pytest.mark.parametrize("M,a", M_a_combos)
@pytest.mark.parametrize("profile,integrand,half_mass,r_tol,error_tol", 
                         parametrize_input_half_mass)
def test_half_mass(M, a, profile, integrand, half_mass, r_tol, error_tol):
    # we calculate the half mass radius, then see if integrating to that point
    # gives us half the mass
    half_mass_radius = half_mass(a)
    integral, error = quad(integrand, 0, half_mass_radius, args=(profile, M, a))
    assert(np.isclose(M/2.0, integral, atol=error*error_tol, rtol=r_tol))




