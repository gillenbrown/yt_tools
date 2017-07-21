from yt_tools import nsc_structure
from yt_tools import profiles

import pytest
import numpy as np
from scipy import interpolate, integrate

def test_fitting_init_iterable_checking():
    with pytest.raises(TypeError):
        nsc_structure.Fitting(0, 0)
    with pytest.raises(TypeError):
        nsc_structure.Fitting([0], 0)
    with pytest.raises(TypeError):
        nsc_structure.Fitting(0, [0])
    nsc_structure.Fitting([0], [0])  # no error

def test_fitting_init_same_size():
    with pytest.raises(ValueError):
        nsc_structure.Fitting([0], [0, 1])  # not same size
    with pytest.raises(ValueError):
        nsc_structure.Fitting([0, 1, 2], [0, 1])  # not same size
    nsc_structure.Fitting([1, 2, 3], [1, 2, 3])  # no error

def test_fitting_log_safety():
    # test the removal of log(0) values
    fit = nsc_structure.Fitting([1, 2, 3], [0, 1, 2])
    assert np.allclose(fit.radii_logsafe, [2, 3])
    assert np.allclose(fit.log_densities, [0, 0.30103])

    fit = nsc_structure.Fitting([1, 2, 3], [1, 2, 3])
    assert np.allclose(fit.radii_logsafe, [1, 2, 3])
    assert np.allclose(fit.log_densities, [0, 0.30103, 0.4771212547])
# -----------------------------------------------------------------------------

# test the fitting process against some simple cases.

# -----------------------------------------------------------------------------
M_c = 10**7.3
M_d = 10**6.2
a_c = 12.5
a_d = 675.3
radii = np.arange(0, 5000)
plummer = profiles.plummer_2d(radii, M_c, a_c)
disk = profiles.exp_disk(radii, M_d, a_d)

def test_fitting_results_plummer():
    fit = nsc_structure.Fitting(radii, plummer)
    fit.fit()
    assert np.isclose(fit.M_c, M_c)
    assert np.isclose(fit.a_c, a_c)
    assert np.isclose(fit.M_d, 0, atol=2)

def test_fitting_results_disk():
    fit = nsc_structure.Fitting(radii, disk)
    fit.fit()
    assert np.isclose(fit.M_d, M_d)
    assert np.isclose(fit.a_d, a_d)
    assert np.isclose(fit.M_c, 0, atol=2)

def test_fitting_results_total():
    fit = nsc_structure.Fitting(radii, plummer + disk)
    fit.fit()
    assert np.isclose(fit.M_c, M_c, atol=0.01)
    assert np.isclose(fit.a_c, a_c, atol=0.01)
    assert np.isclose(fit.M_d, M_d, atol=0.01)
    assert np.isclose(fit.a_d, a_d, atol=0.01)

# -----------------------------------------------------------------------------

# test the transformation of errors. This is used in the log mass calculations

# -----------------------------------------------------------------------------

def test_error_transform_error_checking():
    with pytest.raises(ValueError):
        nsc_structure.error_transform(0, -1)

def test_error_transform_results():
    assert np.allclose(nsc_structure.error_transform(1, 0.1),
                       (2.056717653, 2.589254118))
    assert np.allclose(nsc_structure.error_transform(5.5, 0.5),
                       (216227.766, 683772.234))
    assert np.allclose(nsc_structure.error_transform(4.0, 1.0),
                       (9e3, 9e4))
    assert np.allclose(nsc_structure.error_transform(-3.0, 1.0),
                       (9e-4, 9e-3))
# -----------------------------------------------------------------------------

# test the creation of the structure class.

# -----------------------------------------------------------------------------

@pytest.fixture
def struct_plummer():
    return nsc_structure.NscStructure(radii, plummer)

@pytest.fixture
def struct_disk():
    return nsc_structure.NscStructure(radii, disk)

@pytest.fixture
def struct_total():
    return nsc_structure.NscStructure(radii, plummer + disk)

@pytest.fixture
def struct_total_error():
    return nsc_structure.NscStructure(radii, (plummer + disk) * np.random.normal(1, 0.1, len(radii)))


def test_init_nsc_plummer(struct_plummer):
    assert np.isclose(struct_plummer.M_c_parametric, M_c)
    assert np.isclose(struct_plummer.a_c_parametric, a_c)
    assert np.isclose(struct_plummer.M_d_parametric, 0, atol=2)

def test_init_nsc_disk(struct_disk):
    assert np.isclose(struct_disk.M_d_parametric, M_d)
    assert np.isclose(struct_disk.a_d_parametric, a_d)
    assert np.isclose(struct_disk.M_c_parametric, 0, atol=2)

def test_equality_radius_results(struct_total):
    eq_rad = struct_total.nsc_radius
    plummer_density = profiles.plummer_2d(eq_rad, struct_total.M_c_parametric,
                                          struct_total.a_c_parametric)
    disk_density = profiles.exp_disk(eq_rad, struct_total.M_d_parametric,
                                     struct_total.a_d_parametric)
    assert np.isclose(plummer_density, disk_density, atol=0.001)

def test_equality_radius_plummer(struct_plummer):
    assert struct_plummer.nsc_radius is None
    assert struct_plummer.nsc_radius_err is None

def test_equality_radius_disk(struct_disk):
    assert struct_disk.nsc_radius is None
    assert struct_disk.nsc_radius_err is None

def test_equality_radius_total(struct_total):
    assert struct_total.nsc_radius is not None
    assert struct_total.nsc_radius_err is not None
    assert struct_total.nsc_radius > 0

def test_non_parametric_mass_plummer(struct_plummer):
    assert struct_plummer.M_c_non_parametric is None
    assert struct_plummer.M_c_non_parametric_err is None

def test_non_parametric_mass_disk(struct_disk):
    assert struct_disk.M_c_non_parametric is None
    assert struct_disk.M_c_non_parametric_err is None

def test_non_parametric_mass_total(struct_total):
    assert struct_total.M_c_non_parametric > 0
    assert np.allclose(struct_total.M_c_non_parametric_err, (0, 0))
    assert struct_total.M_c_non_parametric is not None
    assert struct_total.M_c_non_parametric_err is not None

def test_non_parametric_half_mass_plummer(struct_plummer):
    assert struct_plummer.r_half_non_parametric is None
    assert struct_plummer.r_half_non_parametric_err is None

def test_non_parametric_half_mass_disk(struct_disk):
    assert struct_disk.r_half_non_parametric is None
    assert struct_disk.r_half_non_parametric_err is None

def test_non_parametric_half_mass_total(struct_total):
    assert struct_total.r_half_non_parametric is not None
    assert struct_total.r_half_non_parametric_err is not None
    assert struct_total.r_half_non_parametric > 0
    assert np.allclose(struct_total.r_half_non_parametric_err, (0, 0))

def test_errors(struct_total_error):
    assert struct_total_error.nsc_radius is not None
    assert struct_total_error.nsc_radius > 0
    assert struct_total_error.nsc_radius_err is not None
    assert not np.isclose(struct_total_error.nsc_radius_err[0], 0)
    assert not np.isclose(struct_total_error.nsc_radius_err[1], 0)
    assert np.all(struct_total_error.nsc_radius_err > 0)

    assert struct_total_error.M_c_non_parametric is not None
    assert struct_total_error.M_c_non_parametric > 0
    assert struct_total_error.M_c_non_parametric_err is not None
    assert not np.isclose(struct_total_error.M_c_non_parametric_err[0], 0)
    assert not np.isclose(struct_total_error.M_c_non_parametric_err[1], 0)
    assert np.all(struct_total_error.M_c_non_parametric_err > 0)

    assert struct_total_error.r_half_non_parametric is not None
    assert struct_total_error.r_half_non_parametric > 0
    assert struct_total_error.r_half_non_parametric_err is not None
    # assert not np.isclose(struct_total_error.r_half_non_parametric_err[0], 0)
    # assert not np.isclose(struct_total_error.r_half_non_parametric_err[1], 0)
    assert np.all(struct_total_error.r_half_non_parametric_err >= 0)

def test_half_mass_calculation(struct_total_error):
    cluster_mass = struct_total_error.M_c_non_parametric
    half_mass_radius = struct_total_error._half_mass(cluster_mass)

    # then see if that actually works
    integrand_values = []
    radii = []
    for radius in np.arange(0, half_mass_radius, 0.01):
        integrand_values.append(struct_total_error.dens_interp(radius) *
                                2 * np.pi * radius)
        radii.append(radius)

    measured_half_mass = integrate.simps(integrand_values, radii)
    assert np.isclose(measured_half_mass, cluster_mass / 2.0, rtol=0.002)

# -----------------------------------------------------------------------------

# test the axis ratios stuff

# -----------------------------------------------------------------------------
# setup some simple values to use
values = np.linspace(-10, 10, 100)
zeroes = np.zeros(100)
xs = np.concatenate([values, zeroes, zeroes])
ys = np.concatenate([zeroes, values, zeroes])
zs = np.concatenate([zeroes, zeroes, values])
masses = np.ones(300) * 5.34  # mass doesn't matter if all the same

def test_axis_ratios_error_checking():
    nsc_structure.AxisRatios(xs, ys, zs, masses) # no error
    with pytest.raises(ValueError):
        nsc_structure.AxisRatios(xs, ys, zs, [1, 2, 3])
    with pytest.raises(ValueError):
        nsc_structure.AxisRatios([1, 2, 2], ys, zs, masses)

def test_axis_ratios_symmetric():
    """Test with a symmetric structure. Should give 1 values for all ratios."""
    a_r = nsc_structure.AxisRatios(xs, ys, zs, masses)
    assert a_r.a_over_b == 1
    assert a_r.b_over_a == 1
    assert a_r.a_over_c == 1
    assert a_r.c_over_a == 1
    assert a_r.b_over_c == 1
    assert a_r.c_over_b == 1
    assert a_r.ellipticity == 0

def test_axis_ratios_one_not_symmetric():
    """Test with a structure wher one axis is larger than the others. """
    a_r = nsc_structure.AxisRatios(xs * 4, ys, zs, masses)
    assert np.isclose(a_r.a_over_b, 4.0)
    assert np.isclose(a_r.b_over_a, 0.25)
    assert np.isclose(a_r.a_over_c, 4.0)
    assert np.isclose(a_r.c_over_a, 0.25)
    assert np.isclose(a_r.b_over_c, 1.0)
    assert np.isclose(a_r.c_over_b, 1.0)
    assert np.isclose(a_r.ellipticity, 0.75)

def test_axis_ratios_all_not_symmetric():
    """Test with a structure where all three axes are different."""
    a_r = nsc_structure.AxisRatios(xs, ys * 2, zs * 3, masses)
    assert np.isclose(a_r.a_over_b, 1.5)
    assert np.isclose(a_r.b_over_a, 1.0 / 1.5)
    assert np.isclose(a_r.a_over_c, 3.0)
    assert np.isclose(a_r.c_over_a, 1.0 / 3.0)
    assert np.isclose(a_r.b_over_c, 2.0)
    assert np.isclose(a_r.c_over_b, 0.5)
    assert np.isclose(a_r.ellipticity, 2.0 / 3.0)

def test_axis_ratios_all_not_symmetric_rotated():
    """Test with a rotated structure where are three axes are different. """
    new_x = xs
    new_y = ys * 2
    new_z = zs * 3

    # then rotate through an angle
    theta = 23.4
    new_new_xs = new_x * np.cos(theta) - new_y * np.sin(theta)
    new_new_ys = new_x * np.sin(theta) + new_y * np.cos(theta)

    a_r = nsc_structure.AxisRatios(new_new_xs, new_new_ys, new_z, masses)
    assert np.isclose(a_r.a_over_b, 1.5)
    assert np.isclose(a_r.b_over_a, 1.0 / 1.5)
    assert np.isclose(a_r.a_over_c, 3.0)
    assert np.isclose(a_r.c_over_a, 1.0 / 3.0)
    assert np.isclose(a_r.b_over_c, 2.0)
    assert np.isclose(a_r.c_over_b, 0.5)

def test_axis_ratios_different_mass():
    """Test where all locations are the same, but different masses makes the
    axis ratios different. This is just based on the components being different.
    """
    ones = np.ones(100)
    new_masses = np.concatenate([ones, ones*2, ones*3])
    a_r = nsc_structure.AxisRatios(xs, ys, zs, new_masses)
    assert np.isclose(a_r.a_over_b, np.sqrt(1.5))
    assert np.isclose(a_r.b_over_a, np.sqrt(1.0 / 1.5))
    assert np.isclose(a_r.a_over_c, np.sqrt(3.0))
    assert np.isclose(a_r.c_over_a, np.sqrt(1.0 / 3.0))
    assert np.isclose(a_r.b_over_c, np.sqrt(2.0))
    assert np.isclose(a_r.c_over_b, np.sqrt(0.5))

