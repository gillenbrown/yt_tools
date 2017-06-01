from yt_tools import nsc_structure
from yt_tools import profiles

import pytest
import numpy as np

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
    assert np.isclose(fit.M_d, 0)

def test_fitting_results_disk():
    fit = nsc_structure.Fitting(radii, disk)
    fit.fit()
    assert np.isclose(fit.M_d, M_d)
    assert np.isclose(fit.a_d, a_d)
    assert np.isclose(fit.M_c, 0)

def test_fitting_results_total():
    fit = nsc_structure.Fitting(radii, plummer + disk)
    fit.fit()
    assert np.isclose(fit.M_c, M_c, atol=0.01)
    assert np.isclose(fit.a_c, a_c, atol=0.01)
    assert np.isclose(fit.M_d, M_d, atol=0.01)
    assert np.isclose(fit.a_d, a_d, atol=0.01)

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


def test_init_nsc_plummer(struct_plummer):
    assert np.isclose(struct_plummer.M_c_parametric, M_c)
    assert np.isclose(struct_plummer.a_c_parametric, a_c)
    assert np.isclose(struct_plummer.M_d_parametric, 0)

def test_init_nsc_disk(struct_disk):
    assert np.isclose(struct_disk.M_d_parametric, M_d)
    assert np.isclose(struct_disk.a_d_parametric, a_d)
    assert np.isclose(struct_disk.M_c_parametric, 0)

def test_equality_radius_results(struct_total):
    eq_rad = struct_total.nsc_radius
    plummer_density = profiles.plummer_2d(eq_rad, struct_total.M_c_parametric,
                                          struct_total.a_c_parametric)
    disk_density = profiles.exp_disk(eq_rad, struct_total.M_d_parametric,
                                     struct_total.a_d_parametric)
    assert np.isclose(plummer_density, disk_density, atol=0.001)

def test_equality_radius_plummer(struct_plummer):
    assert struct_plummer.nsc_radius is None

def test_equality_radius_disk(struct_disk):
    assert struct_disk.nsc_radius is None

def test_non_parametric_mass_plummer(struct_plummer):
    assert struct_plummer.M_c_non_parametric is None

def test_non_parametric_mass_disk(struct_disk):
    assert struct_disk.M_c_non_parametric is None

def test_non_parametric_mass_total(struct_total):
    assert 0 < struct_total.M_c_non_parametric
    assert struct_total.M_c_non_parametric is not None

def test_non_parametric_half_mass_plummer(struct_plummer):
    assert struct_plummer.r_half_non_parametric is None

def test_non_parametric_half_mass_disk(struct_disk):
    assert struct_disk.r_half_non_parametric is None

def test_non_parametric_half_mass_total(struct_total):
    assert struct_total.r_half_non_parametric is not None
    assert struct_total.r_half_non_parametric > 0
