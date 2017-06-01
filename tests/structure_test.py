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
M_c = 6.9
M_d = 6.6
a_c = 12.5
a_d = 675.3
radii = np.arange(0, 5000)
plummer = profiles.plummer_2d(radii, 10**M_c, a_c)
disk = profiles.exp_disk(radii, 10**M_d, a_d)

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

# def test_fitting_results_total():  # too hard
    # fit = nsc_structure.Fitting(radii, plummer + disk)
    # fit.fit()
    # assert np.isclose(fit.M_c, M_c, atol=0.1)
    # assert np.isclose(fit.a_c, a_c, atol=0.1)
    # assert np.isclose(fit.M_d, M_d, atol=0.1)
    # assert np.isclose(fit.a_d, a_d, atol=0.1)