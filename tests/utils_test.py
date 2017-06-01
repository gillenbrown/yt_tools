from yt_tools import utils
from yt.units import pc
import numpy as np

import pytest

def test_unit_checking():
    with pytest.raises(TypeError):
        utils.test_for_units([1], "")
    with pytest.raises(TypeError):
        utils.test_for_units(1, "")
    assert utils.test_for_units(1*pc, "") is None  # will pass, return nothing

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
