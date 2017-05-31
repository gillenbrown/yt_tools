from yt_tools import utils
from yt.units import pc

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