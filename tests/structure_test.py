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

