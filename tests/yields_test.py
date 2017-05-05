from yt_tools import yields
import pytest
import numpy as np

# create simple object for testing
@pytest.fixture
def yields_test_case():
    return yields.Yields("test")

def test_metals_sum(yields_test_case):
    assert yields_test_case._metals_sum() == 55 - 3  # sum(3..10)
    # this is because the test has elements 1 to 10, with an abundance of 1 to
    # 10, and we exclude H and He

def test_normalize_metals(yields_test_case):
    """Test whether or not the normalization is working properly"""
    yields_test_case.normalize_metals(1)
    assert yields_test_case.H_1 == 1.0 / 52.0  # 1 / sum(3..10)
    assert yields_test_case.F_9 == 9.0 / 52.0
    assert yields_test_case.Na_10 == 10.0 / 52.0
    # the total amount must be larger than the amount of metals we normalized 
    # to, since H and He are included. 
    assert sum(yields_test_case._abundances.values()) > 1.0

    # normalize to a value other than 1
    yields_test_case.normalize_metals(25.0)
    assert yields_test_case.H_1 == 25.0 / 52.0  # 1 / sum(1..10)
    assert yields_test_case.F_9 == 9.0 * 25.0 / 52.0
    assert np.isclose(yields_test_case.Na_10, 10.0 * 25.0 / 52.0)
    # that one required isclose for whatever reason. 
    assert sum(yields_test_case._abundances.values()) > 25.0

def test_set_metallicity_error_checking(yields_test_case):
    """Metallicities are only vaild between zero and one."""
    with pytest.raises(ValueError):
        yields_test_case.set_metallicity(2)

def test_interpolate_z_test(yields_test_case):
    """Test whether the interpolation is working correctly in the test case
    
    In the test case, each isotope goes between the atomic number and the 
    atomic number plus one at metallicities of 0 and 1. 
    So at Z=0, H=1, and at Z=1, H=2"""
    # at zero metallicity, each should be the atomic number
    yields_test_case.set_metallicity(0)
    assert yields_test_case.H_1 == 1.0
    assert yields_test_case.F_9 == 9.0
    assert yields_test_case.Na_10 == 10.0

    # then at all metallicity, each should be the atomic number plus one
    yields_test_case.set_metallicity(1.0)
    assert yields_test_case.H_1 == 2.0
    assert yields_test_case.F_9 == 10.0
    assert yields_test_case.Na_10 == 11.0

    # then test a point halfway in between in log space. This is a little weird,
    # since 0 is at -4 in log space according to my definition. So halfway
    # between -4 and 0=log(1) in log space is log(x) = -2. The resulting
    # value should be atomic number + 0.5
    yields_test_case.set_metallicity(10**-2)
    assert yields_test_case.H_1 == 1.5
    assert yields_test_case.F_9 == 9.5
    assert yields_test_case.Na_10 == 10.5


def test_get_iwamoto_path():
    """Tests the function that gets the path of the Iwamoto yields"""
    file_loc =  yields._get_iwamoto_path()
    # I know the first line, so I can read that and see what it is
    iwamoto_file = open(file_loc, "r")
    assert iwamoto_file.readline() == "# Table 3 from Iwamoto et al 1999\n"

def test_iwamoto_element_parsing():
    """Tests turning the format of the Iwamoto output into the format this
    class needs"""
    assert yields._parse_iwamoto_element("^{8}O") == "O_8"
    assert yields._parse_iwamoto_element("^{12}C") == "C_12"
    assert yields._parse_iwamoto_element("^{55}Mn") == "Mn_55"
    assert yields._parse_iwamoto_element("^{68}Zn") == "Zn_68"

def test_iwamoto_model_parsing():
    """Tests getting the model itself out of the iwamoto name"""
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_W7") == "W7"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_W70") == "W70"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_WDD1") == "WDD1"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_WDD2") == "WDD2"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_WDD3") == "WDD3"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_CDD1") == "CDD1"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_CDD2") == "CDD2"
    with pytest.raises(ValueError):
        yields._parse_iwamoto_model("iwamsdfs")
    with pytest.raises(ValueError):
        yields._parse_iwamoto_model("iwamoto_99_Ia_wer")  #not a valid model

def test_make_iwamoto_w7():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_W7")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 4.83E-02
        assert iwamoto_test.Cl_35 == 1.37E-04
        assert iwamoto_test.Zn_68 == 1.74E-08
        assert iwamoto_test.Zn_64 == 1.06E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_w70():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_W70")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 5.08E-02
        assert iwamoto_test.Cl_35 == 1.06E-05
        assert iwamoto_test.Zn_68 == 1.13E-08
        assert iwamoto_test.Zn_64 == 7.01E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_wdd1():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_WDD1")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 5.42E-03
        assert iwamoto_test.Cl_35 == 9.28E-05
        assert iwamoto_test.Zn_68 == 7.44E-08
        assert iwamoto_test.Zn_64 == 3.71E-06
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_wdd2():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_WDD2")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 8.99E-03
        assert iwamoto_test.Cl_35 == 7.07E-05
        assert iwamoto_test.Zn_68 == 8.81E-08
        assert iwamoto_test.Zn_64 == 3.10E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_wdd3():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_WDD3")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 1.66E-02
        assert iwamoto_test.Cl_35 == 5.33E-05
        assert iwamoto_test.Zn_68 == 9.42E-08
        assert iwamoto_test.Zn_64 == 5.76E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_cdd1():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_CDD1")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 9.93E-03
        assert iwamoto_test.Cl_35 == 9.03E-05
        assert iwamoto_test.Zn_68 == 3.08E-09
        assert iwamoto_test.Zn_64 == 1.87E-06
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_cdd2():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_CDD2")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 5.08E-03
        assert iwamoto_test.Cl_35 == 6.56E-05
        assert iwamoto_test.Zn_68 == 3.03E-08
        assert iwamoto_test.Zn_64 == 3.96E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_met_log():
    """Tests the metallicity log function. Is just like log, but returns a
    fixed value for 0."""
    assert yields._metallicity_log(0) == -4
    assert yields._metallicity_log(1) == 0
    assert yields._metallicity_log(0.01) == -2
    assert yields._metallicity_log(100) == 2

## TODO: test the way I only normalize sometimes after setting the metallicity.
#        make sure that works the way I think it should.