import pytest
import yt
import numpy as np

from yt_tools import minigal
from yt_tools import galaxy

# some setup needed
file_loc = "../../../google_drive/research/simulation_outputs/" \
           "fiducial_destroy/continuous_a0.2406.art"
ds = yt.load(file_loc)

file = open("./real_gal_save.txt", "r")
read_in_gal =  galaxy.read_gal(ds, file)
file.close()
out_file = open("./minigal_save.txt", "w")
read_in_gal.mini_write(out_file)
out_file.close()

@pytest.fixture
def new_gal():
    out_file = open("./minigal_save.txt", "r")
    return minigal.Minigal(out_file)

def test_mass(new_gal):
    mass, mass_err = read_in_gal.nsc_mass_and_errs()
    # these are in stellar masses, convert the errors
    mass_err = [err.to("msun").value for err in mass_err]
    assert np.isclose(mass.to("msun").value, new_gal.nsc_mass)
    assert np.allclose(mass_err, new_gal.nsc_mass_err)

def test_gal_mass(new_gal):
    gal_mass = read_in_gal.stellar_mass(radius_cut=None)
    assert np.isclose(gal_mass.to("msun").value, new_gal.gal_mass)

def test_nsc_radius(new_gal):
    nsc_radius = read_in_gal.nsc.r_half_non_parametric
    assert np.isclose(nsc_radius, new_gal.nsc_r_half)

def test_nsc_radius_errs(new_gal):
    nsc_radius_err = read_in_gal.nsc.r_half_non_parametric_err
    assert np.allclose(nsc_radius_err, new_gal.nsc_r_half_err)

def test_axis_ratios_b(new_gal):
    b_over_a = read_in_gal.nsc_axis_ratios.b_over_a
    assert np.isclose(b_over_a, new_gal.b_over_a)

def test_axis_ratios_c(new_gal):
    c_over_a = read_in_gal.nsc_axis_ratios.c_over_a
    assert np.isclose(c_over_a, new_gal.c_over_a)

def test_axis_ratios_ellipticity(new_gal):
    ellipticity = read_in_gal.nsc_axis_ratios.ellipticity
    assert np.isclose(ellipticity, new_gal.ellipticity)

def test_rotation(new_gal):
    rot_vel = read_in_gal.mean_rot_vel.to("km/s").value
    assert np.isclose(rot_vel, new_gal.nsc_rot_vel)

def test_dispersion(new_gal):
    sigma = read_in_gal.nsc_3d_sigma.to("km/s").value
    assert np.isclose(sigma, new_gal.nsc_3d_sigma)

def test_metallicity(new_gal):
    z = read_in_gal.nsc_abundances.log_z_over_z_sun_total()
    assert np.isclose(z, new_gal.metallicity)

def test_metallicity_spread(new_gal):
    z_sigma = np.sqrt(read_in_gal.nsc_abundances.log_z_over_z_sun_average()[1])
    assert np.isclose(z_sigma, new_gal.metallicity_spread)

def test_fe_on_h(new_gal):
    fe_on_h = read_in_gal.nsc_abundances.x_on_h_total("Fe")
    assert np.isclose(fe_on_h, new_gal.fe_on_h)

def test_fe_on_h_spread(new_gal):
    fe_on_h_spread = np.sqrt(read_in_gal.nsc_abundances.x_on_h_average("Fe")[1])
    assert np.isclose(fe_on_h_spread, new_gal.fe_on_h_spread)

def test_gal_fe_on_h(new_gal):
    gal_fe_on_h = read_in_gal.gal_abundances.x_on_h_total("Fe")
    assert np.isclose(gal_fe_on_h, new_gal.gal_fe_on_h)

def test_gal_fe_on_h_spread(new_gal):
    gal_fe_on_h_s = np.sqrt(read_in_gal.gal_abundances.x_on_h_average("Fe")[1])
    assert np.isclose(gal_fe_on_h_s, new_gal.gal_fe_on_h_spread)

def test_o_on_fe(new_gal):
    o_on_fe = read_in_gal.nsc_abundances.x_on_fe_total("O")
    assert np.isclose(o_on_fe, new_gal.o_on_fe)

def test_o_on_fe_spread(new_gal):
    o_on_fe_spread = np.sqrt(read_in_gal.nsc_abundances.x_on_fe_average("O")[1])
    assert np.isclose(o_on_fe_spread, new_gal.o_on_fe_spread)

def test_mg_on_fe(new_gal):
    mg_on_fe = read_in_gal.nsc_abundances.x_on_fe_total("Mg")
    assert np.isclose(mg_on_fe, new_gal.mg_on_fe)

def test_mg_on_fe_spread(new_gal):
    mg_on_fe_s= np.sqrt(read_in_gal.nsc_abundances.x_on_fe_average("Mg")[1])
    assert np.isclose(mg_on_fe_s, new_gal.mg_on_fe_spread)

def test_al_on_fe(new_gal):
    al_on_fe = read_in_gal.nsc_abundances.x_on_fe_total("Al")
    assert np.isclose(al_on_fe, new_gal.al_on_fe)

def test_al_on_fe_spread(new_gal):
    al_on_fe_s = np.sqrt(read_in_gal.nsc_abundances.x_on_fe_average("Al")[1])
    assert np.isclose(al_on_fe_s, new_gal.al_on_fe_spread)



