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

# Note that none of these tests actually check what happens when a galaxy does
# not have a NSC. I think my code works fine in those situations, but it is not
# directly tested here. I checked it by running it on the real sims, where
# some do not have a NSC. I then fixed the bugs that arose there. Checking that
# it works for galaxies with a NSC is the key point, though.

@pytest.fixture
def new_gal():
    out_file = open("./minigal_save.txt", "r")
    return minigal.Minigal(out_file)

def test_id(new_gal):
    assert read_in_gal.id == new_gal.id

def test_mass(new_gal):
    mass, mass_err = read_in_gal.nsc_mass_and_errs()
    # these are in stellar masses, convert the errors
    mass_err = [err.to("msun").value for err in mass_err]
    assert np.isclose(mass.to("msun").value, new_gal.nsc_mass)
    assert np.allclose(mass_err, new_gal.nsc_mass_err)

def test_nsc_radius(new_gal):
    nsc_radius = read_in_gal.nsc_radius.to("pc").value
    assert np.isclose(nsc_radius, new_gal.nsc_radius)

def test_gal_mass(new_gal):
    gal_mass = read_in_gal.stellar_mass(radius_cut=None)
    assert np.isclose(gal_mass.to("msun").value, new_gal.gal_mass)

def test_gal_half_mass_radius(new_gal):
    half_mass_radius = read_in_gal.galaxy_half_mass_radius()
    assert np.isclose(half_mass_radius.to("kpc").value, new_gal.gal_r_half)

def test_nsc_radius_half(new_gal):
    nsc_radius = read_in_gal.nsc.r_half_non_parametric
    assert np.isclose(nsc_radius, new_gal.nsc_r_half)

def test_nsc_radius_half_errs(new_gal):
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

def test_anisotropy(new_gal):
    beta = read_in_gal.anisotropy_parameter
    assert np.isclose(beta, new_gal.anisotropy_parameter)

def test_dispersion_along_a(new_gal):
    sigma = read_in_gal.nsc_disp_along_a.to("km/s").value
    assert np.isclose(sigma, new_gal.nsc_disp_along_a)

def test_dispersion_along_b(new_gal):
    sigma = read_in_gal.nsc_disp_along_b.to("km/s").value
    assert np.isclose(sigma, new_gal.nsc_disp_along_b)

def test_dispersion_along_c(new_gal):
    sigma = read_in_gal.nsc_disp_along_c.to("km/s").value
    assert np.isclose(sigma, new_gal.nsc_disp_along_c)

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

def test_kde_smoothed_radii_2D(new_gal):
    radii = read_in_gal.kde_radii_smoothed["mass_kde_2D"]
    assert np.allclose(radii, new_gal.kde_radii_2D_smoothed)

def test_kde_smoothed_densities_2D(new_gal):
    densities = read_in_gal.kde_densities_smoothed["mass_kde_2D"]
    assert np.allclose(densities, new_gal.kde_densities_2D_smoothed)

def test_binned_radii(new_gal):
    radii = read_in_gal.binned_radii
    assert np.allclose(radii, new_gal.binned_radii)

def test_binned_densities(new_gal):
    densities = read_in_gal.binned_densities
    assert np.allclose(densities, new_gal.binned_densities)

def test_integrated_kde_radii(new_gal):
    radii = read_in_gal.integrated_kde_radii
    assert np.allclose(radii, new_gal.integrated_kde_radii)

def test_integrated_kde_densities_densities(new_gal):
    densities = read_in_gal.integrated_kde_densities
    assert np.allclose(densities, new_gal.integrated_kde_densities)

def test_fit_components_disk_mass(new_gal):
    M_d = read_in_gal.nsc.M_d_parametric
    assert np.isclose(M_d, new_gal.component_fit_disk_mass)

def test_fit_components_disk_radius(new_gal):
    a_d = read_in_gal.nsc.a_d_parametric
    assert np.isclose(a_d, new_gal.component_fit_disk_scale_radius)

def test_fit_components_cluster_mass(new_gal):
    M_c = read_in_gal.nsc.M_c_parametric
    assert np.isclose(M_c, new_gal.component_fit_cluster_mass)

def test_fit_components_cluster_radius(new_gal):
    a_c = read_in_gal.nsc.a_c_parametric
    assert np.isclose(a_c, new_gal.component_fit_cluster_scale_radius)

def test_kernel_size(new_gal):
    kernel = read_in_gal.kernel_size.in_units("pc").value
    assert np.isclose(kernel, new_gal.kernel_size)

def test_na_abundances(new_gal):
    na_on_fe, masses = read_in_gal.nsc_abundances.x_on_fe_individual("Na")
    assert np.allclose(na_on_fe, new_gal.star_na_on_fe)
    assert np.allclose(masses, new_gal.star_masses)

def test_o_abundances(new_gal):
    o_on_fe, masses = read_in_gal.nsc_abundances.x_on_fe_individual("O")
    assert np.allclose(o_on_fe, new_gal.star_o_on_fe)
    assert np.allclose(masses, new_gal.star_masses)

def test_mg_abundances(new_gal):
    mg_on_fe, masses = read_in_gal.nsc_abundances.x_on_fe_individual("Mg")
    assert np.allclose(mg_on_fe, new_gal.star_mg_on_fe)
    assert np.allclose(masses, new_gal.star_masses)

def test_al_abundances(new_gal):
    al_on_fe, masses = read_in_gal.nsc_abundances.x_on_fe_individual("Al")
    assert np.allclose(al_on_fe, new_gal.star_al_on_fe)
    assert np.allclose(masses, new_gal.star_masses)



