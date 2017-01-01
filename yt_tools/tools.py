import yt
from yt import units
import numpy as np
import matplotlib.pyplot as plt
import betterplotlib as bpl

def radial_profile(sphere, max_r, spacing):
    """
    Create a radial stellar density profile around the center of a sphere.

    :param sphere: yt sphere object containing all the data
    :param max_r: maximum radius to calculate the profile out to. Needs to have
                  a unit attached to it!
    :param spacing: Spacing used to bin the radius. A small one will allow you
                    to see individual large star particles. This must also have
                    a radius attached to it.
    :returns: two lists. The first contains the bin centers used to calculate 
              the density profile, and the second contains the stellar density
              at these radii in solar masses per cubic parsec.
    """

    # first get the positions and masses of all the stars in the sphere
    x = sphere[('STAR', 'POSITION_X')]
    y = sphere[('STAR', 'POSITION_Y')]
    z = sphere[('STAR', 'POSITION_Z')]
    masses = sphere[('STAR', 'MASS')]
    
    # get the center of the sphere
    center = sphere.center
    
    # then compute the radius of each star particle in the sphere
    x_diff = x - center[0]
    y_diff = y - center[1]
    z_diff = z - center[2]
    radius = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
    
    # create our binning in radius. We need to check that the user passed in
    # a value with units. We create inner and outer radii arrays, plus the 
    # center of each bin, just for convenience later
    try:
        inner_radii = np.arange(0, max_r.value, spacing.value) * units.pc 
    except AttributeError:
        raise ValueError("Please pass in both `max_r` and `spacing` "
                         "with yt units.")
    outer_radii = inner_radii + spacing
    bin_centers = (inner_radii + outer_radii) / 2.0
    
    # initialize stellar density array before filling it
    star_density = []
    for i_r, o_r, bin_center in zip(inner_radii, outer_radii, bin_centers):
        # find the stars that are in this bin
        good_idx = np.where(abs(radius - bin_center) < spacing / 2.0)
        # and calculate the total mass in solar masses of stars in this bin
        total_mass = masses[good_idx].sum().in_units("msun")

        # calculate the volume of this radial bin
        volume = (4.0 / 3.0) * np.pi * (o_r**3 - i_r**3)
        # and then calculate the density
        star_density.append(total_mass / volume)

        # plot for debugging
#         fig, ax = plt.subplots()
#         ax.hist(radius[good_idx].in_units("pc"))
#         ax.text(0.5, 0.5, str(bin_center), transform=ax.transAxes)
        
    return bin_centers, star_density

# --------------------------- KDE PROFILES ------------------------------------  



def gaussian_3d_radial(radius, sigma):
    """This is a simplified Gaussian for use here.

    This is a radial Gaussian in 3 dimensions. """
    exponent = (-1) * radius**2 / (2 * sigma**2)
    coefficient = 1.0 / (sigma**3 * (2 * np.pi)**(1.5))
    return coefficient * np.exp(exponent)

def kde(location, star_x, star_y, star_z, masses, size):
    """ This calculates the density at a given point

    This takes the array of star positions in x, y, and z, plus their masses,
    and constructs the Gaussian density at the given point, using the kernel
    size given by `size`

    :param location: 3 value array with a single set of x, y, z value where we
                     will calculate the kde density.
    :param star_x: array with x positions of the stars. Should be in parsecs.
    :param star_y, star_z: same thing
    :param masses: array of the masses of the star particles. Should have units
                   of solar masses.
    :param size: value of the standard deviation to use in the Gaussian kernel.
                 should be about the size of the smallest cell in the simulation
    """
    # get each component of the location
    x, y, z = location
    
    # get the distance of each star from the point of interest (Pythag. theorem)
    diffs_x = star_x - x
    diffs_y = star_y - y
    diffs_z = star_z - z
    distance = np.sqrt(diffs_x**2 + diffs_y**2 + diffs_z**2)
    
    # Then get the density of each star particle at this location, weighted
    # by the mass. This will be an array with the density for each star.
    density = gaussian_3d_radial(distance, size) * masses
    # we want the total density, so sum the contributions of all the star 
    # particles
    return np.sum(density)


def kde_iterations(center, kde_cell_size,
                   star_x, star_y, star_z, masses, kernel_size, plot=False):
    """ Does one iteration of KDE centering. 

    Given a preliminary center, it finds a more precise center in the 
    vicinity of the previous center.

    :param center: 3-element array with x, y, z position of the old center.
                   Should have units of parsecs
    :param kde_cell_size: Size of the discrete boxes where we will calculate
                          the kde density. This is not the same as the size of 
                          the cells in the simulation, this will start much 
                          larger (for early iterations) and can get smaller
                          for later ones.
    :param star_x, star_y, star_z: x, y, and z position of all the stars in the
                                   simulation, respectively. This does need to 
                                   be all stars in a given galaxy, not just the
                                   ones at the very center, because they can all
                                   contribute to the density at the center. 
                                   These should have units of parsecs.
    :param masses: array of masses of the star particles, in solar masses.
    :param kernel_size: Size of the Gaussian kernel used to to the KDE 
                        estimation. Should be approximately the size of the
                        smallest cell in the simulation. 
    :param plot: Whether to plot the density near the center in a 2D 
                 histogram style plot. Useful for debugging only, basically.

    """    
    # get the location of the x, y, and z positions we will be calculating
    # the KDE density at. There will be 11 locations in each dimension, spaced 
    # according to the kde_cell_size parameter. 
    steps = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    xs = [center[0] + kde_cell_size * step for step in steps]
    ys = [center[1] + kde_cell_size * step for step in steps]
    zs = [center[2] + kde_cell_size * step for step in steps]
    # then turn these into a 1D array of x, y, z tuples.
    locations = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    
    # then calculate the density at all these locations.
    densities = [kde(loc, star_x, star_y, star_z, masses, kernel_size) 
                 for loc in locations]
    densities = np.array(densities)

    # Then find the maximum value, and the location at that value.
    test_max = np.argmax(densities)
    center = locations[test_max]

    # then we can plot if we want. We will turn things into a yt data set to 
    # visualize it. 
    if plot:
        num = len(xs)
        densities3d = np.reshape(densities, [num, num, num])

        bbox = np.array([[xs[0], xs[-1]], [ys[0], ys[-1]], [zs[0], zs[-1]]])
        good_x, good_y, good_z = [], [], []
        for (x, y, z) in zip(star_x, star_y, star_z):
            if xs[0] < x < xs[-1] and ys[0] < y < ys[-1] and zs[0] < z < zs[-1]:
                good_x.append(x)
                good_y.append(y)
                good_z.append(z)
                
        data_arr = dict(density = (densities3d, "msun/pc**3"),
                        number_of_particles = len(good_x),
                        particle_position_x = (np.array(good_x), 'pc'),
                        particle_position_y = (np.array(good_y), 'pc'),
                        particle_position_z = (np.array(good_z), 'pc'))
        my_data = yt.load_uniform_grid(data_arr, densities3d.shape, 
                                       length_unit="pc", bbox=bbox)
#         plot = yt.ProjectionPlot(my_data, "x", "density")
        plot.annotate_particles()
        plot.show()
    
    return locations, densities, center

def kde_density_profile(yt_sphere, plot=False, log=False):
    star_x = np.array(yt_sphere[('STAR', 'POSITION_X')].in_units("pc"))
    star_y = np.array(yt_sphere[('STAR', 'POSITION_Y')].in_units("pc"))
    star_z = np.array(yt_sphere[('STAR', 'POSITION_Z')].in_units("pc"))
    masses = np.array(yt_sphere[('STAR', 'MASS')].in_units("msun"))
    size = np.min(yt_sphere[('index', 'dx')].in_units("pc")).value
    
    center = yt_sphere.center.in_units("pc").value
    
    total_locations = None
    total_densities = None
    
    for cell_size in np.logspace(np.log10(50), np.log10(size / 10.0), 4):
        l, d, center = kde_iterations(center, cell_size, star_x, star_y,
                                      star_z, masses, size, plot)
        if total_locations == None:
            total_locations = l
            total_densities = d
        else:
            total_locations = np.concatenate([total_locations, l])
            total_densities = np.concatenate([total_densities, d])

    kde_profile_plot(center, total_locations, total_densities, log=log)
    return total_locations, total_densities, center

def kde_profile_plot(center, locations, densities, log=False):
    xs, ys, zs = locations.T
    
    diff_x = xs - center[0]
    diff_y = ys - center[1]
    diff_z = zs - center[2]
    
    radius = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
    
    r_d_pairs = np.vstack([radius, densities]).T
    r_d_pairs = sorted(r_d_pairs, key=lambda x: x[0])
    if log:
        bins = np.logspace(np.log10(np.sort(radius)[1]), np.log10(max(radius)), 100)
    else:
        bins = np.linspace(min(radius), max(radius), 100)
        
    x_avg, y_avg = [], []
    for i in range(len(bins) - 1):
        le = bins[i]
        re = bins[i + 1]
        good = [pair for pair in r_d_pairs if le < pair[0] < re]
        if len(good) == 0:
            if len(y_avg) != 0:
                avg_d = y_avg[-1]
            else: 
                avg_d = 0
        else:
            avg_d = sum([pair[1] for pair in good]) / len(good)
        
        x_avg.append((bins[i] + bins[i + 1]) / 2.0)
        y_avg.append(avg_d)
        
    
    fig, ax = plt.subplots()
    ax.scatter(radius, densities, lw=0.1, c=bpl.color_cycle[2], alpha=0.5)
    ax.plot(x_avg, y_avg, c=bpl.almost_black, lw=2)
    if log:
        ax.set_xscale("log")
    else:
        ax.set_xlim(left=0)
    ax.set_yscale("log")
    ax.set_xlabel("Radius [pc]")
    ax.set_ylabel(r"Density [$\mathregular{M_\odot / pc^3}$]")