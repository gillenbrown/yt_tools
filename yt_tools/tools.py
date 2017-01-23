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

class yt_kde(object):
    def __init__(self, sphere, particle_id, center=None):
        self.sphere = sphere
        self.particle_id = particle_id
        self.center = center

    def centering(self):
        """Determines the center of the stellar density profile. """
        # get all the items at the very beginning, to reduce computation time.
        # this is because accessing that data can take a while.
        # we also convert to numpy arrays because they are faster.
        star_x = np.array(self.sphere[('STAR', 'POSITION_X')].in_units("pc"))
        star_y = np.array(self.sphere[('STAR', 'POSITION_Y')].in_units("pc"))
        star_z = np.array(self.sphere[('STAR', 'POSITION_Z')].in_units("pc"))
        masses = np.array(self.sphere[('STAR', 'MASS')].in_units("msun"))
        # the smoothing kernel we will use in the KDE process is the size of
        # the smallest cell in the simulation
        kernel_size = np.min(self.sphere[('index', 'dx')].in_units("pc")).value
        
        # use the center of the sphere to start with.
        self.center = self.sphere.center.in_units("pc").value
    
        # We will to the KDE process on increasingly smaller scales. First 
        # have a very large area, and find the center. Then use a smaller area,
        # and find the center in that region. Keep doing that until we have 
        # a good center.
        for kde_cell_size in np.logspace(np.log10(kernel_size), 
                                         np.log10(kernel_size) - 1.9, 
                                         3):
            self.kde_iterations(kde_cell_size, star_x, star_y, 
                                star_z, masses, kernel_size)

        self.center = self.center * yt.units.pc

    def kde_iterations(self, kde_cell_size,
                       star_x, star_y, star_z, masses, kernel_size):
        """ Does one iteration of KDE centering. 

        Given a preliminary center, it finds a more precise center in the 
        vicinity of the previous center.

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
        xs = [self.center[0] + kde_cell_size * step for step in steps]
        ys = [self.center[1] + kde_cell_size * step for step in steps]
        zs = [self.center[2] + kde_cell_size * step for step in steps]
        # then turn these into a 1D array of x, y, z tuples.
        locations = np.array([(x, y, z) for x in xs for y in ys for z in zs])
        
        # then calculate the density at all these locations.
        densities = [self.kde(loc, star_x, star_y, star_z, masses, kernel_size) 
                     for loc in locations]
        densities = np.array(densities)

        # Then find the maximum value, and the location at that value.
        test_max = np.argmax(densities)
        self.center = locations[test_max]


    @staticmethod
    def gaussian_3d_radial(radius, sigma):
        """This is a simplified Gaussian for use here.

        This is a radial Gaussian in 3 dimensions. """
        exponent = (-1) * radius**2 / (2 * sigma**2)
        coefficient = 1.0 / (sigma**3 * (2 * np.pi)**(1.5))
        return coefficient * np.exp(exponent)

    @staticmethod
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
        density =  yt_kde.gaussian_3d_radial(distance, size) * masses
        # we want the total density, so sum the contributions of all the star 
        # particles
        return np.sum(density)

    def radial_profile(self, quantity="MASS", spacing=0.1, outer_radius=500):
        """
        Create a radial profile of a given quantity of the stars, typicall mass.

        :param quantity: the parameter that will be plotted. Has to be
                         associated with the stars, such that they are stored
                         in the form ("STAR", `quantity`). The default here is
                         "MASS".
        :param spacing: The density will be calculated from zero to 
                        `outer_radius` with a spacing determined by this 
                        variable. The points will be equally spaced in radius,
                        while the angular coordinates will be random. The units
                        of `spacing` need to be parsecs.
        :param outer_radius: Maximum radius out to which to calculate the radial
                             profile. Is in parsecs. 
        """
        # get all the radii at which to calculate the density
        radii = np.arange(0, outer_radius, spacing)
        # then generate the random angular coordinates
        # It's more complicated than it looks, 
        # see http://mathworld.wolfram.com/SpherePointPicking.html
        # Equations 1 and 2 are what I used, with switched theta and phi
        u = np.random.uniform(0, 1, len(radii))
        v = np.random.uniform(0, 1, len(radii))
        theta = np.arccos(2 * v - 1)
        phi = 2 * np.pi * u

        # turn this into x,y,z relative to the center
        x_rel = radii * np.sin(theta) * np.cos(phi)
        y_rel = radii * np.sin(theta) * np.sin(phi)
        z_rel = radii * np.cos(theta)

        # then turn them into absolute locations
        x = self.center[0].in_units("pc").value + x_rel
        y = self.center[1].in_units("pc").value + y_rel
        z = self.center[2].in_units("pc").value + z_rel
        locations = zip(x, y, z)

        # we have to get the quantities needed to calculate the KDE density.
        # we also convert to numpy arrays because they are faster.
        star_x = np.array(self.sphere[('STAR', 'POSITION_X')].in_units("pc"))
        star_y = np.array(self.sphere[('STAR', 'POSITION_Y')].in_units("pc"))
        star_z = np.array(self.sphere[('STAR', 'POSITION_Z')].in_units("pc"))
        new_quantity = ('STAR', "{}".format(quantity))
        values = np.array(self.sphere[new_quantity].in_units("msun"))
        # the smoothing kernel we will use in the KDE process is the size of
        # the smallest cell in the simulation
        kernel_size = np.min(self.sphere[('index', 'dx')].in_units("pc")).value

        # then we can get the KDE density
        densities = [self.kde(loc, star_x, star_y, star_z, values, kernel_size) 
                     for loc in locations]
        densities = np.array(densities)

        self.density_radii = radii
        self.densities = densities










#TODO: rewrite this in terms of the class functions
    # def average_profile(radius, densities):
    #     r_d_pairs = np.vstack([radius, densities]).T
    #     r_d_pairs = sorted(r_d_pairs, key=lambda x: x[0])
    #     if log:
    #         bins = np.logspace(np.log10(np.sort(radius)[1]), np.log10(max(radius)), 100)
    #     else:
    #         bins = np.linspace(min(radius), max(radius), 100)
            
            
    #     x_avg, y_avg = [], []
    #     for i in range(len(bins) - 1):
    #         le = bins[i]
    #         re = bins[i + 1]
    #         good = [pair for pair in r_d_pairs if le < pair[0] < re]
    #         if len(good) == 0:
    #             if len(y_avg) != 0:
    #                 avg_d = y_avg[-1]
    #             else: 
    #                 avg_d = 0
    #         else:
    #             avg_d = sum([pair[1] for pair in good]) / len(good)
            
    #         x_avg.append((bins[i] + bins[i + 1]) / 2.0)
    #         y_avg.append(avg_d)