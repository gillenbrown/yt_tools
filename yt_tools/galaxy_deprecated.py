import yt
from yt import units
import numpy as np
import matplotlib.pyplot as plt
import betterplotlib as bpl 

from . import kde

class galaxy_deprecated(object):
    def __init__(self, dataset, center, radius=1000 * yt.units.pc,
                 disk_height=50 * yt.units.pc):
#         """ Create a galaxy object.
#
#         :param dataset: yt dataset object that contains this galaxy.
#         :param center: 3 element array with the center. Must have units.
#         :param radius: Radius that will be used to create both the sphere and disk
#                        objects. Can have a unit, but if no unit is passed in
#                        parsecs are assumed.
#         :param disk_height: Height of the disk that will be created by finding
#                             the angular momentum vector. If this is zero, then
#                             no disk will be created. As this is computationally
#                             expensive, this can be advantageous. As with the
#                             radius, this can be passed with units, but if no
#                             units are included parsecs are assumed.
#
#         """
        self.ds = dataset
#         self.center = center
#         # We have to check the units on the disk height and radius. It is just
#         # a number, it won't have any units, but if it does then we will just
#         # use those units
#         try:
#             radius.units
#         except AttributeError:
#             radius = radius * yt.units.pc
#
#         try:
#             disk_height.units
#         except AttributeError:
#             disk_height = disk_height * yt.units.pc
#
#         self.radius = radius
#         self.disk_height = disk_height
#
#         self.sphere = self.ds.sphere(center=center, radius=self.radius)
#         self.stellar_mass = np.sum(self.sphere[('STAR', "MASS")].in_units("msun"))
#
#         self.radii = dict()
#         self.densities = dict()
#
#         if disk_height > 0:
#             self.add_disk()
        

    # def add_disk(self, radius=None, disk_height=None):
    #     """ Adds a disk aligned with the angular momentum vector"""
    #     if radius is None:
    #         radius = self.radius
    #     if disk_height is None:
    #         disk_height = self.disk_height
    #
    #     j_vec = self.sphere.quantities.angular_momentum_vector()
    #     self.disk = self.ds.disk(center=self.center, normal=j_vec,
    #                                   radius=radius, height=disk_height)
    #     # j_vec = self.disk.quantities.angular_momentum_vector()
    #     # self.disk = self.ds.disk(center=self.center, normal=j_vec,
    #     #                               radius=radius, height=height)

    # def centering(self):
    #     """Determines the center of the stellar density profile. """
    #
    #     # the smoothing kernel we will use in the KDE process is the size of
    #     # the smallest cell in the sphere.
    #     kernel_size = np.min(self.sphere[("index", "dx")]).in_units("pc").value
    #
    #     # # use the maximum stellar density of the sphere to start with.
    #     star_cen = self.sphere.quantities.max_location(("deposit", "STAR_density"))
    #     # # the star_cen above has the density as the first value, then the
    #     # # next three values should be in code length but are in cm for some reason
    #     self.center = self.ds.arr([star_cen[i].value for i in [1, 2, 3]], "code_length")
    #
    #     # get all the items at the very beginning, to reduce computation time.
    #     # this is because accessing that data can take a while.
    #     # we also convert to numpy arrays because they are faster.
    #     temp_sphere = self.ds.sphere(center=self.center, radius=self.radius/2.0)
    #     star_x = np.array(temp_sphere[('STAR', 'POSITION_X')].in_units("pc"))
    #     star_y = np.array(temp_sphere[('STAR', 'POSITION_Y')].in_units("pc"))
    #     star_z = np.array(temp_sphere[('STAR', 'POSITION_Z')].in_units("pc"))
    #     masses = np.array(temp_sphere[('STAR', 'MASS')].in_units("msun"))
    #
    #     # If there are no star particles in the small sphere, don't mess with
    #     # anything, and just return
    #     if len(star_x) == 0:
    #         return
    #
    #     # then get this in parsecs, which is what we want to use
    #     self.center = self.center.in_units("pc").value
    #
    #     # We will to the KDE process on increasingly smaller scales. First
    #     # have a very large area, and find the center. Then use a smaller area,
    #     # and find the center in that region. Keep doing that until we have
    #     # a good center.
    #     max_kde_cell_size = self.radius.in_units("pc").value / 10.0
    #     min_kde_cell_size = kernel_size / 100.0
    #     # in each step, we want to decrease by roughly a factor of 10 in size.
    #     # so we take the log of the ratio of the max and min cell size
    #     n_steps = np.ceil(np.log10(max_kde_cell_size / min_kde_cell_size)) + 1
    #
    #     center_on_edge = []
    #     for kde_cell_size in np.logspace(np.log10(max_kde_cell_size),
    #                                      np.log10(min_kde_cell_size),
    #                                      n_steps):
    #         # To save some computation time, we only want to use the stars near
    #         # the center, especially if we are in a very tiny region. I will
    #         # use the full virial radius until the kde box size (which is 10
    #         # times the kde cell size) gets smaller than the kernel size. Then
    #         # we will use a box that is 5 times the kernel size. We'll reselect
    #         # the stars to use there too.
    #         if kernel_size > 10 * kde_cell_size:
    #             temp_sphere = self.ds.sphere(center=self.center,
    #                                          radius=5 * kernel_size * yt.units.pc)
    #             star_x = np.array(temp_sphere[('STAR', 'POSITION_X')].in_units("pc"))
    #             star_y = np.array(temp_sphere[('STAR', 'POSITION_Y')].in_units("pc"))
    #             star_z = np.array(temp_sphere[('STAR', 'POSITION_Z')].in_units("pc"))
    #             masses = np.array(temp_sphere[('STAR', 'MASS')].in_units("msun"))
    #
    #         center_on_edge.append(self.kde_iterations(kde_cell_size, star_x,
    #                                                   star_y, star_z, masses,
    #                                                   kernel_size))
    #
    #     self.center = self.center * yt.units.pc
    #     self.sphere = self.ds.sphere(center=self.center, radius=self.radius)
    #     if self.disk_height.value > 0:
    #         self.add_disk()
    #
    #     return all(center_on_edge)


    # def kde_iterations(self, kde_cell_size,
    #                    star_x, star_y, star_z, masses, kernel_size):
    #     """ Does one iteration of KDE centering.
    #
    #     Given a preliminary center, it finds a more precise center in the
    #     vicinity of the previous center.
    #
    #     :param kde_cell_size: Size of the discrete boxes where we will calculate
    #                           the kde density. This is not the same as the size of
    #                           the cells in the simulation, this will start much
    #                           larger (for early iterations) and can get smaller
    #                           for later ones.
    #     :param star_x, star_y, star_z: x, y, and z position of all the stars in the
    #                                    simulation, respectively. This does need to
    #                                    be all stars in a given galaxy, not just the
    #                                    ones at the very center, because they can all
    #                                    contribute to the density at the center.
    #                                    These should have units of parsecs.
    #     :param masses: array of masses of the star particles, in solar masses.
    #     :param kernel_size: Size of the Gaussian kernel used to to the KDE
    #                         estimation. Should be approximately the size of the
    #                         smallest cell in the simulation.
    #     :param plot: Whether to plot the density near the center in a 2D
    #                  histogram style plot. Useful for debugging only, basically.
    #     """
    #
    #     # get the location of the x, y, and z positions we will be calculating
    #     # the KDE density at. There will be 11 locations in each dimension, spaced
    #     # according to the kde_cell_size parameter.
    #     steps = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    #     xs = [self.center[0] + kde_cell_size * step for step in steps]
    #     ys = [self.center[1] + kde_cell_size * step for step in steps]
    #     zs = [self.center[2] + kde_cell_size * step for step in steps]
    #     # then turn these into a 1D array of x, y, z tuples.
    #     locations = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    #     step_tuple = [(x, y, z) for x in steps for y in steps for z in steps]
    #
    #     # then calculate the density at all these locations.
    #     densities = [self.kde(loc, [star_x, star_y, star_z], masses, kernel_size)
    #                  for loc in locations]
    #     densities = np.array(densities)
    #
    #     # Then find the maximum value, and the location at that value.
    #     max_location = np.argmax(densities)
    #     self.center = locations[max_location]
    #
    #     step_of_max = step_tuple[max_location]
    #     if 5 in step_of_max or -5 in step_of_max:
    #         return True
    #     else:
    #         return False


    # @staticmethod
    # def kde(location, star_coords, masses, size):
    #     """ This calculates the density at a given point
    #
    #     This takes the array of star positions in x, y, and z, plus their masses,
    #     and constructs the Gaussian density at the given point, using the kernel
    #     size given by `size`
    #
    #     :param location: 3 value array with a single set of x, y, z value where we
    #                      will calculate the kde density.
    #     :param star_coords: list of arrays holding the x, y, and possibly z
    #                         positions of all the stars.
    #     :param masses: array of the masses of the star particles. Should have units
    #                    of solar masses.
    #     :param size: value of the standard deviation to use in the Gaussian kernel.
    #                  should be about the size of the smallest cell in the simulation
    #     """
    #     if len(location) != len(star_coords):
    #         raise ValueError("Length of location and star_coords must be same.")
    #
    #     # get the distance of each star from the point of interest (Pythag. theorem)
    #     sum_squares = 0
    #     for star, loc in zip(star_coords, location):
    #         sum_squares += (star - loc)**2
    #     distance = np.sqrt(sum_squares)
    #
    #     # Then get the density of each star particle at this location, weighted
    #     # by the mass. This will be an array with the density for each star.
    #     if len(location) == 2:
    #         density =  kde.gaussian_2d_radial(distance, size) * masses
    #     elif len(location) == 3:
    #         density =  kde.gaussian_3d_radial(distance, size) * masses
    #     else:
    #         raise ValueError("Only works for 2 or 3 dimensions.")
    #     # we want the total density, so sum the contributions of all the star
    #     # particles
    #     return np.sum(density)

    # def kde_profile_spherical(self, quantities=["MASS"], spacing=0.1,
    #                              outer_radius=1000):
    #     """
    #     Create a radial profile of a given quantity of the stars, typicall mass.
    #
    #     :param quantity: the parameter that will be plotted. Has to be
    #                      associated with the stars, such that they are stored
    #                      in the form ("STAR", `quantity`). The default here is
    #                      "MASS".
    #     :param spacing: The density will be calculated from zero to
    #                     `outer_radius` with a spacing determined by this
    #                     variable. The points will be equally spaced in radius,
    #                     while the angular coordinates will be random. The units
    #                     of `spacing` need to be parsecs.
    #     :param outer_radius: Maximum radius out to which to calculate the radial
    #                          profile. Is in parsecs.
    #     """
    #
    #     # get all the radii at which to calculate the density
    #     radii = np.arange(0, outer_radius, spacing)
    #     # then generate the random angular coordinates
    #     # It's more complicated than it looks,
    #     # see http://mathworld.wolfram.com/SpherePointPicking.html
    #     # Equations 1 and 2 are what I used, with switched theta and phi
    #     u = np.random.uniform(0, 1, len(radii))
    #     v = np.random.uniform(0, 1, len(radii))
    #     theta = np.arccos(2 * v - 1)
    #     phi = 2 * np.pi * u
    #
    #     # turn this into x,y,z relative to the center
    #     x_rel = radii * np.sin(theta) * np.cos(phi)
    #     y_rel = radii * np.sin(theta) * np.sin(phi)
    #     z_rel = radii * np.cos(theta)
    #
    #     # then turn them into absolute locations
    #     x = self.center[0].in_units("pc").value + x_rel
    #     y = self.center[1].in_units("pc").value + y_rel
    #     z = self.center[2].in_units("pc").value + z_rel
    #     locations = zip(x, y, z)
    #
    #     # we have to get the quantities needed to calculate the KDE density.
    #     # we also convert to numpy arrays because they are faster.
    #     star_x = np.array(self.sphere[('STAR', 'POSITION_X')].in_units("pc"))
    #     star_y = np.array(self.sphere[('STAR', 'POSITION_Y')].in_units("pc"))
    #     star_z = np.array(self.sphere[('STAR', 'POSITION_Z')].in_units("pc"))
    #     mass = np.array(self.sphere[('STAR', "MASS")].in_units("msun"))
    #     star_locations = [star_x, star_y, star_z]
    #
    #     # the smoothing kernel we will use in the KDE process is the size of
    #     # the smallest cell in the simulation
    #     kernel_size = np.min(self.sphere[('index', 'dx')].in_units("pc")).value
    #
    #     for quantity in quantities:
    #         if quantity == "MASS":
    #             densities = [self.kde(loc, star_locations, mass, kernel_size)
    #                          for loc in locations]
    #             self.radii["mass_kde_spherical"] = radii
    #             self.densities["mass_kde_spherical"] = np.array(densities)
    #
    #         elif quantity == "Z":
    #             # we want to use the already existing mass profile for
    #             # metallicity, too, so we don't have to calcluate it again.
    #             try:
    #                 self.densities["mass_kde_spherical"]
    #                 self.radii["mass_kde_spherical"]
    #             except KeyError:
    #                 # Mass hasn't been calculated, so we can just do that.
    #                 quantities.append("MASS")
    #                 quantities.append("Z")
    #                 continue
    #
    #             # If the mass profile has been calculated with a different
    #             # set of radii, we need to recalculate the mass profile for our
    #             # radii, but without overwriting the old radii.
    #             if not np.array_equal(self.radii["mass_kde_spherical"], radii):
    #                 mass_densities = [self.kde(loc, star_locations, mass, kernel_size)
    #                                  for loc in locations]
    #                 mass_densities = np.array(mass_densities)
    #             else:
    #                 mass_densities = self.densities["mass_kde_spherical"]
    #
    #             # We can then get the metal quantities, and turn them into
    #             # the total metallicity.
    #             z_Ia = np.array(self.sphere[('STAR', 'METALLICITY_SNIa')])
    #             z_II = np.array(self.sphere[('STAR', 'METALLICITY_SNII')])
    #
    #             total_z = z_Ia + z_II
    #
    #             total_metals = total_z * mass
    #
    #             metal_densities = [self.kde(loc, star_locations, total_metals, kernel_size)
    #                                for loc in locations]
    #
    #             self.densities["z_kde_spherical"] = np.array(metal_densities) / mass_densities
    #             self.radii["z_kde_spherical"] = radii
    #         else:
    #             print("{} is not implemented yet.".format(quantity))
        

    def kde_profile_cylindrical(self, quantities=["MASS"], 
                                spacing=0.1, outer_radius=500):
        """
        Create a radial profile of metallicity
        :param spacing: The density will be calculated from zero to 
                        `outer_radius` with a spacing determined by this 
                        variable. The points will be equally spaced in radius,
                        while the angular coordinates will be random. The units
                        of `spacing` need to be parsecs.
        :param outer_radius: Maximum radius out to which to calculate the radial
                             profile. Is in parsecs. 
         """
        # # get all the radii at which to calculate the density
        # radii = np.arange(0, outer_radius, spacing)
        # # then generate the random angular coordinate
        # theta = np.random.uniform(0, 2 * np.pi, len(radii))
        #
        # # turn this into x,y relative to the center. This is all we need, since
        # # the cylindrical coordinates the star particles use are relative
        # # to the center as well.
        # x = radii * np.cos(theta)
        # y = radii * np.sin(theta)
        #
        # locations = zip(x, y)

        # we have to get the quantities needed to calculate the KDE density.
        # we also convert to numpy arrays because they are faster.
        
        # We want the metallicity at each point, which is the sum of the metals
        # divided by the stellar mass there. To get metals we multiply Z times
        # the stellar mass. We calcluate sum(Z_i m_i) / sum(m_i) 
        # star_r = np.array(self.disk[('STAR', 'particle_position_cylindrical_radius')].in_units("pc"))
        # star_theta = np.array(self.disk[('STAR', 'particle_position_cylindrical_theta')])
        #
        # star_x = star_r * np.cos(star_theta)
        # star_y = star_r * np.sin(star_theta)
        # mass = np.array(self.disk[('STAR', "MASS")].in_units("msun"))
        # star_locations = [star_x, star_y]
        #
        # # the smoothing kernel we will use in the KDE process is the size of
        # # the smallest cell in the simulation
        # kernel_size = np.min(self.disk[('index', 'dx')].in_units("pc")).value
        #
        # for quantity in quantities:
        #     if quantity == "MASS":
        #         densities = [self.kde(loc, star_locations, mass, kernel_size)
        #                      for loc in locations]
        #         self.radii["mass_kde_cylindrical"] = radii
        #         self.densities["mass_kde_cylindrical"] = np.array(densities)
        #
        #     elif quantity == "Z":
        #         # we want to use the already existing mass profile for
        #         # metallicity, too, so we don't have to calcluate it again.
        #         try:
        #             self.densities["mass_kde_cylindrical"]
        #             self.radii["mass_kde_cylindrical"]
        #         except KeyError:
        #             # Mass hasn't been calculated, so we can just do that.
        #             quantities.append("MASS")
        #             quantities.append("Z")
        #             continue
        #
        #         # If the mass profile has been calculated with a different
        #         # set of radii, we need to recalculate the mass profile for our
        #         # radii, but without overwriting the old radii.
        #         if not np.array_equal(self.radii["mass_kde_cylindrical"], radii):
        #             mass_densities = [self.kde(loc, star_locations, mass, kernel_size)
        #                              for loc in locations]
        #             mass_densities = np.array(mass_densities)
        #         else:
        #             mass_densities = self.densities["mass_kde_cylindrical"]
        #
        #         # We can then get the metal quantities, and turn them into
        #         # the total metallicity.
        #         z_Ia = np.array(self.disk[('STAR', 'METALLICITY_SNIa')])
        #         z_II = np.array(self.disk[('STAR', 'METALLICITY_SNII')])
        #
        #         total_z = z_Ia + z_II
        #
        #         total_metals = total_z * mass
        #
        #         metal_densities = [self.kde(loc, star_locations, total_metals, kernel_size)
        #                            for loc in locations]
        #
        #         self.densities["z_kde_cylindrical"] = np.array(metal_densities) / mass_densities
        #         self.radii["z_kde_cylindrical"] = radii
        #     else:
        #         print("{} is not implemented yet.".format(quantity))

    def bin_profile(self, quantities=["MASS"], coords="spherical",
                            spacing=10, outer_radius=500):
        """ Calculates a profile of a given quantity in bins"""

        # get the bin centers, The inner radius of the first bin is at zero.
        mid_radii = np.arange(spacing/2.0, outer_radius, spacing) * yt.units.pc

        # depending on what coordinate system the user wants us to use, we 
        # need to get the star radii in those coordinate systems. We use the 
        # appropriate data container to make sure the coordinate system is
        # right
        if coords == "spherical":
            shape = self.sphere
            star_radii = self.sphere[('STAR', "particle_position_spherical_radius")].in_units("pc")
        elif coords == "cylindrical":
            shape = self.disk
            star_radii = self.disk[('STAR', "particle_position_cylindrical_radius")].in_units("pc")
        
        # then get things we will use often. We won't use metallicity every
        # time, but it's easier to get that data outside of the loop.
        mass = np.array(shape[("STAR", "MASS")].in_units("msun"))
        z_Ia = np.array(shape[('STAR', 'METALLICITY_SNIa')])
        z_II = np.array(shape[('STAR', 'METALLICITY_SNII')])
        
        # We can then calculate the profiles for all the quantities the user
        # asked for
        for quantity in quantities:
            # start a list, which will be appended to.
            this_quantity = []
            # iterate through each bin center
            for m_r in mid_radii:
                # find the star particles that are in this bin. We check their
                # distance away from the bin center, and make sure it's less 
                # than half of the bin width.
                good_idx = np.where(np.abs(star_radii - m_r) < spacing / 2.0)
                # WE then do different things for different profiles.
                if quantity == "MASS":
                    # get the mass values
                    this_mass = mass[good_idx]

                    # we need the volume or area for each bin
                    inner_radius = m_r - (spacing / 2.0) * yt.units.pc
                    outer_radius = m_r + (spacing / 2.0) * yt.units.pc
                    if coords == "spherical":
                        area = np.pi * (4/3)*(outer_radius**3 - inner_radius**3)
                    elif coords == "cylindrical":
                        area = np.pi * (outer_radius**2 - inner_radius**2)
                    # we can then get the mass density.
                    this_quantity.append(np.sum(this_mass) / area.value)
                elif quantity == "Z":
                    # for metallicity we will calculate the total metals mass
                    # in each bin, then divide by the total stellar mass.
                    # We first get the quantities in this bin
                    this_mass = mass[good_idx]
                    this_z_Ia = z_Ia[good_idx]
                    this_z_II = z_II[good_idx]
                    # total metallicity (Z) is the sum of the components
                    total_z = this_z_Ia + this_z_II
                    # we multiply by the mass to get the total metal mass
                    total_metals = total_z * this_mass
                    # We can then recalculate the metallicity
                    this_quantity.append(np.sum(total_metals) / np.sum(this_mass))

            # We can then put the quantities in the appropriate location 
            label = "{}_bins_{}".format(quantity.lower(), coords)
            self.radii[label] = mid_radii
            self.densities[label] = np.array(this_quantity)


                



