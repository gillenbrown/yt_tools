from __future__ import division
import numpy as np
import math

from . import utils

def grid_resolution_steps(initial_resolution, final_resolution):
    # we will use increasingly smaller grid size as we get closer to the
    # real location of the maximum. I want to space those equally in log 
    # space.
    # the final grid resolution will be the accuracy the user requests

    # in each step, we want to decrease by no more than a factor of 3 in size.
    # so we take the log base 3 of the ratio of the max and min cell size to get
    # the number of factors of 3. Then we take the ceiling to turn that
    # into an integer. We then add 1 to this, to account for the fact that
    # we want steps at the beginning AND end.

    # The factor of three was chosen becuase we want to include plenty on either
    # side of the chosen best location each time to allow us to fix issues
    # caused by undersampling the grid. If we choose too large of a step, then
    # we may run into issues if the center isn't right on one of the grid
    # points chosen in a previous step. We always want to allow wiggle room.
    size_ratio = initial_resolution / final_resolution
    num_steps = np.ceil(math.log(size_ratio, 3.0)) + 1
    # I then evenly space in log space.
    return np.logspace(np.log10(initial_resolution),
                       np.log10(final_resolution), 
                       num=num_steps)

def construct_grid(cell_size, center_x, center_y, center_z=None, dimensions=3, 
                   points_per_side=5):
    """constructs a grid centered on the initial point"""
    # we will have a grid with 11 points, where the middle one is centered at
    # the exact location of the current center.
    steps = np.arange(-1 * points_per_side, points_per_side + 1, 1)  # -5 to 5
    xs = [center_x + cell_size * step for step in steps]
    ys = [center_y + cell_size * step for step in steps]
    # then turn these into a 1D array of 3 element tuples
    if dimensions == 2:
        return [(x, y) for x in xs for y in ys]
    elif dimensions == 3:
        zs = [center_z + cell_size * step for step in steps]
        return [(x, y, z) for x in xs for y in ys for z in zs]

def max_in_pairs(values, labels):
    # check that both are the same size
    if len(values) != len(labels):
        raise ValueError("values and labels needs to be the same size.")
    max_idx = np.argmax(values)
    return labels[max_idx]


class KDE(object):
    """Class used for doing kernel density estimation in multiple dimensions"""
    def __init__(self, locations, values=None):
        """ Initialize the KDE object.
        
        :param locations: Location in two or three dimensions of each point.
                          Pass in a list containing an array of x values, an
                          array of y values, and an optional list of z values.
        :param values: Value of whatever quantity we are doing a KDE for. This
                       functions like a weight for each point. If no values are
                       passed, all will be weighted equally.
        :type values: np.array

        Note that the length of values and each dimension of the location must
        be the same. 
        """

        # we need to know how many dimensions we are using. The user should have
        # passed in a list of arrays, so the number of arrays is the number of
        # dimensions their data is in.
        self.dimension = len(locations)

        # get the values out first, since we need to know what they are for 
        # error checking later
        if values is not None:  # if the user provided values
            self.values = values
        # if they didn't pass anything in, weight everything the same
        else:
            self.values = np.ones(len(locations[0]))

        # we also want to check that everything is iterable.
        try:
            iter(locations[0])
            iter(locations[1])
            if self.dimension == 3:
                iter(locations[2])
            iter(self.values)
        except(TypeError):
            raise TypeError("X, Y, Z, and value must be iterable.")

        # then get the x y z data
        if self.dimension == 3:
            self.x = locations[0]
            self.y = locations[1]
            self.z = locations[2]
            # we can keep track of which smoothing function to use. 
            self.smoothing_function = utils.gaussian_3d_radial
            # check that all are the right size
            if not len(self.values) == len(self.x) == len(self.y) == len(self.z):
                raise ValueError("X,Y,Z, and values need to be the same size.")
        elif self.dimension == 2:
            self.x = locations[0]
            self.y = locations[1]
            self.smoothing_function = utils.gaussian_2d_radial
            # check that all are the right size
            if not len(self.values) == len(self.x) == len(self.y):
                raise ValueError("X, Y, and values need to be the same size.")
        else:
            raise ValueError("We can only do KDE in 2 or 3 dimensions now.")

        # enter the parameters for the maximum finding function.
        self.location_max_x = None
        self.location_max_y = None
        self.location_max_z = None
    

    def density(self, kernel_size, x, y, z=None):
        # first check if the location passed in matches the dimensions we have
        if self.dimension == 3 and z is None:
            raise ValueError("In 3D, we need a z coordinate.")
        elif self.dimension == 2 and z is not None:
            raise ValueError("In 2D, we can't use a z coordinate.")

        # get the distances from the location the user passed in to the
        # location of all the points
        if self.dimension == 2:
            distances = utils.distance(self.x, x, self.y, y)
        elif self.dimension == 3:
            distances = utils.distance(self.x, x, self.y, y, self.z, z)

        # then we can calculate the Gaussian density at these distances
        densities = self.smoothing_function(distances, kernel_size)
        # we then multiply each density by the corresponding weight there, and
        # sum that over all points to get the density at this point
        return np.sum(densities * self.values)

    def _initial_cell_size(self):
        if len(self.x) == 1:
            return 1
        x_size = max(self.x) - min(self.x)
        y_size = max(self.y) - min(self.y)
        if self.dimension == 3:
            z_size = max(self.z) - min(self.z)
            return max([x_size, y_size, z_size]) / 10.0
        else:
            return max([x_size, y_size]) / 10.0

    def centering(self, kernel_size, accuracy, initial_guess=None,
                  search_region_size=None):
        """ Determines the location of the densest place in the KDE region

        :param kernel_size: Size of the smoothing kernel.
        :param accuracy: value of how precisely to determine the center. The 
                         center determined will be such that
                         abs(true_center - found_center) < accuracy
        :type accuracy: float
        :param initial_guess: A guess for the center. Will be used as the center
                              at the beginning, and the same process will be
                              done around it.
        :param search_region_size: How big of a box to search within to find
                                   the maximum stellar density. The region
                                   will be centered on initial_guess (if there
                                   is one) and will have a length of
                                   search_region_size

        The algorithm to do this will be to determine the density in a grid of
        points, then gradually refine this down by choosing a smaller and 
        smaller grid centered on the location of the maximum in the larger grid 
        until we get to the accuracy requested.
        """
        # specify the initial values to be in the middle of the values if one
        # wasn't passed in.
        if initial_guess is None:
            initial_x = (min(self.x) + max(self.x)) / 2.0
            initial_y = (min(self.y) + max(self.y)) / 2.0
            if self.dimension == 2:
                best_location = (initial_x, initial_y)
            elif self.dimension == 3:
                initial_z = (min(self.z) + max(self.z)) / 2.0
                best_location = (initial_x, initial_y, initial_z)
        else:
            best_location = tuple(initial_guess)  # best location will be a
                                                  # tuple at the end.

        # we need to iterate through the grid sizes
        if search_region_size is None:
            initial_size = self._initial_cell_size()
        else:
            initial_size = search_region_size / 10.0
            # divide by 10 since this parameter is the cell size, not box size

        for grid_resolution in grid_resolution_steps(initial_size, accuracy):
            # we use whichever is bigger of the kernel_size the user specified
            # or the grid size. We choose the grid size
            # because we don't want to under-sample the grid.
            this_kernel_size = max(kernel_size, 2 * grid_resolution)

            # then get the places to calculate the density
            locations = construct_grid(grid_resolution, *best_location,
                                       dimensions=self.dimension)

            # then calculate the density at all these locations. Unfortunately 
            # we have to do this iteratively at each location, since the 
            # calculation at each location is vectorized over each data point
            # in the KDE class
            densities = [self.density(this_kernel_size, *loc)
                         for loc in locations]
            densities = np.array(densities)

            # then get the maximum density and the location of that max value
            best_location = max_in_pairs(densities, locations)

        if self.dimension == 2:
            self.location_max_x, self.location_max_y = best_location
        if self.dimension == 3:
            x, y, z = best_location
            self.location_max_x = x
            self.location_max_y = y
            self.location_max_z = z


    def radial_profile(self, kernel_size, radii, num_each, center):
        """Create a radial KDE profile.
        
        :param kernel_size: Size of the smoothing kernel.
        :param radii: List of radii at which the density will be calculated.
                      Since KDE binning doesn't make sense, this will work by 
                      putting `num_each` points at each radius, evenly
                      distributed over azimuth. You can then bin these if you
                      like later.
        :param num_each: How many azimuthal points to put at each of the
                         specified radii.
        :param center: location around which the profile will be calculated.
        :returns: List of radii and densities corresponding to those radii.
        """
        # check that the length of the center matches the dimensions we have
        if len(center) != self.dimension:
            raise ValueError("The center must be of the same dimension as the "
                             "data")

        # get those locations we want to sample the density at. These are
        # relative to the center for now.
        # if self.dimension == 3:
        #     rel_locs = utils.get_3d_sperical_points(radii)
        if self.dimension == 2:
            rel_locs = utils.get_2d_polar_points(radii, num_each)
            repeated_radii = np.repeat(radii, num_each)
        else:
            raise ValueError("Only 2D for now.")

        # then turn these into real space locations by adding the galaxy
        # center to the relative locations.
        individual_locs = [rel_locs[i] + center[i] for i in range(len(center))]
        # ^ that is a two/three element list where each element is an array of
        # x/y/z values
        # then turn those into a list of tuples where each point is a single
        # (x, y, [z]) tuple
        locations = zip(*individual_locs)
        # these locations will still be sorted in order of increasing radius,
        # so they are easy to use. We still have to iterate, and can't do it
        # in a vectorized way, unfortunately.
        return repeated_radii, \
               np.array([self.density(kernel_size, *loc) for loc in locations])



        

    