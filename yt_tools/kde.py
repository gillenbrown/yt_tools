import numpy as np

def _gaussian_error_checking(radius, sigma):
    """Checks that both the standard deviation and radius are positive"""

    # Error checking: standard deviation must be positive;
    if sigma <= 0:
        raise ValueError("The standard deviation must be positive.")
    # the radius must be positive too, but we need to check that for both
    # arrays and for single values.
    try:
        if radius < 0:  # will work if a single value
            raise RuntimeError("The radius must be nonnegative")
    except(ValueError): # will happen if we pass in an array
        if any(radius < 0):
            raise RuntimeError("The radius must be nonnegative")

def gaussian_3d_radial(radius, sigma):
    """This is a simplified Gaussian for use here.

    This is a radial Gaussian in 3 dimensions. """
    
    _gaussian_error_checking(radius, sigma)

    # then we can calculate the Gaussian function
    exponent = (-1) * radius**2 / (2 * sigma**2)
    coefficient = 1.0 / (sigma**3 * (2 * np.pi)**(1.5))
    return coefficient * np.exp(exponent)

def gaussian_2d_radial(radius, sigma):
    """This is a simplified Gaussian for use here.

    This is a radial Gaussian in 2 dimensions. """

    _gaussian_error_checking(radius, sigma)

    exponent = (-1) * radius**2 / (2 * sigma**2)
    coefficient = 1.0 / (sigma**2 * 2 * np.pi)
    return coefficient * np.exp(exponent)

def distance(x1, x2, y1=0, y2=0, z1=0, z2=0):
    """ Calculates a distance between two points using the Pythagorean theorem.

    Note: This does not support lists, since they aren't vectorized.
    """
    try:
        x_dist = x1 - x2
        y_dist = y1 - y2
        z_dist = z1 - z2
    except(TypeError):
        raise TypeError("This function does not support lists. Try np.array.")

    return np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)

def grid_resolution_steps(initial_resolution, final_resolution):
    # we will use increasingly smaller grid size as we get closer to the
    # real location of the maximum. I want to space those equally in log 
    # space.
    # the final grid resolution will be the accuracy the user requests

    # in each step, we want to decrease by roughly a factor of 10 in size.
    # so we take the log of the ratio of the max and min cell size to get
    # the number of factors of 10. Then we take the ceiling to turn that 
    # into an integer. We then add 1 to this, to account for the fact that
    # we want steps at the beginning AND end. 
    size_ratio = initial_resolution / final_resolution
    num_steps = np.ceil(np.log10(size_ratio)) + 1
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
            self.smoothing_function = gaussian_3d_radial
            # check that all are the right size
            if not len(self.values) == len(self.x) == len(self.y) == len(self.z):
                raise ValueError("X,Y,Z, and values need to be the same size.")
        elif self.dimension == 2:
            self.x = locations[0]
            self.y = locations[1]
            self.smoothing_function = gaussian_2d_radial
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
            distances = distance(self.x, x, self.y, y)
        elif self.dimension == 3:
            distances = distance(self.x, x, self.y, y, self.z, z)

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

    def centering(self, kernel_size, accuracy):
        """ Determines the location of the densest place in the KDE region

        :param accuracy: value of how precisely to determine the center. The 
                         center determined will be such that
                         abs(true_center - found_center) < accuracy
        :type accuracy: float

        The algorithm to do this will be to determine the density in a grid of
        points, then gradually refine this down by choosing a smaller and 
        smaller grid centered on the location of the maximum in the larger grid 
        until we get to the accuracy requested.
        """
        # specify the initial values to be in the middle of the values
        initial_x = (min(self.x) + max(self.x)) / 2.0
        initial_y = (min(self.y) + max(self.y)) / 2.0
        if self.dimension == 3:
            initial_z = (min(self.z) + max(self.z)) / 2.0

        # then actually use these values to initialize it.
        if self.dimension == 2:
            best_location = (initial_x, initial_y)
        elif self.dimension == 3:
            best_location = (initial_x, initial_y, initial_z)

        # we need to iterate through the grid sizes
        initial_size = self._initial_cell_size()
        for grid_resolution in grid_resolution_steps(initial_size, accuracy):
            # we use whichever is bigger of the kernel_size the user specified
            # or three times the grid size. We choose three times the grid size 
            # because we don't want to under-sample the grid.
            kernel_size = max(kernel_size, 3 * grid_resolution)

            # then get the places to calculate the density
            locations = construct_grid(grid_resolution, *best_location,
                                       dimensions=self.dimension)

            # then calculate the density at all these locations. Unfortunately 
            # we have to do this iteratively at each location, since the 
            # calculation at each location is vectorized over each data point
            # in the KDE class
            densities = [self.density(kernel_size, *loc) for loc in locations]
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
        

    