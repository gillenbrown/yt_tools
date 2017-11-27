from __future__ import division

from yt import units
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
    except(ValueError):  # will happen if we pass in an array
        if any(radius < 0):
            raise RuntimeError("The radius must be nonnegative")


def gaussian_3d_radial(radius, sigma):
    """This is a simplified Gaussian for use here.

    This is a radial Gaussian in 3 dimensions. """

    _gaussian_error_checking(radius, sigma)

    # then we can calculate the Gaussian function
    exponent = (-1.0) * radius ** 2 / (2.0 * sigma ** 2)
    coefficient = 1.0 / (sigma ** 3 * (2.0 * np.pi) ** (1.5))
    return coefficient * np.exp(exponent)


def gaussian_2d_radial(radius, sigma):
    """This is a simplified Gaussian for use here.

    This is a radial Gaussian in 2 dimensions. """

    _gaussian_error_checking(radius, sigma)

    exponent = (-1) * radius ** 2 / (2 * sigma ** 2)
    coefficient = 1.0 / (sigma ** 2 * 2 * np.pi)
    return coefficient * np.exp(exponent)

def gaussian_1d(length, sigma):
    """Just a Gaussian in 1D, nothing fancy."""

    # Don't want regular error checking, since negative distance is fine.
    # Error checking: standard deviation must be positive;
    if sigma <= 0:
        raise ValueError("The standard deviation must be positive.")

    exponent = (-1) * length ** 2 / (2 * sigma ** 2)
    coefficient = 1.0 / np.sqrt(2 * np.pi * sigma**2)
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

    return np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)

def test_for_units(item, name):
    """Tests if an object has units in the yt system. Raises TypeError if not.
    
    :param item: object that needs to be tested.
    :param name: Name to be printed if this fails. 
    """
    try:
        these_units = item.units
    except AttributeError:  # will be raised if not a YTArray with units
        raise TypeError("{} must have be a quantity with units.".format(name))

    # also need to be actually what we want, and not just some other attribute
    if type(these_units) is not units.unit_object.Unit:
        raise TypeError("{} must have be a quantity with units.".format(name))

def test_iterable(item, name):
    """Sees if item is iterable. If not raises a TypeError. 
    
    :param item: object to check.
    :param name: Name to call the object in the error message if it fails."""
    try:
        iter(item)
    except TypeError:
        raise TypeError("{} must be an iterable. ".format(item))

def generate_random_theta_phi(number):
    """Generates random points on the surface of a sphere. 
    
    Returns lists of azimuthal and altitudinal angular coordinates. This is more
    complicated than it might be on first glace. 
    See http://mathworld.wolfram.com/SpherePointPicking.html for the algorithm
    I used (Equations 1 and 2, with theta and phi switched). 
    
    :param number: number of points to create.
    :return: two lists containing angular coordinate. The first contains the 
             azimuthal coordinate, the second the altitudinal. 
    """

    u = np.random.uniform(0, 1, number)
    v = np.random.uniform(0, 1, number)
    azimuth = 2 * np.pi * u
    altitude = np.arccos(2 * v - 1)

    return azimuth, altitude

def convert_polar_to_cartesian(r, phi):
    """Converts polar coordinate to cartesian.

    :param r: radius
    :param phi: azimuthal angularbcoordinate. 
    :returns x: list of x values
    :returns y: list of y values
    """
    return r * np.cos(phi), r * np.sin(phi)

def convert_spherical_to_cartesian(r, theta, phi):
    """Converts spherical coordinates to cartesian.
    
    :param r: radius
    :param theta: altitude angular coordinate
    :param phi: azimuthal angular coordinate 
    :returns: x, y, z
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z

def get_2d_polar_points(radii, num_each):
    """Generates points in 2D, such that each each radius there will be a point
    at a random azimuthal coordinate.  This is centered around zero.
    
    :param radii: list of radii
    :param num_each: number of points to put at each radius. These will be
                     randomly spaced in the azimuthal direction.
    :returns x: list of x values, as described above.
    :returns y: list of y values, as described above.
    """
    phi = np.random.uniform(0, 2 * np.pi, num_each)
    repeated_phi = np.tile(phi, len(radii))
    repeated_radii = np.repeat(radii, num_each)
    return convert_polar_to_cartesian(repeated_radii, repeated_phi)

def get_3d_sperical_points(radii):
    """Gets points in 3D. At each radius given, a point will be placed at a
    random location on a sphere of that radius.
    
    :param radii: list of radii
    :returns: three lists of x, y, and z values generated according to the
              rules outlined above.
    """
    theta, phi = generate_random_theta_phi(len(radii))
    return convert_spherical_to_cartesian(radii, theta, phi)

def weighted_mean(values, weights):
    """Calculate a weighted mean among some values.

    The weighted mean is here defined as

    .. math::
        \\bar{X} = \\frac{\\sum_i x_i w_i}{\\sum_i w_i}

    So conceptually it's the sum of the weights times the values divided by
    the sum of the weights.

    For some simple examples:
    - if the weights are all the same, we just get back the regular mean.
    - if values are [0, 1] and weights are [1, 2], we get 2/3.

    :param values: List or array of values we want to find the mean of.
    :param weights: list or array of weights.

    """
    # convert to arrays to make vectorized computation possible
    if not isinstance(values, np.ndarray):
        values = np.array(values, dtype=np.float64)
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights, dtype=np.float64)

    # make sure all weights are positive
    if not np.all(weights >= 0):
        raise ValueError("All weights must be non-negative.")

    # do error checking with try/except
    try:
        return np.sum(values * weights) / np.sum(weights)
    except ValueError:
        raise ValueError("Weights and values must be the same length.")


def weighted_variance(values, weights, ddof=1):
    """Calculate the weighted variance among some values.

    The weighted variance is defined as

    .. math::
        \\sigma^2 = \\frac{\\sum_i (x_i - \\bar{X})^2 w_i}{\\sum_i w_i - \\rm{ddof}}

    Where ddof is identical to that used by `np.var()`. According to the
    documentation for that:

    "In standard statistical practice, ddof=1 provides an unbiased estimator
    of the variance of a hypothetical infinite population. ddof=0 provides a
    maximum likelihood estimate of the variance for normally distributed
    variables."

    The default setting here is ddof=1.

    :param values: List or array of values to take the variance of.
    :param weights: List or array of weights. These weights MUST be "frequency
                    weights", which means that they correspons to the number
                    of times that the particular values was observed.
                    See Wikipedia for more info if desired:
                    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    :param ddof: described above. Defaults to 1.

    """
    # error checking is done by the weighted mean function.
    mean = weighted_mean(values, weights)
    return np.sum((values - mean)**2 * weights) / (np.sum(weights) - ddof)

def sum_in_quadrature(*args):
    """Sums the values in quadrature.

    This is defined as

    .. math::
        \\sqrt{x_0^2 + x_1^2 + \\ldots + x_{n-1}^2 + x_n^2}

    Just pass in all the values you want to sum. They can either be in a list
    or not.

    Examples:
    sum_in_quadrature(1, 2, 3) == sqrt(14)
    sum_in_quadrature([1, 2, 3]) == sqrt(14)

    """
    # if args is one element, it is either a list or one element.
    if len(args) == 1:
        # see if it's an array
        try:
            return np.sqrt((args[0] ** 2).sum())
        except TypeError:  # it's a list, and doesn't square like that.
            return np.sqrt(sum([x ** 2 for x in args[0]]))
        except AttributeError:  #it's a value, and doesn't have sum
            return args[0]  # square root of itself squared.
    # if we got here it's an else. We have a list of elements in args.
    return np.sqrt(sum([x**2 for x in args]))


def bin_values(values, bin_size=100):
    """
    Bins values together, returning the average in each bin. Does not return any
    information about the bin edges or anything else. This does binning where
    there are an equal amount of data points in each bin, and the bins are done
    in the order the data was passed in.

    When bin_size doesn't evenly divide the size of the dataset passed in, the
    last bin will be larger than the others.

    Some test cases:
    bin_values([1, 1, 2, 2, 3, 3, 4, 4], bin_size=2)
        will return [1, 2, 3, 4]
    bin_values([1, 2, 3, 4, 5, 6, 7, 8, 9], bin_size=3)
        will return [2, 5, 8]
    bin_values([1, 2, 3, 4, 5, 6, 7, 8], bin_size=3)
        will return [2, 6], since there aren't enough values to make three bins,
        and the last 5 values are all put into the last bin.

    :param values: List or array of values to bin
    :param bin_size: Size of bins. There will be this many data points in all
                     bins except the last one, which could have a larger
                     amount of data points if bin_size does not evenly divide
                     the length of the dataset.
    :return: array of values that is the average of the values in each bin,
             plus an array of the spread in each bin. These will be of
             size len(values) // bin_size
    :rtype: np.ndarray
    """

    binned_values = []
    # we want to iterate with constant bin size. If we have
    # a length that isn't divisible by bin_size, the extra stuff
    # at the end will all get put in one bin.
    # This is accomplished by itertaing through the indices
    # in steps of bin_size. We stop when we are somewhere between bin_size and
    # 2 * bin_size - 1 steps from the end.
    for left_idx in range(0, len(values) - bin_size + 1, bin_size):

        # we then get the items in this bin. If we are close to the end,
        # we have to get all the leftover items.
        if len(values) - left_idx < 2 * bin_size:
            these_values = values[left_idx:]
        else:  # we just get the appropriate values
            these_values = values[left_idx:left_idx + bin_size]

        # then keep the mean.
        binned_values.append(np.mean(these_values))

    return np.array(binned_values)

def sphere_intersection(r1, r2, separation):
    """Determines whether two spheres are in contact with each other.

    There is a mathematical formula describing the radius of the circle that
    forms the intersection of the two spheres.

    http://mathworld.wolfram.com/Sphere-SphereIntersection.html

    There is a square root in that. If the argument of that is positive, then we
    will have a real radius of intersection, so they will intersect. If it is
    negative, then the radius would be imaginary, so we do not have an
    intersection.

    :param r1: radius of one sphere
    :param r2: radius of the other sphere. The order in which you pass these two
               radii does not matter.
    :param separation: separation between the two spheres
    :returns: whether or not the two spheres intersect. Touching is defined as
              intersecting, since a sphere intersects with itself.
    :rtype: bool
    """
    # all values must be positive, first
    if r1 < 0 or r2 < 0 or separation < 0 :
        raise ValueError("Values passed in must be non-negative.")
    return 4 * separation**2 * r1**2 - (separation**2 - r2**2 + r1**2)**2 >= 0

def sphere_containment(cen_1, cen_2, r_1, r_2):
    """
    Test whether sphere 1 is contained within sphere 2.

    :param cen_1: three element list with the x, y, z coordinates of the center
                  of the first sphere
    :param cen_2: three element list with the x, y, z coordinates of the center
                  of the second sphere
    :param r_1: radius of one sphere
    :param r_2: radius of the other sphere. The order in which you pass these two
               radii does not matter.
    :returns: whether or not sphere 1 is contained within sphere 2. If they
              touch at any point this will be false. A sphere can not be
              contained within itself, for example.
    :rtype: bool
    """
    # error checking. I just have to check that the lists have three elements,
    # and are iterable, the base function will check the radii
    test_iterable(cen_1, "center")
    test_iterable(cen_2, "center")
    if len(cen_1) != 3 or len(cen_2) != 3:
        raise ValueError("Centers need to be a three element list.")
    if r_1 < 0 or r_2 < 0:
        raise ValueError("Radii passed in must be non-negative.")

    # if the first sphere is smaller than the second, the first can't be
    # contained within the second
    if r_1 >= r_2:
        return False

    # we need to know how far the spheres are apart
    x1, y1, z1 = cen_1
    x2, y2, z2 = cen_2
    dist = distance(x1, x2, y1, y2, z1, z2)

    # if the distance between them is larger than the radius of the second
    # sphere, then the first can't be contained in the second
    if dist > r_2:
        return False
    # if the radius of the second is larger than the separation, then the center
    # of the first is within the second. We can see if the whole thing is
    # contained by seeing whether or not there is an intersection between the
    # two spheres.
    return not sphere_intersection(r_1, r_2, dist)

def log_mean(a, b):
    """
    Returns the mean of the values in log space.

    This is defined as: 10^(mean(log(values))). A simple example would be that
    the mean of 1 (10^0) and 100 (10^2) would be 10 (10^1).

    :param a: first value to take a mean of
    :param b: second value to take a mean of
    :return: mean of the values in log space.
    """
    avg_log = np.mean(np.log10([a, b]))
    return 10**avg_log

def max_above_half(values):
    """
    Sees if the maximum value of an item in the list is more than half of the
    sum of all the items in the list.

    This is used to determine if there should be an upper limit on NSCs in the
    half mass radii.

    :param values: list of values
    :type values: list
    :return: Whether or not the maximum value in the list is more than half
             of the total sum of the list.
    :rtype: bool
    """
    return max(values) > 0.5 * sum(values)
