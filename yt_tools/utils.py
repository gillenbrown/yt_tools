from yt import units
import numpy as np

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

def get_2d_polar_points(radii):
    """Generates points in 2D, such that each each radius there will be a point
    at a random azimuthal coordinate. 
    
    :param radii: list of radii
    :returns x: list of x values, as described above.
    :returns y: list of y values, as described above.
    """
    phi = np.random.uniform(0, 2 * np.pi, len(radii))
    return convert_polar_to_cartesian(radii, phi)

def get_3d_sperical_points(radii):
    """Gets points in 3D. At each radius given, a point will be placed at a 
    random location on a sphere of that radius.
    
    :param radii: list of radii
    :returns: three lists of x, y, and z values generated according to the 
              rules outlined above.
    """
    theta, phi = generate_random_theta_phi(len(radii))
    return convert_spherical_to_cartesian(radii, theta, phi)

