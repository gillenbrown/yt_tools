import numpy as np
from scipy import optimize
from . import profiles
from . import utils

def log_plummer_disk(r, M_c, M_d, a_c, a_d):
    """Returns the log of the sum of an exponential disk and Plummer sphere.
    
    Log values of the masses are used to make it easier on the fitting routine.
    We return the log for a similar reason. The fitting is easier in log space.
    
    :param r: radii
    :param M_c: log base 10 of the mass of the cluster component. 
    :param M_d: log base 10 of the mass of the disk component.
    :param a_c: scale radius of the cluster component. 
    :param a_d: scale radius of the disk component.
    :returns: stellar mass densities at the radii desired.
    """
    cluster_component = profiles.plummer_2d(r, 10**M_c, a_c)
    disk_component = profiles.exp_disk(r, 10**M_d, a_d)
    total = cluster_component + disk_component
    return np.log10(total)

class Fitting(object):
    """Performs the fitting to determine where the NSC is. """
    # set the bounds for the fitting and the initial guess
    bounds = [[0.01, 0.01, 0.01, 0.01], [15, 15, 500, 5000]]
    guess = [6, 6, 400, 10]

    def __init__(self, radii, densities):
        """Create the Fitting object
        
        :param radii: list of radii at which the densities were calculated.
        :param densities: list of densities, corresponding to the radii.
        
        """
        # test that both are iterable
        utils.test_iterable(radii, "radii")
        utils.test_iterable(densities, "densities")
        # test that they are the same
        if len(radii) != len(densities):
            raise ValueError("Radiii and densities need to be the same length.")

        self.radii = radii
        self.densities = densities

    def fit(self):
        """Performs the fit and stores the parameters."""
        params, errs =  optimize.curve_fit(log_plummer_disk, self.radii,
                                           np.log10(self.densities),
                                           bounds=self.bounds, p0=self.guess)
        self.M_c, self.M_d, self.a_c, self.a_d = params

        # if the masses are close to zero, set those
        if np.isclose(self.M_c, self.bounds[0][0], atol=0.02):
            self.M_c = 0
        if np.isclose(self.M_d, self.bounds[0][1], atol=0.02):
            self.M_d = 0

class NscStructure(object):
    """Handles all the structure calculations for the NSCs. 
    
    This includes the mass and half mass radii primarily, in both a parameteric 
    and non parametric way. """
    def __init__(self, radii, densities):
        """Create the structure object

        :param radii: list of radii at which the densities were calculated.
        :param densities: list of densities, corresponding to the radii.

        """

        # first we create the fitting object and use it to get the basic
        # structural parameters that will be used later
        self.fitting = Fitting(radii, densities)
        self.fitting.fit()

