import numpy as np
from scipy import optimize
from scipy import integrate

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
    guess = [6, 6, 10, 400]

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

        # if the masses are close to zero, put them exactly there. Otherwise,
        # put the values back into real space, not log space
        if np.isclose(self.M_c, self.bounds[0][0], atol=0.02):
            self.M_c = 0
        else:
            self.M_c = 10 ** self.M_c

        if np.isclose(self.M_d, self.bounds[0][1], atol=0.02):
            self.M_d = 0
        else:
            self.M_d = 10 ** self.M_d



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

        # that does error checking too
        self.radii = radii
        self.densities = densities

        # we can then assign these parametric params straight out of here
        self.M_c_parametric = self.fitting.M_c
        self.M_d_parametric = self.fitting.M_d
        self.a_c_parametric = self.fitting.a_c
        self.a_d_parametric = self.fitting.a_d
        self.r_half_parametric = profiles.plummer_2d_half_mass(self.fitting.a_c)

        # then we can use that to find the equality radius
        self._find_equality_radius()
        # and the non parametric cluster mass
        self._cluster_mass()
        # and the non parametric half mass radii
        self._half_mass()

    def _find_equality_radius(self):
        """Finds the radius at which the disk and cluster components have 
        equal density. This radius will be defined as the radius of the NSC 
        """
        # the algorithm for this is pretty simple. We assume the cluster
        # dominates the density at the center. Then we walk outwards at return
        # the radius at which the disk first becomes greater. If we have a fine
        # enough step this will be the same as the radius at which they are
        # equal-ish.
        for r in np.arange(0, max(self.radii), 0.1):
            cluster = profiles.plummer_2d(r, self.M_c_parametric,
                                          self.a_c_parametric)
            disk = profiles.exp_disk(r, self.M_d_parametric,
                                     self.a_d_parametric)

            if disk > cluster:
                # here is the place where the disk takes over, so we define it
                # to be the bound of the NSC, except if this happens at zero
                # radius, which happens when the disk always dominates.
                if r == 0:
                    self.nsc_radius = None
                else:
                    self.nsc_radius = r
                return
        # if we got here we didn't find a radius.
        self.nsc_radius = None

    def _cluster_mass(self):
        """Calculates a non-parametric mass for the cluster by integrating the
        density profile out to the radius of equaltiy.
        
        This assumes we calculated everything with a two dimensional profile, 
        which is safe becuase that's what we did above. 
        """
        # first do error checking
        if self.nsc_radius is None:
            self.M_c_non_parametric = None
            return

        # first have to get the densities that are in the cluster
        rad_dens_pairs = [(r, rho) for r, rho in zip(self.radii, self.densities)
                          if r < self.nsc_radius]

        # We then have to turn those back into two lists
        radii, densities = [], []
        for r, d in rad_dens_pairs:
            radii.append(r)
            densities.append(d)

        # then we can integrate
        integrand_values = [d * 2 * np.pi * r for r, d in zip(radii, densities)]
        mass = integrate.simps(integrand_values, radii)
        # then take a log.
        self.M_c_non_parametric = np.log10(mass)

    def _half_mass(self):
        """Calculate the half mass radius for the cluster non-parametrically.
        
        This assumes a 2 dimensional integrand, which is safe because that's 
        what we did above. 
        
        This is done by gradually increasing the bounds of integration for the
        density profile (like what was done for the cluster mass), and finding
        the radius that just barely gives us half of the mass (similar to what
        was done for the equality radius). """

        if self.nsc_radius is None:
            self.r_half_non_parametric = None
            return

        new_radii = []
        integrand_values = []
        # iterate through the values.
        for radius, density in zip(self.radii, self.densities):
            new_radii.append(radius)
            integrand_values.append(density * 2 * np.pi * radius)
            # can't integrate when there's nothing
            if len(integrand_values) <= 2:
                continue
            # do the integration
            this_mass = integrate.simps(integrand_values, new_radii)
            # then see if it is slightly larger than half.
            if this_mass > (self.M_c_non_parametric / 2.0):
                self.r_half_non_parametric = radius
                return


