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

        # convert to numpy arrays
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)
        if not isinstance(densities, np.ndarray):
            densities = np.array(densities)

        # test that they are the same
        if len(radii) != len(densities):
            raise ValueError("Radiii and densities need to be the same length.")

        self.radii = radii
        self.densities = densities

        self._create_log_densities()

    def fit(self):
        """Performs the fit and stores the parameters."""
        params, errs =  optimize.curve_fit(log_plummer_disk, self.radii_logsafe,
                                           self.log_densities,
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

    def _create_log_densities(self):
        # we want to do the fitting in log space, but need to be careful of
        # places where there are zeros, and remove those
        log_densities = np.log10(self.densities)
        # see where it is not infinite
        good_idx = np.isfinite(log_densities)

        # then save the values at those radii
        self.radii_logsafe = self.radii[good_idx]
        self.log_densities = log_densities[good_idx]



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
        try:
            self.fitting.fit()
        except RuntimeError:  # number of iterations was exceeded, fit not found
            self.radii = radii
            self.densities = densities
            self.nsc_radius = None
            self.M_c_parametric = None
            self.M_d_parametric = None
            self.a_c_parametric = None
            self.a_d_parametric = None
            self.r_half_parametric = None
            self._cluster_mass()
            self._half_mass()

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
        # then assign the values
        self.M_c_non_parametric = mass

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


class AxisRatios(object):
    """Calculates the axes ratios of the object using the "intertia tensor."
    
    This is in quotes because it's not really the interia tensor. See Zemp et 
    al 2011. Equation 3 shows what the tensor I'm using is, and equations 5 and
    8 show exactly how I'm doing it. Equation 8 is the calculation I use. The 
    text after that tells how this isn't really the inertia tensor. In the next
    section it talks about how this can give axis rations. We just get the 
    eigenvalues of that tensor and take ratios. """
    def __init__(self, x, y, z, mass):
        """
        Initialize the object. This takes the coordinated and mass, which is all
        that is needed to calculate the axis ratios.
        
        :param x: List of x values for the positions of the stars. 
        :param y: List of x values for the positions of the stars. 
        :param z: List of x values for the positions of the stars. 
        :param mass: List of star masses.
        """
        # error checking
        if not (len(x) == len(y) == len(z) == len(mass)):
            raise ValueError("All parameters need to be the same length.")
        # if it passes set the values
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass

        # create the inertia tensor
        self._create_inertia_tensor()
        # and then get the eigenvalues
        self._create_axes_ratios()

    @staticmethod
    def _inertia_tensor_component(loc_a, loc_b, mass):
        """Calculates a component of the "inertia tensor", as takes from 
        Equation 8 in Zemp et al 2011. """
        numerator = np.sum(loc_a * loc_b * mass)
        denominator = np.sum(mass)
        return numerator / denominator

    def _create_inertia_tensor(self):
        """Calculate the "inertia tensor". This is calculated from equation 8
        in Zemp et al 2011. """
        # set up the initial matrix to be filled
        matrix = np.zeros([3, 3])
        # then put the coords in a list for easier access when filling array
        star_coords = [self.x, self.y, self.z]

        # then calculate that for each value.
        for i in range(3):
            for j in range(3):
                matrix[i][j] = self._inertia_tensor_component(star_coords[i],
                                                              star_coords[j],
                                                              self.mass)

        self.inertia_tensor = np.matrix(matrix)

    def _create_axes_ratios(self):
        """Does the work of getting the axis rations. These are simple the 
        ratios of the equare toors of the eigenvalues. (see discussion after 
        equation 10 in Zemp et al 2011). """
        eigenvalues = np.linalg.eigvals(self.inertia_tensor)
        c, b, a = sorted(eigenvalues) # sort goes from small to big

        # take square root of all values
        a = np.sqrt(a)
        b = np.sqrt(b)
        c = np.sqrt(c)

        # then turn them into axis ratios.
        self.a_over_b = a / b
        self.b_over_a = b / a
        self.a_over_c = a / c
        self.c_over_a = c / a
        self.b_over_c = b / c
        self.c_over_b = c / b

        self.ellipticity = 1.0 - self.c_over_a