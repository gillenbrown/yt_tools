import numpy as np
from scipy import optimize
from scipy import interpolate
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

def error_transform(log_X, log_X_err):
    """
    Transform errors in log(X) into an asymmetric error interval in X.

    :param log_M: The best fit value of log_10(X)
    :param log_M_err: The error on the best fit value of log_10(X)
    :return: The asymmetric error interval for the best fit on X. This is a
             tuple with lower, then upper error ranges.
    """
    # error must be positive
    if log_X_err < 0:
        raise ValueError("Errors must be positive.")

    lower_error = 10**log_X - 10**(log_X - log_X_err)
    upper_error = 10**(log_X + log_X_err) - 10**log_X
    return lower_error, upper_error

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
        params, cov =  optimize.curve_fit(log_plummer_disk, self.radii_logsafe,
                                          self.log_densities,
                                          bounds=self.bounds, p0=self.guess)
        self.cov = cov
        M_c_log, M_d_log, self.a_c, self.a_d = params
        errors = np.sqrt(np.diag(cov))
        M_c_log_err, M_d_log_err, self.a_c_err, self.a_d_err = errors

        # put the masses back into log space
        self.M_c = 10 ** M_c_log
        self.M_c_err = error_transform(M_c_log, M_c_log_err)

        self.M_d = 10 ** M_d_log
        self.M_d_err = error_transform(M_d_log, M_d_log_err)

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
            self.M_c_err_parametric = None
            self.M_d_err_parametric = None
            self.a_c_err_parametric = None
            self.a_d_err_parametric = None
            self.r_half_parametric = None
            self.nsc_radius_and_errors()
            self.cluster_mass_and_errors()
            self.half_mass_radii_and_errors()
            return

        # that does error checking too
        self.radii = radii
        self.densities = densities

        # since the binning in the center is too big to do provide totally
        # accurate answers here, we will interpolate between them. I do this
        # rather than to just pass in all the values to reduce noise. The
        # binning averages lots of things together, so the interpolation will
        # be a less noisy operation than using all points
        self.dens_interp = interpolate.interp1d(self.radii, self.densities,
                                                kind="linear",
                                                bounds_error=False,
                                                fill_value=(min(self.radii),
                                                            max(self.radii)))

        # we can then assign these parametric params straight out of here
        self.M_c_parametric = self.fitting.M_c
        self.M_d_parametric = self.fitting.M_d
        self.a_c_parametric = self.fitting.a_c
        self.a_d_parametric = self.fitting.a_d
        self.M_c_err_parametric = self.fitting.M_c_err
        self.M_d_err_parametric = self.fitting.M_d_err
        self.a_c_err_parametric = self.fitting.a_c_err
        self.a_d_err_parametric = self.fitting.a_d_err
        self.r_half_parametric = profiles.plummer_2d_half_mass(self.fitting.a_c)

        # then we can use that to find the equality radius
        self.nsc_radius_and_errors()
        # and the non parametric cluster mass
        self.cluster_mass_and_errors()
        # and the non parametric half mass radii
        self.half_mass_radii_and_errors()

    def _find_equality_radius(self, M_c, M_d, a_c, a_d):
        """Finds the radius at which the disk and cluster components have 
        equal density. This radius will be defined as the radius of the NSC 
        """
        # the algorithm for this is pretty simple. We assume the cluster
        # dominates the density at the center. Then we walk outwards at return
        # the radius at which the disk first becomes greater. If we have a fine
        # enough step this will be the same as the radius at which they are
        # equal-ish.
        for r in np.arange(0, max(self.radii), 0.1):
            cluster = profiles.plummer_2d(r, M_c, a_c)
            disk = profiles.exp_disk(r, M_d, a_d)

            if disk > cluster:
                # here is the place where the disk takes over, so we define it
                # to be the bound of the NSC, except if this happens at zero
                # radius, which happens when the disk always dominates.
                if r == 0:
                    return None
                else:
                    return r
        # if we got here we didn't find a radius.
        return None

    def _cluster_mass(self, nsc_radius):
        """Calculates a non-parametric mass for the cluster by integrating the
        density profile out to the radius of equaltiy.
        
        This assumes we calculated everything with a two dimensional profile, 
        which is safe becuase that's what we did above.

        :param nsc_radius: radius of the NSC. To get the best fit value, pass
                           in self.nsc_radius. Since we use this to determine
                           errors on the cluster mass, I want this to be a
                           parameter.
        """
        # first do error checking
        if nsc_radius is None:
            return None

        # first have to get the densities that are in the cluster
        rad_dens_pairs = [(r, rho) for r, rho in zip(self.radii, self.densities)
                          if r < nsc_radius]
        # then append the one that is right on the nsc radius
        if np.isfinite(nsc_radius):
            rad_dens_pairs.append((nsc_radius, self.dens_interp(nsc_radius)))
        # if there are no radii less than that, know the mass well
        if len(rad_dens_pairs) == 0:
            return 0

        # We then have to turn those back into two lists
        radii, densities = [], []
        for r, d in rad_dens_pairs:
            radii.append(r)
            densities.append(d)

        # then we can integrate
        integrand_values = [d * 2 * np.pi * r for r, d in zip(radii, densities)]
        return integrate.simps(integrand_values, radii, even="first")
        # I used first there because it uses the trapezoidal rule on the last
        # interval, which is less important than the first interval, since
        # there is more mass in this first interval.

    def _half_mass(self, cluster_mass):
        """Calculate the half mass radius for the cluster non-parametrically.
        
        This assumes a 2 dimensional integrand, which is safe because that's 
        what we did above. 
        
        This is done by gradually increasing the bounds of integration for the
        density profile (like what was done for the cluster mass), and finding
        the radius that just barely gives us half of the mass (similar to what
        was done for the equality radius).

        :param cluster_mass: Mass of the cluster. To find the best fit value
                             for this, pass in self.M_c_non_parametric
        """

        if cluster_mass is None:
            return None

        cumulative_integral = 0
        # iterate through the values. The radii were created by binning a bunch
        # of KDE values, so they are already in the center of a bin. Therefore
        # I can just use these values as my radii for integration, and we will
        # be using the center of each bin, which is the best.
        bin_width = 0.01
        for radius in np.arange(min(self.radii), max(self.radii), bin_width):
            integrand_here = self.dens_interp(radius) * 2 * np.pi * radius
            # do the integration
            cumulative_integral += integrand_here * bin_width
            # then see if it is slightly larger than half.
            if cumulative_integral > (cluster_mass / 2.0):
                return radius

    def nsc_radius_and_errors(self):
        """
        Find the NSC radius and it's error.

        The best fit value finds the radius at which the cluster and disk
        components are equal.

        Error is done by moving the cluster and disk components both up down
        by one sigma in mass. Then the NSC radius is determined on each of those
        perturbations, and the one sigma error is determined by the range of
        these perturbed NSC radii. The error interval is asymmetric.

        """
        # do a check for a None value
        if self.M_c_parametric is None:
            self.nsc_radius_err = None
            return

        # get the parameters to make my life easier
        M_c = self.M_c_parametric
        M_d = self.M_d_parametric
        a_c = self.a_c_parametric
        a_d = self.a_d_parametric
        M_c_dn, M_c_up = self.M_c_err_parametric
        M_d_dn, M_d_up = self.M_d_err_parametric

        # get the NSC radius for the best fit values
        self.nsc_radius = self._find_equality_radius(M_c, M_d, a_c, a_d)
        # do a check for a None value
        if self.nsc_radius is None:
            self.nsc_radius_err = None
            return

        # Find the NSC radius when we perturb the best fit by one sigma in all
        # possible combinations.
        nsc_1 = self._find_equality_radius(M_c + M_c_up, M_d + M_d_up, a_c, a_d)
        nsc_2 = self._find_equality_radius(M_c - M_c_dn, M_d + M_d_up, a_c, a_d)
        nsc_3 = self._find_equality_radius(M_c + M_c_up, M_d - M_d_dn, a_c, a_d)
        nsc_4 = self._find_equality_radius(M_c - M_c_dn, M_d - M_d_dn, a_c, a_d)

        # then get the maximum and minimum NSC radii of those
        nsc_radii = [nsc_1, nsc_2, nsc_3, nsc_4]
        min_nsc_rad = min(nsc_radii)
        max_nsc_rad = max(nsc_radii)

        # a None value indicates that there is no NSC
        if min_nsc_rad is None:
            min_nsc_rad = 0
        if max_nsc_rad is None:
            max_nsc_rad = np.inf

        # then the error is just the distance from the best fit value
        self.nsc_radius_err = (self.nsc_radius - min_nsc_rad,
                               max_nsc_rad - self.nsc_radius)


    def cluster_mass_and_errors(self):
        """
        Find the non parametric cluster mass and it's errors.

        The best fit cluster mass is found by integrating the density
        distribution inside the best fit NSC radius. The errors on that
        are found by doing the same with the radii corresponding to the
        one sigma errors on the NSC radius.
        """
        # first check that we have an NSC
        if self.nsc_radius is None:
            self.M_c_non_parametric = None
            self.M_c_non_parametric_err = None
            return

        # get the best fit value
        self.M_c_non_parametric = self._cluster_mass(self.nsc_radius)

        # then perturb by one sigme errors
        M_c_up = self._cluster_mass(self.nsc_radius + self.nsc_radius_err[1])
        M_c_down = self._cluster_mass(self.nsc_radius - self.nsc_radius_err[0])
        # then errors are the differences between that and the best fit value
        self.M_c_non_parametric_err = (self.M_c_non_parametric - M_c_down,
                                       M_c_up - self.M_c_non_parametric)

    def half_mass_radii_and_errors(self):
        """
        Like the other two functions, this calculates the best fit and errors
        for the half mass radii. This is done using a similar procedure as the
        other two, and uses the _half_mass function to do the heavy lifting.

        """
        # first check that we have an NSC
        if self.nsc_radius is None:
            self.r_half_non_parametric = None
            self.r_half_non_parametric_err = None
            return

        self.r_half_non_parametric = self._half_mass(self.M_c_non_parametric)

        # then perturb
        r_up = self._half_mass(self.M_c_non_parametric +
                               self.M_c_non_parametric_err[1])
        r_down = self._half_mass(self.M_c_non_parametric -
                                 self.M_c_non_parametric_err[0])

        # then turn that into an error
        self.r_half_non_parametric_err = (self.r_half_non_parametric - r_down,
                                          r_up - self.r_half_non_parametric)


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