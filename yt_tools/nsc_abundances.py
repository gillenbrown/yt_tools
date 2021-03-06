import os

import numpy as np
from scipy.misc import derivative

import yields
from . import utils

class NSC_Abundances(object):
    """Holds infomation about the abundances of an object. """
    # get some of the solar information
    z_sun, solar_metal_fractions = yields.create_solar_metal_fractions()

    def __init__(self, masses, Z_Ia, Z_II, mZZ_II, m_i,
                 II_type="nomoto"):
        """Create an abundance object.

        :param masses: The masses of the star partices.
        :type masses: list, np.ndarray
        :param Z_Ia: Values for the metallicity from Type II supernovae of the
                     star particles. Must have the same length as `masses`.
        :type Z_Ia: list, np.ndarray
        :param Z_II: Values for the metallicity from Type II supernovae of the
                     star particles. Must have the same length as `masses`.
        :type Z_II: list, np.ndarray
        :param II_type: Which model of Type Ia supernovae to use. Either
                        "nomtoto" or "ww".
        :type II_type: str
        :returns: None, but sets attributes.
        """
        # convert to numpy arrays if needed
        if not isinstance(masses, np.ndarray):
            masses = np.array(masses)
        if not isinstance(Z_Ia, np.ndarray):
            Z_Ia = np.array(Z_Ia)
        if not isinstance(Z_II, np.ndarray):
            Z_II = np.array(Z_II)
        if not isinstance(mZZ_II, np.ndarray):
            mZZ_II = np.array(mZZ_II)
        if not isinstance(m_i, np.ndarray):
            m_i = np.array(m_i)

        # do error checking.
        # masses must be positive.
        if any(masses <= 0) or any(m_i <= 0):
            raise ValueError("Masses must all be positive. ")
        # all arrays must be the same length
        if not len(masses) == len(Z_Ia) == len(Z_II) == len(mZZ_II) == len(m_i):
            raise ValueError("All arrays must be the same length. ")
        # the metallicity must be between 0 and 1.
        for z_type in [Z_Ia, Z_II]:
            if any(z_type < 0) or any(z_type > 1):
                raise ValueError("Metallicity must be between 0 and 1.")
        # spreads must all be positive
        if any(mZZ_II < 0):
            raise ValueError("Metallicity spreads must be non-negative.")

        # assign instance attributes
        self.mass = masses
        self.initial_masses = m_i
        self.Z_Ia = Z_Ia
        self.Z_II = Z_II
        self.mZZ_II = mZZ_II
        self.Z_tot = self.Z_Ia + self.Z_II
        self.one_minus_Z_tot = 1.0 - self.Z_tot

        # also have to check that the total metallicity isn't larger than one.
        if any(self.Z_tot < 0) or any(self.Z_tot > 1):
            raise ValueError("Total metallicity can't be larger than one. ")

        # create the yield objects that will be used to calculate the SN yields
        self.yields_Ia = yields.Yields("iwamoto_99_Ia_W7")
        if II_type == "nomoto":
            self.yields_II = yields.Yields("nomoto_06_II_imf_ave")
        elif II_type == "ww":
            self.yields_II = yields.Yields("ww_95_imf_ave")

        # also create the yield abundance object that is used to calculate
        # things for each star particle.
        self.abund = yields.Abundances()

        # Calculate a few simple things that will be useful later
        var_z_temp = self.mZZ_II / self.initial_masses - self.Z_II ** 2
        # then throw away negative values
        self.var_z_II_int_ind = np.clip(var_z_temp, a_min=0, a_max=None)

        # stuff for the whole cluster
        self.mean_Z_Ia = utils.weighted_mean(self.Z_Ia,
                                              weights=self.mass)
        self.mean_Z_II = utils.weighted_mean(self.Z_II,
                                              weights=self.mass)
        self.mean_Z_tot = utils.weighted_mean(self.Z_tot,
                                              weights=self.mass)

        self.var_z_II_int_tot = utils.weighted_mean(self.var_z_II_int_ind,
                                                    weights=self.mass)
        self.var_z_group = utils.weighted_variance(self.Z_tot,
                                                   weights=self.mass,
                                                   ddof=0)
        self.var_z_tot = self.var_z_II_int_tot + self.var_z_group

        # # then transform to log_Z
        # numerator = np.log(1 + self.sigma_squared_z / self.Z_II ** 2)
        # denominator = (np.log(10)) ** 2
        # self.sigma_squared_log_z = numerator / denominator

    def weighted_hydrogen(self):
        """Calculate the fraction of mass in H for. This assumes a solar
        ratio of Helium to Hydrogen, and we use the following math:

        X + Y + Z = 1
        X (1 + Y/X) = 1 - Z
        X = (1 - Z)/(1 + Y/X)"""
        fracs = np.array([self.abund.hydrogen(Z) for Z in self.Z_tot])
        return np.sum(fracs * self.mass)

    def z_on_h_total(self):
        """Calculate the Z on H value for this collection of stars, by
        dividing the total Z by the total H.

        This is calculated in the following way. The derivation for this is
        in my notebook, but here is the important equation.

        .. math::
            [Z/H] = \log_{10} \left[ \frac{\sum_\star M_\star Z_{tot \star}}
            {\sum_\star M_\star (1 - Z_{tot \star})}
            \frac{1 - Z_\odot}{Z_\odot} \right]

        This is basically the sum of metals divided by sum of not metals for
        both the sun and the stars. Not metals is a proxy for Hydrogen, since
        we assume cosmic abundances for both (not quite right, but not too bad).

        :returns: [Z/H] value for this collection of stars
        :rtype: float
        """
        star_num = np.sum(self.mass * self.Z_tot)
        star_denom = self.weighted_hydrogen()
        star_frac = star_num / star_denom

        sun_frac = self.z_sun / self.abund.hydrogen(self.abund.Z_sun)

        return np.log10(star_frac / sun_frac)

    def x_on_h_total(self, element):
        """Calculate the [X/H] value for this collection of stars.

        This is calculated in the following way.

        .. math::
            [X/H] = \log_{10} \left[ \frac{\sum_\star M_\star (Z_\star^{Ia}
            f_X^{Ia} + Z_\star^{II} f_X^{II})}{\sum_\star M_\star
            (1 - Z_{tot \star})}\frac{1 - Z_\odot}{Z_\odot f_{X \odot}} \right]

        Where f is the fraction of the total metals element x takes up for
        either the type Ia or II yields.

        This calculation is basically the sum of the mass in that metal divided
        by the mass of not metals for both the sun and the star. This works
        because we assume a cosmic abundance for H, making the mass that isn't
        in metals a proxy for H.

        :param element: Element to be used in place of X.
        :type element: str
        :returns: Value of [X/H] for the given element.
        :rtype: float
        """
        # get the metal mass fractions
        f_Ia = self.yields_Ia.mass_fraction(element, self.Z_Ia)
        f_II = self.yields_II.mass_fraction(element, self.Z_II)
        star_num = np.sum(self.mass * (self.Z_Ia * f_Ia + self.Z_II * f_II))
        star_denom = self.weighted_hydrogen()
        star_frac = star_num / star_denom

        sun_num = self.z_sun * self.solar_metal_fractions[element]
        sun_denom = self.abund.hydrogen(self.abund.Z_sun)
        sun_frac = sun_num / sun_denom

        return np.log10(star_frac / sun_frac)

    def x_on_fe_total(self, element):
        """Calculate the [X/Fe] value for this collection of stars.

        This is calculated in the following way.

        .. math::
            [X/Fe] = \log_{10} \left[ \frac{\sum_\star M_\star
            (Z_\star^{Ia}f_X^{Ia} + Z_\star^{II} f_X^{II})}{\sum_\star M_\star
            (Z_\star^{Ia}f_{Fe}^{Ia} + Z_\star^{II} f_{Fe}^{II})}
            \frac{f_{Fe \odot}}{f_{X \odot}} \right]

        Where f is the fraction of the total metals element x takes up for
        either the type Ia or II yields.

        This calculation is basically the sum of the mass in that metal divided
        by the mass of iron for both the sun and the star.

        :param element: Element to be used in place of X.
        :type element: str
        :returns: Value of [X/Fe] for the given element.
        :rtype: float
        """
        # get the metal mass fractions
        f_Ia_x = self.yields_Ia.mass_fraction(element, self.Z_Ia)
        f_II_x = self.yields_II.mass_fraction(element, self.Z_II)
        f_Ia_Fe = self.yields_Ia.mass_fraction("Fe", self.Z_Ia)
        f_II_Fe = self.yields_II.mass_fraction("Fe", self.Z_II)

        star_num = np.sum(self.mass * (self.Z_Ia * f_Ia_x + self.Z_II * f_II_x))
        star_denom = np.sum(self.mass * (self.Z_Ia * f_Ia_Fe +
                                         self.Z_II * f_II_Fe))
        star_frac = star_num / star_denom

        sun_num = self.solar_metal_fractions["Fe"]
        sun_denom = self.solar_metal_fractions[element]
        sun_frac = sun_num / sun_denom

        return np.log10(star_frac * sun_frac)

    def log_z_over_z_sun_total(self):
        """Returns the value of log(Z/Z_sun).

        This is a pretty straightforward calculation. We just take the total
        mass in metals and divide by the total stellar mass to get the
        overall metallicity of the star particles, then divide that by the
        solar metallicity.

        :returns: value of log(Z/Z_sun)
        :rtype: float
        """

        total_metals = np.sum(self.Z_tot * self.mass)
        total_mass = np.sum(self.mass)
        metallicity = total_metals / total_mass
        return np.log10(metallicity / self.z_sun)

    def x_on_h_individual(self, element):
        star_x_on_h = self.abund.x_on_h(element, self.Z_Ia, self.Z_II)
        return utils.to_array(star_x_on_h), self.mass

    def x_on_fe_individual(self, element):
        star_x_on_fe = self.abund.x_on_fe(element, self.Z_Ia, self.Z_II)
        return utils.to_array(star_x_on_fe), self.mass

    def _get_z_err(self, form):
        if form == "internal":
            z_variance = self.var_z_II_int_tot
        elif form == "group":
            z_variance = self.var_z_group
        elif form == "total":
            z_variance = self.var_z_tot
        else:
            raise ValueError("Wrong form.")

        return np.sqrt(z_variance)

    def _get_z_edges(self, form, z_type):
        z_err = self._get_z_err(form)
        if z_type == "total":
            z_mean = self.mean_Z_tot
        elif z_type == "II":
            z_mean = self.mean_Z_II
        else:
            raise ValueError("Wrong z type.")

        z_up = z_mean + z_err
        z_down = max(0, z_mean - z_err)

        return z_down, z_up

    def log_z_err(self, form):
        z_down, z_up = self._get_z_edges(form, z_type="total")
        mean_logz = self.log_z_over_z_sun_total()
        up = np.log10(z_up / self.z_sun)
        down = np.log10(z_down / self.z_sun)
        return (mean_logz - down,
                up - mean_logz)

    def z_on_h_err(self, form):
        z_down, z_up = self._get_z_edges(form, z_type="II")
        mean_zh = self.z_on_h_total()
        up = self.abund.z_on_h(self.mean_Z_Ia, z_up)
        down = self.abund.z_on_h(self.mean_Z_Ia, z_down)

        return (mean_zh - down,
                up - mean_zh)

    def abund_err(self, element, over, form):
        z_err = self._get_z_err(form)
        slope = self._elt_derivative(element, over, self.mean_Z_II)
        return abs(slope * z_err)  # takes care of sqrt of squared terms

    def abund_err_individual(self, element, over):
        z_var = self.var_z_II_int_ind
        slopes = np.array([self._elt_derivative(element, over, z_II)
                           for z_II in self.Z_II])
        return np.sqrt(z_var * slopes**2)

    # def x_on_h_err(self, element, form):
    #     z_down, z_up = self._get_z_edges(form, z_type="II")
    #     mean_xh = self.x_on_h_total(element)
    #     up = self.abund.x_on_h(element, self.mean_Z_Ia, z_up)
    #     down = self.abund.x_on_h(element, self.mean_Z_Ia, z_down)
    #
    #     return (mean_xh - down,
    #             up - mean_xh)
    #
    # def x_on_fe_err(self, element, form):
    #     z_down, z_up = self._get_z_edges(form, z_type="II")
    #     mean_xfe = self.x_on_fe_total(element)
    #     up = self.abund.x_on_fe(element, self.mean_Z_Ia, z_up)
    #     down = self.abund.x_on_fe(element, self.mean_Z_Ia, z_down)
    #
    #     return (mean_xfe - down,
    #             up - mean_xfe)
    #
    # def x_on_fe_err_individual(self, element):
    #     """Only internal dispersion can be considered here, since this is
    #     on a star by star basis."""
    #     z_errs = np.sqrt(self.var_z_II_int_ind)
    #     # the mean values are equivalent to what we do in x_on_fe_individual
    #     mean_values = self.abund.x_on_fe(element, self.Z_Ia, self.Z_II)
    #
    #     z_up = self.Z_II + z_errs
    #     z_down = np.clip(self.Z_II - z_errs, a_min=0, a_max=1)
    #
    #     up = self.abund.x_on_fe(element, self.Z_Ia, z_up)
    #     down = self.abund.x_on_fe(element, self.Z_Ia, z_down)
    #
    #     up_diff = up - mean_values
    #     down_diff = mean_values - down
    #
    #     return utils.to_array(np.mean([up_diff, down_diff], axis=0))

    # def _x_on_h_log_derivative(self, element):
    #     """
    #     Calculates the derivative of [X/H] against log(Z_II), which is what is
    #     needed to calculate the internal dispersion of the clusters.
    #
    #     :return: Value of d[X/H] / d log_Z_II for all the metallicities of the
    #              star particles in the cluster.
    #     """
    #     def x_on_h_wrapper(log_z_II):
    #         return self.abund.x_on_h(element, 0, 10**log_z_II)
    #
    #     slopes = []
    #     for log_z in np.log10(self.Z_II):
    #         slopes.append(derivative(x_on_h_wrapper, log_z, dx=0.01))
    #
    #     return np.array(slopes)

    # def _x_on_fe_log_derivative(self, element):
    #     """
    #     Calculates the derivative of [X/Fe] against log(Z_II), which is what is
    #     needed to calculate the internal dispersion of the clusters.
    #
    #     :return: Value of d[X/Fe] / d log_Z_II for all the metallicities of the
    #              star particles in the cluster.
    #     """
    #     def x_on_fe_wrapper(log_z_II):
    #         return self.abund.x_on_fe(element, 0, 10**log_z_II)
    #
    #     slopes = []
    #     for log_z in np.log10(self.Z_II):
    #         slopes.append(derivative(x_on_fe_wrapper, log_z, dx=0.01))
    #
    #     return np.array(slopes)

    def _elt_derivative(self, element, over, Z_II):
        """
        Calculates the derivative of [X/H] against Z_II, which is what is
        needed to calculate the internal dispersion of the clusters.

        :return: Value of d[X/H] / d log_Z_II for all the metallicities of the
                 star particles in the cluster.
        """
        if over == "Fe":
            def wrapper(Z_II):
                return self.abund.x_on_fe(element, 0, Z_II)
        elif over == "H":
            def wrapper(Z_II):
                return self.abund.x_on_h(element, 0, Z_II)
        else:
            raise ValueError("over must be either 'Fe' or 'H'")

        return derivative(wrapper, Z_II, dx=Z_II/10.0)
    #
    # def internal_variance_individual_derivative(self, element, over):
    #     """Variance in elemental abundances that come from internal dispersion.
    #
    #     This is calculated in my notebook, and the equation will be in the
    #     paper.
    #
    #     :return: List of the values of the variances in an elemental abundance
    #              due to internal dispersion.
    #     """
    #     slopes = [self._elt_derivative(element, over, z) for z in self.Z_II]
    #     return self.var_z_II_ind * np.array(slopes)**2
    #
    # def internal_variance_elt(self, element, over):
    #     """Total variance that comes from the spread among [Fe/H]
    #     within particles.
    #
    #     This is calculated in my notebook, and the equation will be in the
    #     paper.
    #
    #     :return: Value of the variance in [Fe/H] contributed by the internal
    #              dispersion within star particles.
    #     """
    #     individual_var = self.internal_variance_individual_derivative(element, over)
    #     return utils.weighted_mean(individual_var, weights=self.mass)
    #
    # def x_on_h_err_new_internal(self, element):
    #     z_err = np.sqrt(self.var_z_II_int_tot)
    #     # slope = self._x_on_h_derivative(element, self.mean_Z_II)
    #     # return slope * z_err
    #     mean_x_on_h = self.x_on_h_total(element)
    #     up = self.abund.x_on_h(element, self.mean_Z_Ia, self.mean_Z_II + z_err)
    #     down = self.abund.x_on_h(element, self.mean_Z_Ia, max(self.mean_Z_II - z_err, 0))
    #
    #     return (mean_x_on_h - down,
    #             up - mean_x_on_h)
    #
    # def x_on_h_err_new_group(self, element):
    #     z_err = np.sqrt(self.var_z_group)
    #     # slope = self._x_on_h_derivative(element, self.mean_Z_II)
    #     # return slope * z_err
    #     mean_x_on_h = self.x_on_h_total(element)
    #     up = self.abund.x_on_h(element, self.mean_Z_Ia, self.mean_Z_II + z_err)
    #     down = self.abund.x_on_h(element, self.mean_Z_Ia, max(self.mean_Z_II - z_err, 0))
    #
    #     return (mean_x_on_h - down,
    #             up - mean_x_on_h)
    #
    # def x_on_h_err_new_total(self, element):
    #     z_err = np.sqrt(self.var_z_tot)
    #     slope = self._x_on_h_derivative(element, self.mean_Z_II)
    #     return slope * z_err
    #     # up = self.abund.x_on_h(element, self.mean_Z_Ia, self.mean_Z_II + z_err)
    #     # down = self.abund.x_on_h(element, self.mean_Z_Ia, max(self.mean_Z_II - z_err, 0))
    #     #
    #     # return (mean_x_on_h - down,
    #     #         up - mean_x_on_h)
