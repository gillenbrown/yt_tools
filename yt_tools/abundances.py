import os

import numpy as np

import yields

# need to get the solar abundances
def create_solar_metal_fractions():
    this_file_loc = os.path.dirname(__file__)
    enclosing_dir = os.sep.join(this_file_loc.split(os.sep)[:-1])
    solar_file = enclosing_dir + '/data/solar_abundance.txt'
    solar = np.genfromtxt(solar_file, dtype=None,
                          names=['Natom', 'name', 'fN', 'log', 'f_mass'])
    z_mass = np.sum(solar["f_mass"][2:])

    metal_fractions = dict()
    for row in solar:
        elt = row["name"]
        f_mass = row["f_mass"]
        metal_fractions[elt] = f_mass / z_mass

    return z_mass, metal_fractions

class Abundances(object):
    """Holds infomation about the abundances of an object. """
    # get some of the solar information
    z_sun, solar_metal_fractions = create_solar_metal_fractions()
    def __init__(self, masses, Z_Ia, Z_II):
        """Create an abundance object.

        :param masses: The masses of the star partices.
        :type masses: np.ndarray
        :param Z_Ia: Values for the metallicity from Type II supernovae of the
                     star particles. Must have the same length as `masses`.
        :type Z_Ia: np.ndarray
        :param Z_II: Values for the metallicity from Type II supernovae of the
                     star particles. Must have the same length as `masses`.
        :type Z_II: np.ndarray
        :returns: None, but sets attributes.
        """
        # convert to numpy arrays if needed
        if not isinstance(masses, np.ndarray):
            masses = np.array(masses)
        if not isinstance(Z_Ia, np.ndarray):
            Z_Ia = np.array(Z_Ia)
        if not isinstance(Z_II, np.ndarray):
            Z_II = np.array(Z_II)

        # do error checking.
        # masses must be positive.
        if not all(masses > 0):
            raise ValueError("Masses must all be positive. ")
        # all arrays must be the same length
        if not len(masses) == len(Z_Ia) == len(Z_II):
            raise ValueError("All arrays must be the same length. ")
        # the metallicity must be between 0 and 1.
        for z_type in [Z_Ia, Z_II]:
            if any(z_type < 0) or any(z_type > 1):
                raise ValueError("Metallicity must be between 0 and 1.")

        # assign instance attributes
        self.mass = masses
        self.Z_Ia = Z_Ia
        self.Z_II = Z_II
        self.Z_tot = self.Z_Ia + self.Z_II
        self.one_minus_Z_tot = 1.0 - self.Z_tot

        # also have to check that the total metallicity isn't larger than one.
        if any(self.Z_tot < 0) or any(self.Z_tot > 1):
            raise ValueError("Total metallicity can't be larger than one. ")

        # create the yield objects that will be used to calculate the SN yields
        self.yields_Ia = yields.Yields("iwamoto_99_Ia_W7")
        self.yields_II = yields.Yields("nomoto_06_II_imf_ave")

    def z_on_h(self):
        """Calculate the Z on H value for this collection of stars.

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
        star_denom = np.sum(self.mass * self.one_minus_Z_tot)
        star_frac = star_num / star_denom
        sun_frac = (1.0 - self.z_sun) / self.z_sun

        return np.log10(star_frac * sun_frac)

    def x_on_h(self, element):
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
        star_denom = np.sum(self.mass * self.one_minus_Z_tot)
        star_frac = star_num / star_denom

        sun_num = 1.0 - self.z_sun
        sun_denom = self.z_sun * self.solar_metal_fractions[element]
        sun_frac = sun_num / sun_denom

        return np.log10(star_frac * sun_frac)

    def x_on_fe(self, element):
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
