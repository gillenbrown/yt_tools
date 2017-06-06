import yt
import numpy as np

from . import kde
from . import utils
from . import nsc_structure

class Galaxy(object):
    def __init__(self, dataset, center, radius, j_radius=None, disk_radius=None,
                 disk_height=None):
        """Create a galaxy object at the specified location with the 
        given size. 
        
        :param dataset: yt dataset object that contains this galaxy
        :param center: 3 element array with the center of the galaxy. Must have
                       units.
        :param radius: Radius that will be used to create the sphere object.
                       Must also have units.
        :params j_radius, disk_radius, disk_height: used for the creation of
                the disk. See the _add_disk functionality for detailed
                explanation. If these are left blank no disk will be
                created.
        """

        # Check that the dataset is actually a datset, then assign
        if not isinstance(dataset, yt.data_objects.static_output.Dataset):
            raise TypeError("Dataset must be a yt dataset.")
        self.ds = dataset

        # Error checking on center. It must be a three element array with units.
        utils.test_for_units(center, "center")
        if len(center) != 3:
            raise ValueError("Center must be a three element array (x, y, z).")
        # set the attribute if it passes tests
        self.center = center

        # we need to check that the radius has units too
        utils.test_for_units(radius, "radius")
        # set the attribute if it passes tests
        self.radius = radius

        # create the sphere that contains the galaxy.
        self.sphere = self.ds.sphere(center=self.center, radius=self.radius)

        # and find the smallest cell size (used for KDE)
        self.min_dx = np.min(self.sphere[('index', 'dx')])

        # we can then initialize the KDE object, which will be used for the
        # centering process as well as profiles
        self._star_kde_mass_3d = self._create_kde_object(3, "mass")
        self._star_kde_metals_3d = self._create_kde_object(3, "Z")
        # have cylindrical coords that aren't used yet
        self._star_kde_mass_2d = None
        self._star_kde_metals_2d = None

        # then there are several quantities we initialize to zero or blank, but
        # will be filled in future analyses
        self.radii = dict()  # used for radial profiles
        self.densities = dict()  # used for radial profiles
        self.disk = None  # used for cylindrical plots
        self.nsc = None  # used for NSC analysis
        self.nsc_radius = None  # used for NSC analysis
        self.nsc_idx = None  # used for NSC analysis

        # we can then add a disk if the user wants to, and initialize the rest
        # of everything that comes after that.
        if (j_radius is not None) or (disk_height is not None) or \
                (disk_radius is not None):
            self.add_disk(j_radius, disk_radius, disk_height)
            self.find_nsc_radius()
            self.create_axis_ratios()

    def _create_kde_object(self, dimension=2, quantity="mass"):
        """Creates a KDE object in the desired coordinates for the desired
        quantity.
        
        :param dimension: Whether to do this in cartesian (3D) or cylindrical 
                          (2D) coordinates. Cartesian requires no prep. 
                          Cylindrical requires the existence of a disk data 
                          container, which is created by the `add_disk` 
                          function. Under the hood, everything is converted 
                          to cartesian, however. 
        :param quantity: Which quantities to do this for. Currently all that is
                         supported are "mass" and "Z", which is total 
                         metallicity.
        """
        # error checking
        self._kde_profile_error_checking(quantity, dimension)

        # first get the locations of the stars. Depending on the coordinate
        # sytem used, we use a different data container, that we need to save
        # for later when getting the values
        if dimension == 3:
            container = self.sphere
            x = np.array(container[('STAR', 'POSITION_X')].in_units("pc"))
            y = np.array(container[('STAR', 'POSITION_Y')].in_units("pc"))
            z = np.array(container[('STAR', 'POSITION_Z')].in_units("pc"))

            # store these locations
            locations = (x, y, z)
        else:  # cylindrical
            container = self.disk
            r = container[('STAR', 'particle_position_cylindrical_radius')]
            r = np.array(r.in_units("pc"))
            theta = container[('STAR', 'particle_position_cylindrical_theta')]
            theta = np.array(theta)
            # convert to cartesian
            x, y = utils.convert_polar_to_cartesian(r, theta)
            # store these location
            locations = (x, y)

        # then get the quantity we care about. We need masses for both
        masses = np.array(container[('STAR', 'MASS')].in_units("msun"))
        if quantity == "mass":
            values = masses
        elif quantity == "Z":
            z_Ia = np.array(container[('STAR', 'METALLICITY_SNIa')])
            z_II = np.array(container[('STAR', 'METALLICITY_SNII')])
            total_z = z_Ia + z_II
            values = total_z * masses  # this is total metals

        # we can then create the KDE object
        return kde.KDE(locations, values)

    def add_disk(self, j_radius=None, disk_radius=None, disk_height=None):
        """Creates a disk aligned with the angular momentum of the galaxy.
        
        This works by creating a sphere with the radius=`j_radius`, calculating 
        the angular momentum vector within that sphere, then creating a disk
        who's normal vector is that angular momentum vector. This should give 
        us a disk that aligns with the actual disk of the galaxy.
        
        :param j_radius: radius within which we will calculate the angular 
                         momentum. Needs units.
        :param disk_radius: radius of the resulting disk. Needs units.
        :param disk_height: height of the resulting disk. Needs units.
        :returns: None, but creates the disk attribute for the galaxy. 
        """
        # set default values if not already set
        if j_radius is None:
            j_radius = self.radius  # not an ideal value, but a good default
        if disk_radius is None:
            disk_radius = self.radius
        if disk_height is None:
            disk_height = self.radius
        # then check units
        utils.test_for_units(j_radius, "j_radius")
        utils.test_for_units(disk_radius, "disk_radius")
        utils.test_for_units(disk_height, "disk_height")

        # then we can go ahead and do things. First create the new sphere
        j_vec_sphere = self.ds.sphere(center=self.center, radius=j_radius)
        # find its angular momentum
        j_vec = j_vec_sphere.quantities.angular_momentum_vector()
        # then create the disk
        self.disk = self.ds.disk(center=self.center, normal=j_vec,
                                 height=disk_height, radius=disk_radius)

        # we can then use this disk to create the 2D KDE objects
        self._star_kde_mass_2d = self._create_kde_object(2, "mass")
        self._star_kde_metals_2d = self._create_kde_object(2, "Z")

    def centering(self, accuracy=0.1):
        """Recenters the galaxy on the location of the maximum stellar density.
        
        :param accuracy: Precision with which the center will be found. More
                         specifically, the value reported will be within
                         "accuracy" of the true maximum stellar density. This 
                         value is in parsecs. The default value is pretty low, 
                         at 0.1 pc, but this is good, since we want good
                         precision so we can accurately calculate things like
                         the angular momentum. 
         
         This is done by using the KDE class. """

        # Our kernel size will be the size of the smallest cell in the sphere,
        # in parsecs. It needs to be in parsecs, since the x, y, and z values
        # are in parsecs (see this class's __init__).
        kernel_size = self.min_dx.in_units("pc").value

        # we can then go ahead with the centering, using the user's desired
        # accuracy
        self._star_kde_mass_3d.centering(kernel_size, accuracy)

        # then get the values
        cen_x = self._star_kde_mass_3d.location_max_x
        cen_y = self._star_kde_mass_3d.location_max_y
        cen_z = self._star_kde_mass_3d.location_max_z
        # those are scalars, so put them into a yt array. The KDE object is
        # the values when the units are in parsecs (see the __init__), so we
        # add those units back.
        self.center = [cen_x, cen_y, cen_z] * yt.units.pc

        # recreate the sphere that contains the galaxy.
        self.sphere = self.ds.sphere(center=self.center, radius=self.radius)

    def _kde_profile_error_checking(self, quantity, dimension):
        """Handles the error checking for functions that involve KDE profiles.
        
        We have to check that the quantity is something we know about, and that
        the coordinate system is valid. Also, if the user wants to do things
        in 2D, there needs to be an already existing disk. """
        # check that the coordinate system is okay
        if dimension not in [2, 3]:
            raise ValueError("`dimension` must be either 2 or 3.")
        # check that we are able to actually do the cylindrical profiles
        if dimension == 2 and self.disk is None:
            raise RuntimeError("Need to create a disk first. ")
        if quantity.lower() not in ["mass", "z"]:
            raise ValueError("Only mass and Z are supported. ")

    def stellar_mass(self, nsc=False):
        """Calculate the total stellar mass in this galaxy. 
        
        This sums the masses of the star particles that are inside the radius 
        of the galaxy or NSC, and returns the answer in solar masses.

        """

        masses = self.sphere[('STAR', "MASS")].in_units("msun")
        if nsc:
            self._check_nsc_existence()  # will raise error if no NSC
            return np.sum(masses[self.nsc_idx])

        else:  # whole galaxy
            return np.sum(masses)

    def kde_profile(self, quantity="MASS", dimension=2, spacing=None,
                    outer_radius=None):
        """
        Create a radial profile of a given quantity of the stars, typicall mass.

        :param quantity: the parameter that will be plotted. Has to be
                         associated with the stars, such that they are stored
                         in the form ("STAR", `quantity`). The default here is
                         "MASS".
        :param dimension: Coordinate system to use when calculating the profile. 
                          Must be either 2 or 3. If you are in 2D, the galaxy 
                          must have a disk object to use first, otherwise this 
                          will throw an error.
        :param spacing: The density will be calculated from zero to 
                        `outer_radius` with a spacing determined by this 
                        variable. The points will be equally spaced in radius,
                        while the angular coordinates will be random. This needs
                        to have a unit on it. If not provided, this defaults
                        to a spacing that will give 100 bins between 0 and 
                        `outer_radius`.
        :param outer_radius: Maximum radius out to which to calculate the radial
                             profile. Needs to have a unit on it. If not 
                             provided, it defaults to the radius of the galaxy.
        """
        if outer_radius is None:
            outer_radius = self.radius
        if spacing is None:
            spacing = outer_radius / 100.0
        # test that both outer_radius and spacing have units
        utils.test_for_units(outer_radius, "outer_radius")
        utils.test_for_units(spacing, "spacing")

        # do the KDE error checking
        self._kde_profile_error_checking(quantity, dimension)

        # All the KDE stuff is in parsecs, so convert those to parsecs
        outer_radius = outer_radius.in_units("pc").value
        spacing = spacing.in_units("pc").value
        center = self.center.in_units("pc").value

        # the KDE objects that we want to use depend on the coordinate system,
        # as does the center. In 3D we just use the center of the galaxy, since
        # it is in Cartesian coords, but in 2D we use 0, since the transform to
        # xy from cylindrical already subtracts off the center
        if dimension == 2:
            center = [0, 0]
            mass_kde = self._star_kde_mass_2d
            metals_kde = self._star_kde_metals_2d
        else:  # spherical: keep regular center
            mass_kde = self._star_kde_mass_3d
            metals_kde = self._star_kde_metals_3d

        # then create the radii
        radii = np.arange(0, outer_radius, spacing)

        # the smoothing kernel we will use in the KDE process is the size of
        # the smallest cell in the sphere
        kernel_size = self.min_dx.in_units("pc").value

        # we are then able to do the actual profiles
        if quantity.lower() == "mass":
            # use the mass KDE object that's already set up
            final_densities = mass_kde.radial_profile(kernel_size, radii,
                                                      center)
        elif quantity == "Z":
            # for metallicity we have to calculate the mass in metals, and
            # divide it by the mass in stars
            metal_densities = metals_kde.radial_profile(kernel_size, radii,
                                                        center)
            mass_densities = mass_kde.radial_profile(kernel_size, radii,
                                                     center)
            final_densities = metal_densities / mass_densities  # Z
        else:
            raise ValueError("{} is not implemented yet.".format(quantity))

        # then set the attributes. The key used to store the data is needed
        key = "{}_kde_{}D".format(quantity.lower(), dimension)
        self.radii[key] = radii
        self.densities[key] = final_densities

    def find_nsc_radius(self):
        """
        Finds the radius of the NSC, using the KDE profile and the associated
        functionality in the NscStructure class.

        :return:
        """

        # first need to to the KDE fitting procedure
        self.kde_profile("MASS", spacing=0.05 * yt.units.pc,
                         outer_radius=1000 * yt.units.pc)
        self.nsc = nsc_structure.NscStructure(self.radii["mass_kde_2D"],
                                              self.densities["mass_kde_2D"])

        self.nsc_radius = self.nsc.nsc_radius * yt.units.pc
        # then get the indices of the stars actually in the NSC
        radius_key = ('STAR', 'particle_position_spherical_radius')
        self.nsc_idx = np.where(self.sphere[radius_key] < self.nsc_radius)

    def create_axis_ratios(self):
        """Creates the axis ratios object. """
        self._check_nsc_existence()

        # get the locations of the stars in the NSC. Convert to numpy arrays
        # for speed.
        x = np.array(self.sphere[('STAR', 'POSITION_X')][self.nsc_idx])
        y = np.array(self.sphere[('STAR', 'POSITION_Y')][self.nsc_idx])
        z = np.array(self.sphere[('STAR', 'POSITION_Z')][self.nsc_idx])
        mass = np.array(self.sphere[('STAR', "MASS")][self.nsc_idx])

        self.nsc_axis_ratios = nsc_structure.AxisRatios(x, y, z, mass)

    def _check_nsc_existence(self):
        """Checks if we have a NSC radius set or not. Raises an AttributeError
        if not. """
        if self.nsc_radius is None:
            raise AttributeError("NSC radius has not been set.")



