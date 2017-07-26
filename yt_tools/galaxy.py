import yt
import numpy as np

from . import kde
from . import utils
from . import nsc_structure
from . import abundances


# define some helper functions for the read/write process
def _write_single_item(file_obj, item, name, units=False, multiple=False):
    """Writes a single item to the file. The file must already be opened and
    ready for writing.

    Will write a single row to the file of the format:
    "name: item"  # if a single value with no units
    "name: item units"  # if a single value with units
    "name: item[0]  item[1] units"  # if it has multiple parts and units

    :param file_obj: already opened file object that the data will be
                     written to.
    :param item: item to be written to the file. It can be a single item, or
                 something like a list or array too.
    :param name: Name to call the item in the output file.
    :param units: Whether or not the item has units on it or not.
    :type units: bool
    :param multiple: Whether the item has multiple parts that need to be written
                     separately. This would be true if it is a list or array.
    :type multiple: bool

    """
    file_obj.write("{}:\t".format(name))
    # check for a None type
    if item is None:
        file_obj.write("None\n")
        return

    # when there are multiple items we have to parse them individually
    if multiple:
        for i in item:
            if units:  # we don't want the units now
                file_obj.write(str(i.value) + "\t")
            else:
                file_obj.write(str(i) + "\t")
    else:  # not multiple. Same as above
        if units:
            file_obj.write(str(item.value) + "\t")
        else:
            file_obj.write(str(item) + "\t")
    # then write the units if we need to
    if units:
        file_obj.write(str(item.units))

    file_obj.write("\n")


def _parse_line(line, multiple=False, units=False, new_type=float):
    """Parses one line from the output of the galaxy.write() function.

    This does not handle lines that contain KDe data, that is handled
    separately.

    :param line: Line to parse.
    :param multiple: Whether or not this line contains an item with multiple
                     parts, like a list or tuple.
    :type multiple: bool
    :param units: Whether or not the iten on this line has units.
    :type units: bool
    :param new_type: What to conver the individual elements of the array to
                     if there are no units.
    """
    split_line = line.split()  # get the components
    # check for a None type
    if split_line[-1] == "None":
        return None
    # otherwise we can do our thing.
    if units:
        unit = split_line[-1]  # unit will be the last thing
        if multiple:
            values = split_line[1:-1]  # first item is the name, last is unit
        else:
            values = split_line[1]  # just get the single value, not a list
        return yt.YTArray(values, unit)  # put the unit on the data
    else:  # no units.
        # We have to convert everything to floats here, since we don't have
        # yt to do that for us.
        if multiple:
            return np.array([new_type(item) for item in split_line[1:]])
        else:
            return new_type(split_line[1])


def _parse_kde_line(line):
    """ Parse a single line containing data from the KDE profile.

    This will return a three element tuple. The first is the type of data this
    belongs to, which will either be "radii" or "densities". The next will be
    the key for either the radii or densities dictionary that this line belongs
    to. The last item will be the list of values that is the value in this
    dictionary.

    :param line: Line containing KDE data to parse.
    :returns: "radii" or "density", telling which dictionary these values
              belong to.
    :returns: key into the dictonary above
    :returns: values that go in that dictionary, holding either the radii or
              densities, as indicated.
    """
    split_line = line.split()
    parsed_key = split_line[0]
    # this parsed key is of the format "dict_key", where key will have
    # underscores in it too.
    data_type = parsed_key.split("_")[0]
    # we then mangle the rest back into the right format.
    key = "_".join(parsed_key.split("_")[1:]).strip(":")
    values = [float(item) for item in split_line[1:]]
    return data_type, key, values


def read_gal(ds, file_obj):
    """Reads a galaxy object from a file.

    The file has to be an already opened file object that is at the location
    of a galaxy object, written by the Galaxy.write() function. This function
    will return a Galaxy object with all the attributes filled in.

    If the file is not in the right spot, a ValueError will be raised.

    :param ds: dataset that the galaxy belongs to.
    :param file_obj: already opened file that is at the location of a new
                     galaxy object, as described above.
    :returns: Galaxy object with the data filled in.
    :rtype: Galaxy
    """
    # first find the location in the file where the the galaxy object starts.
    # there could be blank lines, which we ignore. I can't iterate through the
    # file directly, since I need to get individual lines later, and Python
    # won't allow to mix both styles.
    while True:
        line = file_obj.readline()
        if line.strip() == "new_galaxy_here":
            break  # this is what we want.
        elif line != "\n":  # if it's anything other than blank. The end of the
            # file will be an empty string, so it will get caught too.
            raise ValueError("File is not in the right spot for reading")

    # we are now at the right spot. Each line following this is a single
    # known value that is easy to grab.
    id = _parse_line(file_obj.readline(), multiple=False, units=False)
    center = _parse_line(file_obj.readline(), multiple=True, units=True)
    radius = _parse_line(file_obj.readline(), multiple=False, units=True)
    disk_kde_radius = _parse_line(file_obj.readline(),
                                  multiple=False, units=True)
    disk_kde_height = _parse_line(file_obj.readline(),
                                  multiple=False, units=True)
    disk_kde_normal = _parse_line(file_obj.readline(),
                                  multiple=True, units=False)
    disk_nsc_radius = _parse_line(file_obj.readline(),
                                  multiple=False, units=True)
    disk_nsc_height = _parse_line(file_obj.readline(),
                                  multiple=False, units=True)
    disk_nsc_normal = _parse_line(file_obj.readline(),
                                  multiple=True, units=False)

    nsc_idx_sphere = _parse_line(file_obj.readline(),
                                 multiple=True, units=False, new_type=int)
    nsc_idx_disk_nsc = _parse_line(file_obj.readline(),
                                 multiple=True, units=False, new_type=int)
    nsc_idx_disk_kde = _parse_line(file_obj.readline(),
                                 multiple=True, units=False, new_type=int)

    mean_rot_vel = _parse_line(file_obj.readline(), multiple=False, units=True)
    nsc_3d_sigma = _parse_line(file_obj.readline(), multiple=False, units=True)

    # we can create the galaxy at this point
    gal = Galaxy(ds, center, radius, id)
    # we then add the disk without calculating angular momentum by
    # specifying the normal vector. This saves computation time.
    gal.add_disk(disk_radius=disk_kde_radius, disk_height=disk_kde_height,
                 normal=disk_kde_normal)
    if disk_nsc_radius is not None:
        gal.add_disk(disk_radius=disk_nsc_radius, disk_height=disk_nsc_height,
                     normal=disk_nsc_normal, disk_type="nsc")
    else:
        gal.disk_nsc = None

    # assign the NSC indices and velocity stuff
    gal.nsc_idx_sphere = nsc_idx_sphere
    gal.nsc_idx_disk_nsc = nsc_idx_disk_nsc
    gal.nsc_idx_disk_kde = nsc_idx_disk_kde
    gal.mean_rot_vel = mean_rot_vel
    gal.nsc_3d_sigma = nsc_3d_sigma

    # then we get to the KDE values.
    while True:
        line = file_obj.readline()
        if line.strip() == "end_of_galaxy":
            break  # this is the end of the file
        # if we aren't at the end, parse the line
        data_type, key, values = _parse_kde_line(line)
        # then assign the values to the correct dictionary
        if data_type == "radii":
            gal.radii[key] = values
            gal.binned_radii[key] = utils.bin_values(values, 100)
        else:  # densities
            gal.densities[key] = values
            gal.binned_densities[key] = utils.bin_values(values, 100)

    # then we can do the fun stuff where we calculate everythign of interest.
    # this should all be pretty quick, since the KDE process has already been
    # read in and doesn't need to be repeated.
    try:
        gal.find_nsc_radius()
        gal.create_axis_ratios()
        gal.create_abundances()
    except AttributeError:  # will happen if no NSC
        pass

    return gal

id_start = 100
def _assign_id():
    global id_start
    id_start += 1
    return id_start

class Galaxy(object):
    def __init__(self, dataset, center, radius, id=None, j_radius=None,
                 disk_radius=None, disk_height=None):
        """Create a galaxy object at the specified location with the 
        given size. 
        
        :param dataset: yt dataset object that contains this galaxy
        :param center: 3 element array with the center of the galaxy. Must have
                       units.
        :param radius: Radius that will be used to create the sphere object.
                       Must also have units.
        :param id: identification number for this galaxy. Can be arbitrary.
        :params j_radius, disk_radius, disk_height: used for the creation of
                the disk. See the _add_disk functionality for detailed
                explanation. If these are left blank no disk will be
                created.
        """
        if id is not None:
            self.id = id
        else:
            self.id = _assign_id()

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
        # the kernel we will use should be the width of the cell, to match the
        # smoothing length of the simulation.
        self.kernel_size = 6 * yt.units.pc

        # then there are several quantities we initialize to zero or blank, but
        # will be filled in future analyses
        self._star_kde_mass_3d = None  # used for KDE profiles
        self._star_kde_metals_3d = None  # used for KDE profiles
        self._star_kde_mass_2d = None  # used for KDE profiles
        self._star_kde_metals_2d = None  # used for KDE profiles
        self.radii = dict()  # used for radial profiles
        self.densities = dict()  # used for radial profiles
        self.binned_radii = dict()  # used for radial profiles
        self.binned_densities = dict()  # used for radial profiles
        self.disk_kde = None  # used for cylindrical plots
        self.disk_nsc = None  # used for cylindrical plots
        self.nsc = None  # used for NSC analysis
        self.nsc_radius = None  # used for NSC analysis
        self.nsc_idx_sphere = None  # used for NSC analysis
        self.nsc_idx_disk_kde = None  # used for NSC analysis
        self.nsc_idx_disk_nsc = None  # used for NSC analysis
        self.nsc_axis_ratios = None  # used for rotation analysis
        self.mean_rot_vel = None  # used for rotation analysis
        self.nsc_3d_sigma = None  # used for rotation analysis
        self.nsc_abundances = None  # used for elemental abundances
        self.gal_abundances = None  # used for elemental abundances

        # we can then add a disk if the user wants to, and initialize the rest
        # of everything that comes after that.
        if (j_radius is not None) or (disk_height is not None) or \
                (disk_radius is not None):
            self.add_disk(j_radius, disk_radius, disk_height)
            self.find_nsc_radius()
            self.create_axis_ratios()
            self.nsc_rotation()
            self.create_abundances()

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
            container = self.disk_kde
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
        else:
            raise ValueError("Quanity {} is not supported.".format(quantity))

        # we can then create the KDE object
        return kde.KDE(locations, values)

    def add_disk(self, j_radius=None, disk_radius=None, disk_height=None,
                 normal=None, disk_type="kde"):
        """Creates a disk aligned with the angular momentum of the galaxy.
        
        This works by creating a sphere with the radius=`j_radius`, calculating 
        the angular momentum vector within that sphere, then creating a disk
        who's normal vector is that angular momentum vector. This should give 
        us a disk that aligns with the actual disk of the galaxy.
        
        :param j_radius: radius within which we will calculate the angular 
                         momentum. Needs units.
        :param disk_radius: radius of the resulting disk. Needs units.
        :param disk_height: height of the resulting disk. Needs units.
        :param normal: If you already know the normal, this will add the disk
                       without calculating any of the angular momentum stuff.
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

        if normal is None:
            # then we can go ahead and do things. First create the new sphere
            j_vec_sphere = self.ds.sphere(center=self.center, radius=j_radius)
            # find its angular momentum
            normal = j_vec_sphere.quantities.angular_momentum_vector()
        # then create the disk
        disk = self.ds.disk(center=self.center, normal=normal,
                            height=disk_height, radius=disk_radius)
        if disk_type == "kde":
            self.disk_kde = disk
        elif disk_type == "nsc":
            self.disk_nsc = disk
        else:
            raise ValueError("Disk type not specified properly.")

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

        # Our kernel needs to be in parsecs, since the x, y, and z values
        # are in parsecs (see this class's __init__).
        kernel_size = self.kernel_size.in_units("pc").value

        # we can then go ahead with the centering, using the user's desired
        # accuracy
        if self._star_kde_mass_3d is None:
            self._star_kde_mass_3d = self._create_kde_object(3, "mass")

        # we will start the centering at the center of mass of the galaxy
        # and will only search a 1 kpc region around this center
        com = self.sphere.quantities.center_of_mass(use_particles=True)
        com = np.array(com.in_units("pc"))
        self._star_kde_mass_3d.centering(kernel_size, accuracy,
                                         initial_guess=com,
                                         search_region_size=1000*yt.units.pc)

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
        if dimension == 2 and self.disk_kde is None:
            raise RuntimeError("Need to create a disk first. ")
        if quantity.lower() not in ["mass", "z"]:
            raise ValueError("Only mass and Z are supported. ")

    def stellar_mass(self, radius_cut=None):
        """Calculate the total stellar mass in this galaxy. 
        
        This sums the masses of the star particles that are inside the radius 
        of the galaxy or NSC, and returns the answer in solar masses.

        """

        masses = self.sphere[('STAR', "MASS")].in_units("msun")
        if radius_cut is not None:
            # check for units
            utils.test_for_units(radius_cut, "radius_cut")
            # get the mass within this radius.
            radius_key = ('STAR', 'particle_position_spherical_radius')
            idx = np.where(self.sphere[radius_key] < radius_cut)[0]
            return np.sum(masses[idx])

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
            # create the KDE objects if needed
            if self._star_kde_mass_2d is None:
                self._star_kde_mass_2d = self._create_kde_object(2, "mass")
            mass_kde = self._star_kde_mass_2d
            # only assign metals if needed
            if quantity == "Z":
                if self._star_kde_metals_2d is None:
                    self._star_kde_metals_2d = self._create_kde_object(2, "Z")
                metals_kde = self._star_kde_metals_2d
        else:  # spherical: keep regular center
            if self._star_kde_mass_3d is None:
                self._star_kde_mass_3d = self._create_kde_object(3, "mass")
            mass_kde = self._star_kde_mass_3d
            # only assigne metals if needed
            if quantity == "Z":
                if self._star_kde_metals_3d is None:
                    self._star_kde_metals_3d = self._create_kde_object(3, "Z")
                metals_kde = self._star_kde_metals_3d

        # then create the radii
        radii = np.arange(0, outer_radius, spacing)

        # the smoothing kernel we will use in the KDE process is half the size
        # of the smallest cell in the sphere
        kernel_size = self.kernel_size.in_units("pc").value

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

        # store the binned radii too, since I will be using those
        self.binned_radii[key] = utils.bin_values(radii, 100)
        self.binned_densities[key] = utils.bin_values(final_densities, 100)

    def find_nsc_radius(self):
        """
        Finds the radius of the NSC, using the KDE profile and the associated
        functionality in the NscStructure class.

        :return:
        """

        # first need to to the KDE fitting procedure, possibly.
        if "mass_kde_2D" not in self.radii:
            self.kde_profile("MASS", spacing=0.05 * yt.units.pc,
                             outer_radius=1000 * yt.units.pc)
        self.nsc = nsc_structure.NscStructure(self.binned_radii["mass_kde_2D"],
                                              self.binned_densities["mass_kde_2D"])

        if self.nsc.nsc_radius is None:
            self.nsc_radius = None
            return
        # if not none we continue
        self.nsc_radius = self.nsc.nsc_radius * yt.units.pc
        self.nsc_radius_err = self.nsc.nsc_radius_err * yt.units.pc

        # then create a new disk object that entirely contains the NSC. I choose
        # twoce the radius so that we are sure to include everything inside,
        # since yt selects based on cells, not particles.
        self.add_disk(normal=self.disk_kde._norm_vec,
                      disk_height=2 * self.nsc_radius,
                      disk_radius=2 * self.nsc_radius, disk_type="nsc")

        # then get the indices of the stars actually in the NSC
        if self.nsc_idx_sphere is None:
            radius_key = ('STAR', 'particle_position_spherical_radius')
            self.nsc_idx_disk_kde = np.where(self.disk_kde[radius_key] <
                                             self.nsc_radius)[0]
            self.nsc_idx_disk_nsc = np.where(self.disk_nsc[radius_key] <
                                             self.nsc_radius)[0]
            self.nsc_idx_sphere = np.where(self.sphere[radius_key] <
                                           self.nsc_radius)[0]

        # then check that there are actually stars in the NSC
        if len(self.nsc_idx_sphere) == 0 or len(self.nsc_idx_disk_nsc) == 0:
            self.nsc_radius = None

    def create_axis_ratios(self):
        """Creates the axis ratios object. """
        self._check_nsc_existence()

        # get the locations of the stars in the NSC. Convert to numpy arrays
        # for speed.
        x = np.array(self.sphere[('STAR', 'POSITION_X')].in_units("pc"))
        y = np.array(self.sphere[('STAR', 'POSITION_Y')].in_units("pc"))
        z = np.array(self.sphere[('STAR', 'POSITION_Z')].in_units("pc"))
        mass = np.array(self.sphere[('STAR', "MASS")].in_units("msun"))
        # then get just the NSC ones, and subtract off the center
        x = x[self.nsc_idx_sphere] - self.center[0].in_units("pc").value
        y = y[self.nsc_idx_sphere] - self.center[1].in_units("pc").value
        z = z[self.nsc_idx_sphere] - self.center[2].in_units("pc").value
        mass = mass[self.nsc_idx_sphere]

        self.nsc_axis_ratios = nsc_structure.AxisRatios(x, y, z, mass)

    def _check_nsc_existence(self):
        """Checks if we have a NSC radius set or not. Raises an AttributeError
        if not. """
        if self.nsc_radius is None:
            raise AttributeError("NSC radius has not been set.")

    def nsc_rotation(self):
        """Calculates the mean rotational velocity and 3D velocity dispersion.

        These quantities are mass weighted."""

        self._check_nsc_existence()  # need an NSC

        radial_key = ('STAR', 'particle_velocity_cylindrical_radius')
        theta_key = ('STAR', 'particle_velocity_cylindrical_theta')
        z_key = ('STAR', 'particle_velocity_cylindrical_z')

        vel_rad = self.disk_nsc[radial_key].in_units("km/s")
        vel_rot = self.disk_nsc[theta_key].in_units("km/s")
        vel_z = self.disk_nsc[z_key].in_units("km/s")
        masses = self.disk_nsc[('STAR', 'MASS')].in_units("msun")

        # then restrict down the NSC disk
        vel_rad = vel_rad[self.nsc_idx_disk_nsc]
        vel_rot = vel_rot[self.nsc_idx_disk_nsc]
        vel_z = vel_z[self.nsc_idx_disk_nsc]
        masses = masses[self.nsc_idx_disk_nsc]

        self.mean_rot_vel = utils.weighted_mean(vel_rot, masses)

        sigma_radial = np.sqrt(utils.weighted_variance(vel_rad, masses, ddof=0))
        sigma_rot = np.sqrt(utils.weighted_variance(vel_rot, masses, ddof=0))
        sigma_z = np.sqrt(utils.weighted_variance(vel_z, masses, ddof=0))

        self.nsc_3d_sigma = utils.sum_in_quadrature(sigma_z, sigma_rot,
                                                    sigma_radial)

    def create_abundances(self):
        """Creates the abundance objects, which handle all the elemental
        abundance stuff, like [Z/H], [Fe/H], and [X/Fe].

        One object is created for the NSC (self.nsc_abundances), and one for
        the whole galaxy (self.gal_abundances)."""

        self._check_nsc_existence()  # need an NSC

        # we need masses, Z_Ia, and Z_II. I can convert these to arrays to
        # help speed, since the units don't matter in the metallicity
        # calculations. As long as the masses are relative it will still work.
        masses = np.array(self.sphere[('STAR', 'MASS')].in_units("msun"))
        z_Ia = np.array(self.sphere[('STAR', 'METALLICITY_SNIa')])
        z_II = np.array(self.sphere[('STAR', 'METALLICITY_SNII')])

        # the objects can then be created, where for the NSC we select only
        # the objects in the NSC
        self.gal_abundances = abundances.Abundances(masses, z_Ia, z_II)
        self.nsc_abundances = abundances.Abundances(masses[self.nsc_idx_sphere],
                                                    z_Ia[self.nsc_idx_sphere],
                                                    z_II[self.nsc_idx_sphere])

    def write(self, file_obj):
        """Writes the galaxy object to a file, to be read in later.

        This is done to save time, since the KDE process can be expensive and
        takes a while. This function only writes out the densities and radii
        in the KDE processes that have been used so far, as well as some
        essential data like the center.

        :param file_obj: already opened file object that is ready to be
                         written to.
        :returns: None, but writes to the file.
        """

        # write essential data first.
        file_obj.write("\n")
        file_obj.write("new_galaxy_here\n")  # tag so later read-in is easier
        _write_single_item(file_obj, self.id, "id", units=False, multiple=False)
        _write_single_item(file_obj, self.center.in_units("pc"), "center",
                           units=True, multiple=True)
        _write_single_item(file_obj, self.radius.in_units("pc"), "radius",
                           units=True, multiple=False)
        # disk values
        _write_single_item(file_obj, self.disk_kde.radius.in_units("pc"),
                           "disk_kde_radius", units=True)
        _write_single_item(file_obj, self.disk_kde.height.in_units("pc"),
                           "disk_kde_height", units=True)
        _write_single_item(file_obj, self.disk_kde._norm_vec, "disk_kde_normal",
                           multiple=True)
        if self.disk_nsc is not None:
            _write_single_item(file_obj, self.disk_nsc.radius.in_units("pc"),
                               "disk_nsc_radius", units=True)
            _write_single_item(file_obj, self.disk_nsc.height.in_units("pc"),
                               "disk_nsc_height", units=True)
            _write_single_item(file_obj, self.disk_nsc._norm_vec,
                               "disk_kde_normal", multiple=True)
            # NSC indexes, which take a while to build in the first place, so it's
            # better to write them to file now
            _write_single_item(file_obj, self.nsc_idx_sphere, "nsc_idx_sphere",
                               multiple=True)
            _write_single_item(file_obj, self.nsc_idx_disk_nsc,
                               "nsc_idx_disk_nsc", multiple=True)
            _write_single_item(file_obj, self.nsc_idx_disk_kde,
                               "nsc_idx_disk_kde", multiple=True)
            # same with the rotational stuff, which requires access to the disk obj
            _write_single_item(file_obj, self.mean_rot_vel, "mean_rot_vel",
                               multiple=False, units=True)
            _write_single_item(file_obj, self.nsc_3d_sigma, "nsc_3d_sigma",
                               multiple=False, units=True)
        else:
            _write_single_item(file_obj, None, "disk_nsc_radius")
            _write_single_item(file_obj, None, "disk_nsc_height")
            _write_single_item(file_obj, None, "disk_kde_normal")
            _write_single_item(file_obj, None, "nsc_idx_sphere")
            _write_single_item(file_obj, None, "nsc_idx_disk_nsc")
            _write_single_item(file_obj, None, "nsc_idx_disk_kde")
            _write_single_item(file_obj, None, "mean_rot_vel")
            _write_single_item(file_obj, None, "nsc_3d_sigma")


        # then all we need are the KDE values
        for key in self.radii:
            _write_single_item(file_obj, self.radii[key],
                               "radii_{}".format(key), multiple=True)
            _write_single_item(file_obj, self.densities[key],
                              "density_{}".format(key), multiple=True)
        file_obj.write("end_of_galaxy\n")  # tag so later read-in is easier

    def contains(self, other_gal):
        """
        Checks whether this galaxy entirely contains another.

        A galaxy cannot contain itself.

        :param other_gal: Other galaxy that will be used to see if this gal is
                          contained within self.
        :return: Whether or not this galaxy is contained within the other.
        :rtype: bool
        """
        # to use the comparison in the utils file, everything has to be in the
        # same units.
        cen_1 = other_gal.center.in_units("kpc").value
        cen_2 = self.center.in_units("kpc").value

        radius_1 = other_gal.radius.in_units("kpc").value
        radius_2 = self.radius.in_units("kpc").value

        return utils.sphere_containment(cen_1, cen_2, radius_1, radius_2)






