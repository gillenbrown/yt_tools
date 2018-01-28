import yt
import numpy as np
import time
from scipy import integrate

from . import kde
from . import utils
from . import nsc_structure
from . import nsc_abundances


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
            return yt.YTArray(values, unit)  # put the unit on the data
        else:
            values = float(split_line[1])  # just get the single value
            # have to convert to float because YTQuantity doesn't take
            # strings, unlike YTArray
            return yt.YTQuantity(values, unit)  # put the unit on the data

    else:  # no units.
        # We have to convert everything to floats here, since we don't have
        # yt to do that for us.
        if multiple:
            return np.array([new_type(item) for item in split_line[1:]])
        else:
            return new_type(split_line[1])


def _parse_kde_line(line):
    """ Parse a single line containing data from the KDE profile.

    This will return a four element tuple. The first is the type of data this
    belongs to, which will either be "radii" or "densities". The next will be
    the key for either the radii or densities dictionary that this line belongs
    to. The third will be a bool indicating whether or not these value are
    smoothed. The last item will be the list of values that is the value in this
    dictionary.

    :param line: Line containing KDE data to parse.
    :returns: "radii" or "density", telling which dictionary these values
              belong to.
    :returns: key into the dictonary above
    :returns: bool of whether or not this is smoothed.
    :returns: values that go in that dictionary, holding either the radii or
              densities, as indicated.
    """
    split_line = line.split()
    parsed_key = split_line[0]
    # this parsed key is of the format "dict_key", where key will have
    # underscores in it too.
    if "radii" in parsed_key:
        data_type = "radii"
    else:
        data_type = "density"

    # get the values
    values = [float(item) for item in split_line[1:]]
    # see if it is smoothed
    smoothed = "smoothed" in line
    # we then mangle the rest back into the right format.
    if smoothed:
        key = "_".join(parsed_key.split("_")[2:]).strip(":")
    else:
        key = "_".join(parsed_key.split("_")[1:]).strip(":")
    return data_type, key, smoothed, values


# and a custom exception when the file is done with galaxies.
class EndOfGalsErr(Exception):
    pass


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
        elif line.strip() == "End of file.":
            raise EndOfGalsErr
        elif line != "\n":  # if it's anything other than blank. The end of the
            # file will be an empty string, so it will get caught too.
            raise ValueError("File is not in the right spot for reading")

    # we are now at the right spot. Each line following this is a single
    # known value that is easy to grab.
    id = _parse_line(file_obj.readline(), multiple=False, units=False,
                     new_type=int)
    center = _parse_line(file_obj.readline(), multiple=True, units=True)
    radius = _parse_line(file_obj.readline(), multiple=False, units=True)
    j_radius = _parse_line(file_obj.readline(), multiple=False, units=True)
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

    nsc_idx_j_sphere = _parse_line(file_obj.readline(),
                                   multiple=True, units=False, new_type=int)
    nsc_idx_disk_nsc = _parse_line(file_obj.readline(),
                                 multiple=True, units=False, new_type=int)

    # mean_rot_vel = _parse_line(file_obj.readline(), multiple=False, units=True)
    # nsc_3d_sigma = _parse_line(file_obj.readline(), multiple=False, units=True)
    # nsc_sigma_a = _parse_line(file_obj.readline(), multiple=False, units=True)
    # nsc_sigma_b = _parse_line(file_obj.readline(), multiple=False, units=True)
    # nsc_sigma_c = _parse_line(file_obj.readline(), multiple=False, units=True)

    r_nsc = _parse_line(file_obj.readline(), multiple=False, units=True)
    r_nsc_err = _parse_line(file_obj.readline(), multiple=True, units=True)
    # r_half = _parse_line(file_obj.readline(), multiple=False, units=False)
    # r_half_err = _parse_line(file_obj.readline(), multiple=True, units=False)

    # we can create the galaxy at this point
    gal = Galaxy(ds, center, radius, j_radius, id)
    # we then add the disk without calculating angular momentum by
    # specifying the normal vector. This saves computation time.
    gal.add_disk(disk_radius=disk_kde_radius, disk_height=disk_kde_height,
                 normal=disk_kde_normal)
    if disk_nsc_radius is not None:
        gal.add_disk(disk_radius=disk_nsc_radius, disk_height=disk_nsc_height,
                     normal=disk_nsc_normal, disk_type="nsc")
    else:
        gal.disk_nsc = None

    # assign the radii of interest
    gal.nsc_radius = r_nsc
    gal.nsc_radius_err = r_nsc_err
    # gal.half_mass_radius = r_half
    # gal.half_mass_radius_errs = r_half_err

    # assign the NSC indices
    gal.nsc_idx_j_sphere = nsc_idx_j_sphere
    gal.nsc_idx_disk_nsc = nsc_idx_disk_nsc
    # and velocity stuff
    # gal.mean_rot_vel = mean_rot_vel
    # gal.nsc_3d_sigma = nsc_3d_sigma
    # gal.nsc_disp_along_a = nsc_sigma_a
    # gal.nsc_disp_along_b = nsc_sigma_b
    # gal.nsc_disp_along_c = nsc_sigma_c

    # then the binned profile
    gal.binned_radii = _parse_line(file_obj.readline(),
                                   units=False, multiple=True)
    gal.binned_densities = _parse_line(file_obj.readline(),
                                       units=False, multiple=True)
    gal.integrated_kde_radii = _parse_line(file_obj.readline(),
                                           units=False, multiple=True)
    gal.integrated_kde_densities = _parse_line(file_obj.readline(),
                                               units=False, multiple=True)

    # then we get to the KDE values.
    while True:
        line = file_obj.readline()
        if line.strip() == "end_of_galaxy":
            break  # this is the end of the file
        # if we aren't at the end, parse the line
        data_type, key, smoothed, values = _parse_kde_line(line)
        # then assign the values to the correct dictionary
        if smoothed:
            if data_type == "radii":
                gal.kde_radii_smoothed[key] = values
            else:  # densities
                gal.kde_densities_smoothed[key] = values
        else:
            if data_type == "radii":
                gal.kde_radii[key] = values
            else:  # densities
                gal.kde_densities[key] = values

    # then we can do the fun stuff where we calculate everythign of interest.
    # this should all be pretty quick, since the KDE process has already been
    # read in and doesn't need to be repeated.
    gal.setup_nsc_object()
    # try:
    #     gal.setup_nsc_object()
    #     gal.create_axis_ratios_nsc()
    #     gal.create_axis_ratios_gal()
    #     gal.create_abundances()
    #     gal.nsc_rotation()
    # except AttributeError:  # will happen if no NSC
    #     pass

    return gal

id_start = 100
def _assign_id():
    global id_start
    id_start += 1
    return id_start

class Galaxy(object):
    def __init__(self, dataset, center, radius, j_radius, id=None):
        """Create a galaxy object at the specified location with the 
        given size. 
        
        :param dataset: yt dataset object that contains this galaxy
        :param center: 3 element array with the center of the galaxy. Must have
                       units.
        :param radius: Radius that will be used to create the sphere object.
                       Must also have units.
        :param j_radius: Radius used to calculate the disk plane. Only the
                         part of the galaxy inside this radius will be used.
                         0.2 times the virial radius is typical.
        :param id: identification number for this galaxy. Can be arbitrary.
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
        self.center = self.ds.arr(center)

        # we need to check that the radius has units too
        utils.test_for_units(radius, "radius")
        utils.test_for_units(j_radius, "j_radius")
        # set the attribute if it passes tests
        self.radius = self.ds.quan(radius)
        self.j_radius = self.ds.quan(j_radius)

        # create the sphere that contains the galaxy.
        self.sphere = self.ds.sphere(center=self.center, radius=self.radius)
        self.j_sphere = self.ds.sphere(center=self.center, radius=self.j_radius)

        # and find the smallest cell size (used for KDE)
        self.min_dx = np.min(self.sphere[('index', 'dx')])
        # the kernel we will use should be the width of the cell, to match the
        # smoothing length of the simulation.
        self.kernel_size = 3 * yt.units.pc

        # then there are several quantities we initialize to zero or blank, but
        # will be filled in future analyses
        self._star_kde_mass_3d = None  # used for KDE profiles
        self._star_kde_metals_3d = None  # used for KDE profiles
        self._star_kde_mass_2d = None  # used for KDE profiles
        self._star_kde_metals_2d = None  # used for KDE profiles
        self._star_kde_mass_1d = None  # used for KDE profiles
        self._star_kde_metals_1d = None  # used for KDE profiles
        self.kde_radii = dict()  # used for radial profiles
        self.kde_densities = dict()  # used for radial profiles
        self.kde_radii_smoothed = dict()  # used for radial profiles
        self.kde_densities_smoothed = dict()  # used for radial profiles
        self.binned_radii = None  # used for radial profiles
        self.binned_densities = None  # used for radial profiles
        self.integrated_kde_radii = None  # used for radial profiles
        self.integrated_kde_densities = None  # used for radial profiles
        self.surface_1D_radii = None  # used for radial profiles
        self.surface_1D_densities = None  # used for radial profiles
        self.disk_kde = None  # used for cylindrical plots
        self.disk_whole = None  # used for cylindrical plots
        self.disk_nsc = None  # used for cylindrical plots
        self.nsc = None  # used for NSC analysis
        self.nsc_radius = None  # used for NSC analysis
        self.nsc_radius_err = None # used for NSC analysis
        self.nsc_idx_j_sphere = None  # used for NSC analysis
        self.nsc_idx_disk_kde = None  # used for NSC analysis
        self.nsc_idx_disk_nsc = None  # used for NSC analysis
        self.half_mass_radius = None  # used for NSC analysis
        self.half_mass_radius_errs = None  # used for NSC analysis
        self.nsc_axis_ratios = None  # used for rotation analysis
        self.gal_axis_ratios = None  # used for disk plane
        self.mean_rot_vel = None  # used for rotation analysis
        self.nsc_sigma_radial = None  # used for rotation analysis
        self.nsc_sigma_rot = None  # used for rotation analysis
        self.nsc_sigma_z = None  # used for rotation analysis
        self.nsc_3d_sigma = None  # used for rotation analysis
        self.nsc_disp_along_a = None  # used for rotation analysis
        self.nsc_disp_along_b = None  # used for rotation analysis
        self.nsc_disp_along_c = None  # used for rotation analysis
        self.nsc_abundances = None  # used for elemental abundances
        self.gal_abundances = None  # used for elemental abundances
        self.sfh_time = None  # used for elemental abundances

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
        elif dimension == 2:  # cylindrical
            container = self.disk_kde
            r = container[('STAR', 'particle_position_cylindrical_radius')]
            r = np.array(r.in_units("pc"))
            theta = container[('STAR', 'particle_position_cylindrical_theta')]
            theta = np.array(theta)
            # convert to cartesian
            x, y = utils.convert_polar_to_cartesian(r, theta)
            # store these location
            locations = (x, y)
        else: # radial
            container = self.disk_kde
            r = container[('STAR', 'particle_position_cylindrical_radius')]
            r = np.array(r.in_units("pc"))
            # store these location
            locations = (r,)  # weird comma needed for single element tuple

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

    def add_disk(self, disk_radius=None, disk_height=None, normal=None,
                 disk_type="kde", method="axis_ratios"):
        """Creates a disk aligned with the angular momentum of the galaxy.
        
        This works by creating a sphere with the radius=`j_radius`, calculating 
        the angular momentum vector within that sphere, then creating a disk
        who's normal vector is that angular momentum vector. This should give 
        us a disk that aligns with the actual disk of the galaxy.

        :param disk_radius: radius of the resulting disk. Needs units.
        :param disk_height: height of the resulting disk. Needs units.
        :param normal: If you already know the normal, this will add the disk
                       without calculating any of the angular momentum stuff.
        :returns: None, but creates the disk attribute for the galaxy. 
        """
        # set default values if not already set
        if disk_radius is None:
            disk_radius = self.radius
        if disk_height is None:
            disk_height = self.radius
        # then check units
        utils.test_for_units(disk_radius, "disk_radius")
        utils.test_for_units(disk_height, "disk_height")

        if normal is None:
            if method == "axis_ratios":
                if self.gal_axis_ratios is None:
                    self.create_axis_ratios_gal()
                normal = self.gal_axis_ratios.c_vec  # vector of smallest axis.
            elif method == "angular_momentum":
                normal = self.j_sphere.quantities.angular_momentum_vector(use_gas=True,
                                                                          use_particles=True)
            else:
                raise ValueError("Need to specify method for disk "
                                 "orientation correctly.")
        # then create the disk
        disk = self.ds.disk(center=self.center, normal=normal,
                            height=disk_height, radius=disk_radius)
        if disk_type == "kde":
            self.disk_kde = disk
        elif disk_type == "nsc":
            self.disk_nsc = disk
        else:
            raise ValueError("Disk type not specified properly.")

        # then create a whole disk if needed
        if self.disk_whole is None:
            big_disk = self.ds.disk(center=self.center, normal=normal,
                                    height=disk_height, radius=self.radius)
            self.disk_whole = big_disk

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
        # and will only search a 10 kpc region around this center
        # com_yt = self.sphere.quantities.center_of_mass(use_particles=True,
        #                                                use_gas=False)
        # com_yt = np.array(com_yt.in_units("pc"))
        com_x = utils.weighted_mean(self._star_kde_mass_3d.x,
                                    self._star_kde_mass_3d.values)
        com_y = utils.weighted_mean(self._star_kde_mass_3d.y,
                                    self._star_kde_mass_3d.values)
        com_z = utils.weighted_mean(self._star_kde_mass_3d.z,
                                    self._star_kde_mass_3d.values)
        com = [com_x, com_y, com_z] # parsecs are already assumed.
        self._star_kde_mass_3d.centering(kernel_size, accuracy,
                                         initial_guess=com,
                                         search_region_size=10000) # parsecs

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
        if dimension not in [1, 2, 3]:
            raise ValueError("`dimension` must be either 1, 2, or 3.")
        # check that we are able to actually do the cylindrical profiles
        if dimension in [1, 2] and self.disk_kde is None:
            raise RuntimeError("Need to create a disk first. ")
        if quantity.lower() not in ["mass", "z"]:
            raise ValueError("Only mass and Z are supported. ")

    def stellar_mass(self, radius_cut=None, spherical=True):
        """Calculate the total stellar mass in this galaxy. 
        
        This sums the masses of the star particles that are inside the radius 
        of the galaxy or NSC, and returns the answer in solar masses.

        """
        if spherical:
            container = self.sphere
            radius_key = ('STAR', 'particle_position_spherical_radius')
        else:
            container = self.disk_kde
            radius_key = ('STAR', 'particle_position_cylindrical_radius')

        masses = container[('STAR', "MASS")].in_units("msun")

        if radius_cut is not None:
            # check for units
            utils.test_for_units(radius_cut, "radius_cut")
            # get the mass within this radius.
            idx = np.where(container[radius_key] < radius_cut)[0]
            return np.sum(masses[idx])

        else:  # whole galaxy
            return np.sum(masses)

    def galaxy_half_mass_radius(self):
        """Calculates the half mass radius of the galaxy.

        This is just based on star particles, and finds the radius at which
        there is more than half the total mass enclosed. This only has a
        resolution of 50 pc right now, which is hopefully good enough for the
        brief comparison we will make.
        """
        total_gal_mass = self.stellar_mass(radius_cut=None)
        # then iterate through a range of radii
        for radius in np.arange(0, 10, 0.05) * yt.units.kpc:
            interior_mass = self.stellar_mass(radius_cut=radius)
            # when we get to more than half of the total, we have the
            # half mass radius.
            if interior_mass > 0.5 * total_gal_mass:
                return radius

    def _parse_half_mass_radii(self, half_mass_best, half_mass_down,
                               half_mass_up):
        # turn into errors
        # then check if the NSC is dominated by one massive star particle. If
        # so, then our half mass is just an upper limit
        # half mass upper limits
        nsc_star_masses = self.j_sphere[('STAR', "MASS")][self.nsc_idx_j_sphere]
        if utils.max_above_half(nsc_star_masses):  # one star particle dominates
            err_down = half_mass_best  # just the radius.
        else:
            err_down = half_mass_best - half_mass_down
        # the upper error doesn't depend on the radius.
        err_up = half_mass_up - half_mass_best

        # then set the errors
        self.half_mass_radius = half_mass_best
        self.half_mass_radius_errs = (err_down, err_up)

    def _kde_nsc_half_mass_radius(self):
        # get the upper and lower limits on the NSC radius
        nsc_low = self.nsc_radius - self.nsc_radius_err[0]
        nsc_high = self.nsc_radius + self.nsc_radius_err[1]

        # the smoothing kernel we will use in the KDE process is half the size
        # of the smallest cell in the sphere when we are inside 12pc, but
        # twice this when outside of that distance.
        inner_kernel = float(self.kernel_size.in_units("pc").value)
        outer_kernel = inner_kernel * 2
        break_radius = inner_kernel * 4

        # then calclate the mass enclosed in each of those radii
        density_integrand = self._star_kde_mass_2d.density
        kwargs = {"inner_kernel": inner_kernel,
                  "break_radius": break_radius,
                  "outer_kernel": outer_kernel}
        half_masses = []
        for radius in [self.nsc_radius, nsc_low, nsc_high]:
            total_mass = utils.mass_annulus(density_func=density_integrand,
                                            radius_a=0,
                                            radius_b=radius.to("pc").value,
                                            error_tolerance=0.01,
                                            density_func_kwargs=kwargs)

            half_masses.append(total_mass / 2.0)

        # then calculate the mass in each annulus, and make it cumulative
        # until we reach half the mass
        bin_edges = np.concatenate([[0], np.logspace(0, 3, 301)])
        cumulative_mass = 0
        half_mass_radii = [0, 0, 0]
        half_mass_done = [False, False, False]
        for left_idx in range(len(bin_edges) - 1):
            right_idx = left_idx + 1
            radius_a = bin_edges[left_idx]
            radius_b = bin_edges[right_idx]
            # integrate over the annulus
            this_mass = utils.mass_annulus(density_func=density_integrand,
                                           radius_a=radius_a,
                                           radius_b=radius_b,
                                           error_tolerance=0.01,
                                           density_func_kwargs=kwargs)
            # then add it to the total
            cumulative_mass += this_mass

            # then check for each radius
            for idx in range(3):
                if not half_mass_done[idx]:
                    # If it's greater than half, we have our half mass radius
                    if cumulative_mass >= half_masses[idx]:
                        half_mass_radii[idx] = radius_b
                        half_mass_done[idx] = True
            # once we hit all of them, we are done.
            if all(half_mass_done):
                break

        self._parse_half_mass_radii(*half_mass_radii)

    def _particle_nsc_half_mass_radius(self):
        nsc_low = self.nsc_radius - self.nsc_radius_err[0]
        nsc_high = self.nsc_radius + self.nsc_radius_err[1]

        best_nsc_mass = self.stellar_mass(radius_cut=self.nsc_radius)
        low_nsc_mass = self.stellar_mass(radius_cut=nsc_low)
        high_nsc_mass = self.stellar_mass(radius_cut=nsc_high)

        half_masses = [mass / 2.0 for mass in
                       [best_nsc_mass, low_nsc_mass, high_nsc_mass]]
        half_mass_radii = [0, 0, 0]
        half_mass_done = [False, False, False]

        # then iterate through a range of radii
        for radius in np.arange(0, 500, 0.01) * yt.units.pc:
            interior_mass = self.stellar_mass(radius_cut=radius)
            # when we get to more than half of the total, we have the
            # half mass radius.
            for idx in range(3):
                if not half_mass_done[idx]:
                    # If it's greater than half, we have our half mass radius
                    if interior_mass >= half_masses[idx]:
                        half_mass_radii[idx] = radius.to("pc").value
                        half_mass_done[idx] = True
            # once we hit all of them, we are done.
            if all(half_mass_done):
                break
        print()

        self._parse_half_mass_radii(*half_mass_radii)

    def nsc_half_mass_radius(self):
        """
        Calculate the half mass radius of the galaxy by integrating the
        real KDE density, not the profile.

        :return: NSC half mass radius and errors
        """
        return 10, [2, 2]
        try:
            self._check_nsc_existence()
        except AttributeError:
            return None, [None, None]

        if self._star_kde_mass_2d is None:
            self._star_kde_mass_2d = self._create_kde_object(2, "mass")
        if len(self._star_kde_mass_2d.x) > 5 * 10**5:
            print("\n\n\n\n\doing particle half mass \n\n\n\n")
            self._particle_nsc_half_mass_radius()
        else:
            self._kde_nsc_half_mass_radius()

    def nsc_mass_and_errs(self):
        """Calculates the mass of the NSC and the errors on that.

        The mass of the NSC is determined by the sum of the mass of all the star
        particles that are inside the NSC radius. Then the errors are the errors
        that come from increasing the NSC radius by its errors. This is the
        same way that the errors are calculated in the NscStructure module for
        the smoothed profiles. """
        try:
            self._check_nsc_existence()
        except AttributeError:
            return None, [None, None]

        # get the upper and lower limits on the NSC radius
        nsc_low = self.nsc_radius - self.nsc_radius_err[0]
        nsc_high = self.nsc_radius + self.nsc_radius_err[1]

        this_nsc_mass = self.stellar_mass(radius_cut=self.nsc_radius)
        this_nsc_low_mass = self.stellar_mass(radius_cut=nsc_low)
        this_nsc_high_mass = self.stellar_mass(radius_cut=nsc_high)

        errors = (this_nsc_mass - this_nsc_low_mass,
                  this_nsc_high_mass - this_nsc_mass)
        return this_nsc_mass, errors

    def kde_profile(self, quantity="MASS", dimension=2, outer_radius=None):
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
        # test that both outer_radius and spacing have units
        utils.test_for_units(outer_radius, "outer_radius")

        # do the KDE error checking
        self._kde_profile_error_checking(quantity, dimension)

        # All the KDE stuff is in parsecs, so convert those to parsecs
        outer_radius = outer_radius.in_units("pc").value

        # the KDE objects that we want to use depend on the coordinate system,
        # as does the center. In 3D we just use the center of the galaxy, since
        # it is in Cartesian coords, but in 2D we use 0, since the transform to
        # xy from cylindrical already subtracts off the center
        if dimension == 1:
            # We'll say the 1D KDE isn't supported, although it really is. It
            # just isn't useful, so we won't include it.
            raise ValueError("1D KDE profile not supported now.")
            # center = [0]
            # # create the KDE objects if needed
            # if self._star_kde_mass_1d is None:
            #     self._star_kde_mass_1d = self._create_kde_object(1, "mass")
            # mass_kde = self._star_kde_mass_1d
            # # only assign metals if needed
            # if quantity == "Z":
            #     if self._star_kde_metals_1d is None:
            #         self._star_kde_metals_1d = self._create_kde_object(1, "Z")
            #     metals_kde = self._star_kde_metals_1d
        elif dimension == 2:
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
            raise ValueError("3D profile not supported")

        # then create the radii
        # we will do linear spacing within 100 pc, then log spacing outside
        # of that.
        radii = np.arange(0, outer_radius, 1)

        # the smoothing kernel we will use in the KDE process is half the size
        # of the smallest cell in the sphere when we are inside 12pc, but
        # twice this when outside of that distance.
        inner_kernel = self.kernel_size.in_units("pc").value
        outer_kernel = inner_kernel * 2
        break_radius = inner_kernel * 4

        # we are then able to do the actual profiles
        if dimension == 1:
            num_azimuthal = 1
        else:
            num_azimuthal = 100

        if quantity.lower() == "mass":
            # use the mass KDE object that's already set up
            profile = mass_kde.radial_profile(radii=radii, kernel=inner_kernel,
                                              outer_kernel=outer_kernel,
                                              break_radius=break_radius,
                                              num_each=num_azimuthal)
            full_radii, final_densities = profile

            # If we are in 1D, we need to divide by 2 pi r to get surface density
            if dimension == 1:
                surface_densities = final_densities / (2 * np.pi * full_radii)

        elif quantity == "Z":
            # for metallicity we have to calculate the mass in metals, and
            # divide it by the mass in stars
            metal_profile = metals_kde.radial_profile(radii=radii,
                                                      kernel=inner_kernel,
                                                      outer_kernel=outer_kernel,
                                                      break_radius=break_radius,
                                                      num_each=num_azimuthal)
            full_radii, metal_densities = metal_profile

            mass_profile = mass_kde.radial_profile(radii=radii,
                                                   kernel=inner_kernel,
                                                   outer_kernel=outer_kernel,
                                                   break_radius=break_radius,
                                                   num_each=num_azimuthal)
            full_radii, mass_densities = mass_profile
            final_densities = metal_densities / mass_densities  # Z
        else:
            raise ValueError("{} is not implemented yet.".format(quantity))

        # then set the attributes. The key used to store the data is needed
        key = "{}_kde_{}D".format(quantity.lower(), dimension)
        self.kde_radii[key] = full_radii
        self.kde_densities[key] = final_densities

        # store the binned radii too, since I will be using those
        bin_radii = utils.bin_values(full_radii, num_azimuthal)
        bin_density = utils.bin_values(final_densities, num_azimuthal)

        self.kde_radii_smoothed[key] = bin_radii
        self.kde_densities_smoothed[key] = bin_density

    def setup_nsc_object(self):
        """
        This creates tje object that will be used to fit the profile to
        determine where the NSC is.
        """
        # create the bins. We want them evenly space in log space from 1pc to
        # 1000 pc. There will be one central bin from zero to one parsec, but
        # this won't work for the logspace, so we put it in by hand. We have
        # 31 edges, which makes 30 bins.
        # then we parse those into bins to be used in in the inner and outer
        inner_bins = np.concatenate([[0], np.logspace(0, 2, 20)])
        outer_bins = np.logspace(2, 3, 11)
        # then we can create these profiles if needed.
        if self.integrated_kde_densities is None:
            self.integrated_kde_profile(inner_bins)
        if self.binned_densities is None:
            self.histogram_profile(outer_bins)
        # then concatenate the profiles together
        fit_radii = np.concatenate([self.integrated_kde_radii,
                                    self.binned_radii])
        fit_densities = np.concatenate([self.integrated_kde_densities,
                                        self.binned_densities])

        # then do the heavy work in the NSC class
        self.nsc = nsc_structure.NscStructure(fit_radii, fit_densities)

    def find_nsc_radius(self):
        """
        Finds the radius of the NSC, using the KDE profile and the associated
        functionality in the NscStructure class.

        :return:
        """
        if self.nsc is None:
            self.setup_nsc_object()

        if self.nsc.nsc_radius is None:
            self.nsc_radius = None
            return
        # if not none we continue
        self.nsc_radius = self.nsc.nsc_radius * yt.units.pc
        self.nsc_radius_err = self.nsc.nsc_radius_err * yt.units.pc

        # then create a new disk object that entirely contains the NSC. I choose
        # 10 times the radius so that we are sure to include everything inside,
        # since yt selects based on cells, not particles.
        self.add_disk(normal=self.disk_kde._norm_vec,
                      disk_height=10 * self.nsc_radius,
                      disk_radius=10 * self.nsc_radius, disk_type="nsc")

        # then get the indices of the stars actually in the NSC
        if self.nsc_idx_j_sphere is None:  # checks for read in
            radius_key = ('STAR', 'particle_position_spherical_radius')
            self.nsc_idx_disk_kde = np.where(self.disk_kde[radius_key] <
                                             self.nsc_radius)[0]
            self.nsc_idx_disk_nsc = np.where(self.disk_nsc[radius_key] <
                                             self.nsc_radius)[0]
            self.nsc_idx_j_sphere = np.where(self.j_sphere[radius_key] <
                                             self.nsc_radius)[0]

        # The same number of stars should be in the NSC to matter what
        # container is being used.
        if not len(self.nsc_idx_disk_nsc) == len(self.nsc_idx_j_sphere):
            raise RuntimeError("NSC disk and sphere don't have same NSC stars.")

        # then check that there are actually stars in the NSC
        if len(self.nsc_idx_disk_nsc) == 0:
            raise RuntimeError("No stars in NSC.")

        # then check if the NSC is dominated by one massive star particle. If
        # so, then our half mass is just an upper limit
        # half mass upper limits
        nsc_star_masses = self.j_sphere[('STAR', "MASS")][self.nsc_idx_j_sphere]
        if utils.max_above_half(nsc_star_masses):  # one star particle dominates
            new_lower_err = self.nsc.r_half_non_parametric  # just the radius.
            # since the errors are a tuple we have to be more clever about
            # setting the lower limit to be a lower limit.
            new_errors = new_lower_err, self.nsc.r_half_non_parametric_err[1]
            self.nsc.r_half_non_parametric_err = new_errors

    def create_axis_ratios_nsc(self):
        """Creates the axis ratios object. """
        self._check_nsc_existence()

        # get the locations of the stars in the NSC. Convert to numpy arrays
        # for speed.
        x = np.array(self.j_sphere[('STAR', 'POSITION_X')].in_units("pc"))
        y = np.array(self.j_sphere[('STAR', 'POSITION_Y')].in_units("pc"))
        z = np.array(self.j_sphere[('STAR', 'POSITION_Z')].in_units("pc"))
        mass = np.array(self.j_sphere[('STAR', "MASS")].in_units("msun"))
        # then get just the NSC ones, and subtract off the center
        x = x[self.nsc_idx_j_sphere] - self.center[0].in_units("pc").value
        y = y[self.nsc_idx_j_sphere] - self.center[1].in_units("pc").value
        z = z[self.nsc_idx_j_sphere] - self.center[2].in_units("pc").value
        mass = mass[self.nsc_idx_j_sphere]

        self.nsc_axis_ratios = nsc_structure.AxisRatios(x, y, z, mass)

    def create_axis_ratios_gal(self):
        """
        Create axis ratios for the whole galaxy. Uses the j radius passed in
        at the very beginning.
        """

        key = "particle_position_relative_{}"
        x = np.array(self.j_sphere[('STAR', key.format("x"))].to("pc").value)
        y = np.array(self.j_sphere[('STAR', key.format("y"))].to("pc").value)
        z = np.array(self.j_sphere[('STAR', key.format("z"))].to("pc").value)
        mass = np.array(self.j_sphere[('STAR', 'MASS')].to("solMass").value)

        self.gal_axis_ratios = nsc_structure.AxisRatios(x, y, z, mass)

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

        sigma_squared_radial = utils.weighted_variance(vel_rad, masses, ddof=0)
        sigma_squared_rot = utils.weighted_variance(vel_rot, masses, ddof=0)
        sigma_squared_z = utils.weighted_variance(vel_z, masses, ddof=0)

        sigma_radial = np.sqrt(sigma_squared_radial)
        sigma_rot = np.sqrt(sigma_squared_rot)
        sigma_z = np.sqrt(sigma_squared_z)

        self.nsc_sigma_radial = sigma_radial
        self.nsc_sigma_rot = sigma_rot
        self.nsc_sigma_z = sigma_z

        self.nsc_3d_sigma = utils.sum_in_quadrature(sigma_z, sigma_rot,
                                                    sigma_radial)

    def nsc_dispersion_eigenvectors(self):
        """Calculate the dispersion along each of the eigenvalues of the
           shape matrix of the NSC. These should line up with the axis ratios
           if they are not due to rotation. """

        # need to check for NSC first
        self._check_nsc_existence()

        # first get the vectors that will will project along.
        a_vec = self.nsc_axis_ratios.a_vec
        b_vec = self.nsc_axis_ratios.b_vec
        c_vec = self.nsc_axis_ratios.c_vec
        # if there are too few particles to do anything, then the eigenvector
        # will be None. If we can't do this, then just set the dispersions
        # to zero.
        if a_vec is None:
            self.nsc_disp_along_a = 0 * yt.units.km / yt.units.second
            self.nsc_disp_along_b = 0 * yt.units.km / yt.units.second
            self.nsc_disp_along_c = 0 * yt.units.km / yt.units.second
            return

        # then get the velocities in all directions.
        key = "particle_velocity_{}"
        v_x_old = self.j_sphere[('STAR', key.format("x"))].to("km/s").value
        v_y_old = self.j_sphere[('STAR', key.format("y"))].to("km/s").value
        v_z_old = self.j_sphere[('STAR', key.format("z"))].to("km/s").value
        mass = self.j_sphere[('STAR', 'MASS')].to("solMass").value

        # then restrict down to the NSC
        v_x_old = v_x_old[self.nsc_idx_j_sphere]
        v_y_old = v_y_old[self.nsc_idx_j_sphere]
        v_z_old = v_z_old[self.nsc_idx_j_sphere]
        mass = mass[self.nsc_idx_j_sphere]

        # then transform the positions to be in the new coordinate system.
        v_as, v_bs, v_cs = [], [], []
        for vx, vy, vz in zip(v_x_old, v_y_old, v_z_old):
            loc = utils.transform_coords([vx, vy, vz], a_vec, b_vec, c_vec)
            v_as.append(loc[0])
            v_bs.append(loc[1])
            v_cs.append(loc[2])

        # add back the units
        v_as = np.array(v_as) * yt.units.km / yt.units.second
        v_bs = np.array(v_bs) * yt.units.km / yt.units.second
        v_cs = np.array(v_cs) * yt.units.km / yt.units.second

        # then we can easily get the dispersion along each eigenvalue.
        sigma_a = np.sqrt(utils.weighted_variance(v_as, mass, ddof=0))
        sigma_b = np.sqrt(utils.weighted_variance(v_bs, mass, ddof=0))
        sigma_c = np.sqrt(utils.weighted_variance(v_cs, mass, ddof=0))

        # then save those
        self.nsc_disp_along_a = sigma_a
        self.nsc_disp_along_b = sigma_b
        self.nsc_disp_along_c = sigma_c

    def create_abundances(self):
        """Creates the abundance objects, which handle all the elemental
        abundance stuff, like [Z/H], [Fe/H], and [X/Fe].

        One object is created for the NSC (self.nsc_abundances), and one for
        the whole galaxy (self.gal_abundances)."""

        self._check_nsc_existence()  # need an NSC

        # we need masses, Z_Ia, and Z_II. I can convert these to arrays to
        # help speed, since the units don't matter in the metallicity
        # calculations. The masses to need to be in code masses, though, since
        # the metallicity dispersion values are in code masses, even though
        # yt thinks they are unitless.
        mass = np.array(self.j_sphere[('STAR', 'MASS')].in_units("code_mass"))
        m_i = np.array(self.j_sphere[('STAR', 'INITIAL_MASS')].to("code_mass"))
        z_Ia = np.array(self.j_sphere[('STAR', 'METALLICITY_SNIa')])
        z_II = np.array(self.j_sphere[('STAR', 'METALLICITY_SNII')])
        z_disp = np.array(self.j_sphere[('STAR', 'METAL_DISPERSION')])

        # the objects can then be created, where for the NSC we select only
        # the objects in the NSC
        self.gal_abundances = nsc_abundances.NSC_Abundances(mass, z_Ia, z_II,
                                                            z_disp, m_i)
        self.nsc_abundances = nsc_abundances.NSC_Abundances(mass[self.nsc_idx_j_sphere],
                                                            z_Ia[self.nsc_idx_j_sphere],
                                                            z_II[self.nsc_idx_j_sphere],
                                                            z_disp[self.nsc_idx_j_sphere],
                                                            m_i[self.nsc_idx_j_sphere])

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

    def histogram_profile(self, bin_edges):
        """
        Create a radial profile by just binning the star particles.

        This doesn't use any fancy KDE, just does a weighted binning, where
        each star particle contributes its mass.

        :param bin_edges: List of bin edges to be used. MUST be in parsecs.
        :return: None, but sets the `binned_densities` and `binned_radii`
                 object attributes.
        """
        # error checking
        if self.disk_whole is None:
            raise RuntimeError("Need to add disk before doing histogram "
                               "profile.")
        utils.test_iterable(bin_edges, "bin_edges")
        if not np.all(np.array(bin_edges) >= 0):
            raise ValueError("All radii need to be positive")

        # first get the values. We bin in radius, and weight by mass.
        container = self.disk_whole  # contains the whole galaxy.
        r = container[('STAR', 'particle_position_cylindrical_radius')]
        r = np.array(r.in_units("pc"))
        star_masses = np.array(container[('STAR', 'MASS')].in_units("msun"))

        # then do the actual binning.
        bin_masses, bin_edges = np.histogram(r, bins=bin_edges,
                                             weights=star_masses)

        # then get the area of each bin
        areas = [utils.annulus_area(bin_edges[idx], bin_edges[idx + 1])
                 for idx in range(len(bin_edges) - 1)]
        # then turn the bin edges to bin_centers.
        centers = [np.mean([bin_edges[idx], bin_edges[idx + 1]])
                   for idx in range(len(bin_edges) - 1)]

        self.binned_radii = centers
        self.binned_densities = bin_masses / np.array(areas)

    def integrated_kde_profile(self, bin_edges):
        """
        Create a radial profile by integrating the 2D KDE density.

        This will use the function in utils that calculates the surface mass
        density in an annulus. This will do Monte Carlo integration within that
        annulus to get the mass, then divide by the area to get surface
        density.

        :param bin_edges: List of bin edges to be used. We will integrate from
                          one bin to another when doing the profile. MUST be in
                          parsecs.
        :return: None, but sets the `integrated_kde_densities` and
                 `integrated_kde_radii` object attributes.
        """
        # error checking
        if self.disk_kde is None:
            raise RuntimeError("Need to add disk before doing histogram "
                               "profile.")
        utils.test_iterable(bin_edges, "bin_edges")
        if not np.all(np.array(bin_edges) >= 0):
            raise ValueError("All radii need to be positive")

        # the smoothing kernel we will use in the KDE process is half the size
        # of the smallest cell in the sphere when we are inside 12pc, but
        # twice this when outside of that distance.
        inner_kernel = float(self.kernel_size.in_units("pc").value)
        outer_kernel = inner_kernel * 2
        break_radius = inner_kernel * 4

        # then do the actual profile creation.
        self.integrated_kde_densities = []
        # iterate through the bin edges, which will be used for the inner and
        # outer radii of integration
        # create the KDE objects if needed
        if self._star_kde_mass_2d is None:
            self._star_kde_mass_2d = self._create_kde_object(2, "mass")
        # then use it.
        density_integrand = self._star_kde_mass_2d.density
        for lower_idx in range(len(bin_edges) - 1):
            # get the radii to integrate over
            lower_radius = bin_edges[lower_idx]
            upper_radius = bin_edges[lower_idx + 1]

            kwargs = {"inner_kernel": inner_kernel,
                      "break_radius": break_radius,
                      "outer_kernel": outer_kernel}
            dens = utils.surface_density_annulus(density_func=density_integrand,
                                                 radius_a=lower_radius,
                                                 radius_b=upper_radius,
                                                 error_tolerance=0.01,
                                                 density_func_kwargs=kwargs)
            self.integrated_kde_densities.append(dens)

        # then turn the bin edges to bin_centers.
        centers = [np.mean([bin_edges[idx], bin_edges[idx + 1]])
                   for idx in range(len(bin_edges) - 1)]

        self.integrated_kde_radii = centers

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
        _write_single_item(file_obj, self.j_radius.in_units("pc"), "j_radius",
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
            _write_single_item(file_obj, self.nsc_idx_j_sphere,
                               "nsc_idx_j_sphere", multiple=True)
            _write_single_item(file_obj, self.nsc_idx_disk_nsc,
                               "nsc_idx_disk_nsc", multiple=True)
            # same with the rotational stuff, which requires access to the disk obj
            # _write_single_item(file_obj, self.mean_rot_vel, "mean_rot_vel",
            #                    multiple=False, units=True)
            # _write_single_item(file_obj, self.nsc_3d_sigma, "nsc_3d_sigma",
            #                    multiple=False, units=True)
            # _write_single_item(file_obj, self.nsc_disp_along_a,
            #                    "nsc_disp_along_a", multiple=False, units=True)
            # _write_single_item(file_obj, self.nsc_disp_along_b,
            #                    "nsc_disp_along_b", multiple=False, units=True)
            # _write_single_item(file_obj, self.nsc_disp_along_c,
            #                    "nsc_disp_along_c", multiple=False, units=True)
            # then the actual radius and half mass radius, which take a while
            # to calculate
            _write_single_item(file_obj, self.nsc_radius, "r_NSC",
                               multiple=False, units=True)
            _write_single_item(file_obj, self.nsc_radius_err, "r_NSC_err",
                               multiple=True, units=True)
            # _write_single_item(file_obj, self.half_mass_radius, "r_half",
            #                    multiple=False, units=False)
            # _write_single_item(file_obj, self.half_mass_radius_errs,
            #                    "r_half_err", multiple=True, units=False)

        else:
            _write_single_item(file_obj, None, "disk_nsc_radius")
            _write_single_item(file_obj, None, "disk_nsc_height")
            _write_single_item(file_obj, None, "disk_kde_normal")
            _write_single_item(file_obj, None, "nsc_idx_j_sphere")
            _write_single_item(file_obj, None, "nsc_idx_disk_nsc")
            # _write_single_item(file_obj, None, "mean_rot_vel")
            # _write_single_item(file_obj, None, "nsc_3d_sigma")
            # _write_single_item(file_obj, None, "nsc_disp_along_a")
            # _write_single_item(file_obj, None, "nsc_disp_along_b")
            # _write_single_item(file_obj, None, "nsc_disp_along_c")
            _write_single_item(file_obj, None, "r_NSC")
            _write_single_item(file_obj, None, "r_NSC_err")
            # _write_single_item(file_obj, None, "r_half")
            # _write_single_item(file_obj, None, "r_half_err")

        # then the binned radii too
        _write_single_item(file_obj, self.binned_radii,
                           "binned_radii", multiple=True)
        _write_single_item(file_obj, self.binned_densities,
                           "binned_densities", multiple=True)
        _write_single_item(file_obj, self.integrated_kde_radii,
                           "integrated_kde_radii", multiple=True)
        _write_single_item(file_obj, self.integrated_kde_densities,
                           "integrated_kde_densities", multiple=True)

        # then all we need are the KDE values. Only write the smoothed ones,
        # since those are what we really use anyway.
        # for key in self.kde_radii:
        #     _write_single_item(file_obj, self.kde_radii[key],
        #                        "radii_{}".format(key), multiple=True)
        #     _write_single_item(file_obj, self.kde_densities[key],
        #                       "density_{}".format(key), multiple=True)
        for key in self.kde_radii_smoothed:
            _write_single_item(file_obj, self.kde_radii_smoothed[key],
                               "smoothed_radii_{}".format(key), multiple=True)
            _write_single_item(file_obj, self.kde_densities_smoothed[key],
                               "smoothed_densities_{}".format(key),
                               multiple=True)

        file_obj.write("end_of_galaxy\n")  # tag so later read-in is easier

    def mini_write(self, file_obj):
        """Writes the galaxy object to a file, to be read in later.

        This is done to save time, since the KDE process can be expensive and
        takes a while. This writes out the essential info to be plotted, so it
        can be read by the minigal class to make things even less expensive.

        :param file_obj: already opened file object that is ready to be
                         written to.
        :returns: None, but writes to the file.
        """
        file_obj.write("\n")
        file_obj.write("new_galaxy_here\n")  # tag so later read-in is easier

        #ID
        _write_single_item(file_obj, self.id, "id")

        # KDE lists These need to be written first because they are always
        # used in the plotting, regardless of whether or not there is a NSC
        # _write_single_item(file_obj, self.kde_radii["mass_kde_2D"],
        #                    "kde_radii_2D", multiple=True)
        # _write_single_item(file_obj, self.kde_densities["mass_kde_2D"],
        #                    "kde_densities_2D", multiple=True)
        if "mass_kde_2D" in self.kde_densities_smoothed:
            _write_single_item(file_obj,
                               self.kde_radii_smoothed["mass_kde_2D"],
                               "kde_radii_2D_smoothed", multiple=True)
            _write_single_item(file_obj,
                               self.kde_densities_smoothed["mass_kde_2D"],
                               "kde_densities_2D_smoothed", multiple=True)
        _write_single_item(file_obj, self.binned_radii,
                           "binned_radii", multiple=True)
        _write_single_item(file_obj, self.binned_densities,
                           "binned_densities", multiple=True)
        _write_single_item(file_obj, self.integrated_kde_radii,
                           "integrated_kde_radii", multiple=True)
        _write_single_item(file_obj, self.integrated_kde_densities,
                           "integrated_kde_densities", multiple=True)

        # fit components
        _write_single_item(file_obj, self.nsc.M_d_parametric,
                           "component_fit_disk_mass")
        _write_single_item(file_obj, self.nsc.M_c_parametric,
                           "component_fit_cluster_mass")
        _write_single_item(file_obj, self.nsc.a_d_parametric,
                           "component_fit_disk_scale_radius")
        _write_single_item(file_obj, self.nsc.a_c_parametric,
                           "component_fit_cluster_scale_radius")

        # kernel size
        _write_single_item(file_obj, self.kernel_size.to("pc").value,
                           "kernel_size")

        # masses. If there is not an NSC, we will exit here.
        mass, mass_err = self.nsc_mass_and_errs()
        if mass is None:
            _write_single_item(file_obj, 0, "nsc_mass")
            file_obj.write("end_of_galaxy\n")  # tag so later read-in is easier
            return

        # parse mas_err properly
        mass_err = [item.to("Msun").value for item in mass_err]
        _write_single_item(file_obj, mass.to("Msun").value, "nsc_mass")
        _write_single_item(file_obj, mass_err, "nsc_mass_err", multiple=True)

        # nsc radius
        _write_single_item(file_obj, self.nsc_radius.to("pc").value,
                           "nsc_radius")
        # nsc half mass radius (already in pc)
        _write_single_item(file_obj, self.half_mass_radius,
                           "nsc_r_half")
        _write_single_item(file_obj, self.half_mass_radius_errs,
                           "nsc_r_half_err", multiple=True)

        # galaxy mass
        gal_mass = self.stellar_mass(radius_cut=None)
        _write_single_item(file_obj, gal_mass.to("Msun").value, "gal_mass")

        # galaxy half mass radius
        gal_half_mass_radius = self.galaxy_half_mass_radius()
        _write_single_item(file_obj, gal_half_mass_radius.to("kpc").value,
                           "gal_r_half")

        # axis ratios
        _write_single_item(file_obj, self.nsc_axis_ratios.b_over_a, "b_over_a")
        _write_single_item(file_obj, self.nsc_axis_ratios.c_over_a, "c_over_a")
        _write_single_item(file_obj, self.nsc_axis_ratios.c_over_b, "c_over_b")
        _write_single_item(file_obj, self.nsc_axis_ratios.ellipticity,
                           "ellipticity")

        # rotation and dispersion
        _write_single_item(file_obj, self.mean_rot_vel.to("km/s").value,
                           "nsc_rot_vel")
        _write_single_item(file_obj, self.nsc_sigma_radial.to("km/s").value,
                           "nsc_sigma_radial")
        _write_single_item(file_obj, self.nsc_sigma_rot.to("km/s").value,
                           "nsc_sigma_rot")
        _write_single_item(file_obj, self.nsc_sigma_z.to("km/s").value,
                           "nsc_sigma_z")
        _write_single_item(file_obj, self.nsc_3d_sigma.to("km/s").value,
                           "nsc_3d_sigma")

        # metallicity
        log_z = self.nsc_abundances.log_z_over_z_sun_total()
        log_z_sd = np.mean(self.nsc_abundances.log_z_err("total"))
        _write_single_item(file_obj, log_z, "log_z_z_sun")
        _write_single_item(file_obj, log_z_sd, "log_z_z_sun_sd_total")

        # [Z/H] errors
        zh_sd_int = self.nsc_abundances.z_on_h_err("internal")
        zh_sd_group = self.nsc_abundances.z_on_h_err("group")
        zh_sd_total = self.nsc_abundances.z_on_h_err("total")
        _write_single_item(file_obj, zh_sd_int, "z_on_h_sd_internal")
        _write_single_item(file_obj, zh_sd_group, "z_on_h_sd_group")
        _write_single_item(file_obj, zh_sd_total, "z_on_h_sd_total")

        # NSC [Fe/H]
        feh = self.nsc_abundances.x_on_h_total("Fe")
        feh_sd_int = self.nsc_abundances.x_on_h_err("Fe", "internal")
        feh_sd_group = self.nsc_abundances.x_on_h_err("Fe", "group")
        feh_sd_total = self.nsc_abundances.x_on_h_err("Fe", "total")
        _write_single_item(file_obj, feh, "fe_on_h")
        _write_single_item(file_obj, feh_sd_int, "fe_on_h_sd_internal")
        _write_single_item(file_obj, feh_sd_group, "fe_on_h_sd_group")
        _write_single_item(file_obj, feh_sd_total, "fe_on_h_sd_total")

        # Gal [Fe/H]
        gal_feh = self.gal_abundances.x_on_h_total("Fe")
        gal_feh_sd = self.gal_abundances.x_on_h_err("total")
        _write_single_item(file_obj, gal_feh, "gal_fe_on_h")
        _write_single_item(file_obj, gal_feh_sd, "gal_fe_on_h_sd_total")

        # [O/Fe]
        ofe = self.nsc_abundances.x_on_fe_total("O")
        ofe_sd = self.nsc_abundances.x_on_fe_err("O", "total")
        _write_single_item(file_obj, ofe, "o_on_fe")
        _write_single_item(file_obj, ofe_sd, "o_on_fe_sd_total")

        # [Mg/Fe]
        mgfe = self.nsc_abundances.x_on_fe_total("Mg")
        mgfe_sd = self.nsc_abundances.x_on_fe_err("Mg", "total")
        _write_single_item(file_obj, mgfe, "mg_on_fe")
        _write_single_item(file_obj, mgfe_sd, "mg_on_fe_sd_total")

        # [Al/Fe]
        alfe = self.nsc_abundances.x_on_fe_total("Al")
        alfe_sd = self.nsc_abundances.x_on_fe_err("Al", "total")
        _write_single_item(file_obj, alfe, "al_on_fe")
        _write_single_item(file_obj, alfe_sd, "al_on_fe_sd_total")

        # individual abundances
        for elt in ["Mg", "Al", "O", "Na"]:
            abund, star_masses = self.nsc_abundances.x_on_fe_individual(elt)
            elt_sd = self.nsc_abundances.x_on_fe_err_individual(elt)

            abund_name = "star_{}_on_fe".format(elt.lower())
            sd_name = "star_{}_on_fe_sd".format(elt.lower())

            _write_single_item(file_obj, abund, multiple=True, name=abund_name)
            _write_single_item(file_obj, elt_sd, multiple=True, name=sd_name)

        _write_single_item(file_obj, star_masses, "star_masses", multiple=True)

        # SFH time
        _write_single_item(file_obj, self.sfh_time, "sfh_time")



        # # fraction of mass within r_half
        # nsc_mass = self.stellar_mass(self.nsc_radius, spherical=False)
        # half_mass = self.stellar_mass(self.half_mass_radius*yt.units.pc,
        #                               spherical=False)
        # fraction = (half_mass / nsc_mass).value
        # _write_single_item(file_obj, fraction, "mass_fraction", units=False)

        file_obj.write("end_of_galaxy\n")  # tag so later read-in is easier

    def _sort_mass_and_birth_nsc(self):
        """
        Sorts the stars in the NSC by their birth time.

        This gets the mass and birth time of the stars in the NSC, then sorts both
        of those arrays by the birth time.

        :return: two arrays, for birth times (in Myr) and masses (in solar massees)
        """
        # get the original arrays
        birth_times = np.array(self.j_sphere[('STAR', 'creation_time')].to("Myr"))
        masses = np.array(self.j_sphere[('STAR', 'MASS')].to("Msun"))

        # just get the NSC
        birth_times = birth_times[self.nsc_idx_j_sphere]
        masses = masses[self.nsc_idx_j_sphere]

        # sort by birth times
        sort_idx = np.argsort(birth_times)
        birth_times = birth_times[sort_idx]
        masses = masses[sort_idx]

        # normalize time to zero
        birth_times -= min(birth_times)

        return birth_times, masses

    def _timescale(self, cumulative_mass, birth_times, min_level, max_level):
        """
        Calculate the timescale for star formation.

        This is defined as the time it took to go from `min_level` of the final mass
        to `max_level`. Typical values are 0.01 and 0.91. I use this extra 1% to
        ignore stars that formed at very early times and just happened to be in the
        NSC, when they aren't really connected.

        :param cumulative_mass: Array with the cumulative mass formation history.
        :param birth_times: Array with the times that match with cumulative_mass.
        :param min_level: Level of mass formation to start the clock.
        :param max_level: Level of mass formation to end the clock.
        :return: Time to assemble this mass, in Myr.
        """
        min_idx = np.argmin(np.abs(cumulative_mass - min_level))
        max_idx = np.argmin(np.abs(cumulative_mass - max_level))

        return birth_times[max_idx] - birth_times[min_idx]

    def cumulative_sfh_nsc(self):
        """
        Find the time required to assemble 90% of the mass of the cluster.

        :return:
        """
        if self.nsc_radius is None:
            return
        birth_times_sorted, masses = self._sort_mass_and_birth_nsc()

        from yt import YTArray

        # average_age = self.j_sphere[("STAR", "AVERAGE_AGE")][self.nsc_idx_j_sphere]
        creation_time = self.j_sphere[("STAR", "creation_time")][self.nsc_idx_j_sphere]
        termination_time = self.j_sphere[("STAR", "TERMINATION_TIME")][self.nsc_idx_j_sphere]

        time = YTArray(self.ds._handle.tphys_from_tcode_array(termination_time) / 1e6, "Myr") - creation_time.in_units("Myr")
        max_time = np.max(time.to("Myr").value)
        # get the fraction of mass that has formed as a function of time
        # the masses are sorted by their age, which is why this works.
        total_mass = np.sum(masses)
        fractional_masses = masses / total_mass
        cumulative_mass = np.cumsum(fractional_masses)

        # then find the timescale for 90% formation. We try many combinations
        # that allow for 90% formation, then pick the one that has the shortest
        #time.
        min_timescale = 10**99
        for min_level in np.linspace(0, 0.1, 500):
            max_level = min_level + 0.9
            timescale =  self._timescale(cumulative_mass, birth_times_sorted,
                                         min_level, max_level)

            if timescale < min_timescale:
                min_timescale = timescale

        self.sfh_time = min_timescale + max_time
