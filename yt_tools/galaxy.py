import yt
import numpy as np

from . import kde
from . import utils

class Galaxy(object):
    def __init__(self, dataset, center, radius):
        """Create a galaxy object at the specified location with the 
        given size. 
        
        :param dataset: yt dataset object that contains this galaxy
        :param center: 3 element array with the center of the galaxy. Must have
                       units.
        :param radius: Radius that will be used to create the sphere object.
                       Must also have units.
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

        # we can then initialize the KDE object, which will be used for the
        # centering process as well as profiles
        # first get the locations in the x, y, z coordinates as well as the
        # mass, which will be the weights in the KDE procedure
        star_x = np.array(self.sphere[('STAR', 'POSITION_X')].in_units("pc"))
        star_y = np.array(self.sphere[('STAR', 'POSITION_Y')].in_units("pc"))
        star_z = np.array(self.sphere[('STAR', 'POSITION_Z')].in_units("pc"))
        masses = np.array(self.sphere[('STAR', 'MASS')].in_units("msun"))

        self.star_kde = kde.KDE([star_x, star_y, star_z], masses)

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
        kernel_size = np.min(self.sphere[("index", "dx")]).in_units("pc").value

        # we can then go ahead with the centering, using the user's desired
        # accuracy
        self.star_kde.centering(kernel_size, accuracy)

        # then get the values
        cen_x = self.star_kde.location_max_x
        cen_y = self.star_kde.location_max_y
        cen_z = self.star_kde.location_max_z
        # those are scalars, so put them into a yt array. The KDE object is
        # the values when the units are in parsecs (see the __init__), so we
        # add those units back. 
        self.center = [cen_x, cen_y, cen_z] * yt.units.pc

        # recreate the sphere that contains the galaxy.
        self.sphere = self.ds.sphere(center=self.center, radius=self.radius)

    def total_stellar_mass(self):
        """Calculate the total stellar mass in this galaxy. 
        
        This sums the masses of the star particles that are inside the radius 
        of the galaxy, and returns the answer in solar masses. """
        return np.sum(self.sphere[('STAR', "MASS")].in_units("msun"))


