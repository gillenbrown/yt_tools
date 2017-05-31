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

        # create the sphere that contains the galaxy
        self.sphere = self.ds.sphere(center=self.center, radius=self.radius)

        # set the total stellar mass within the sphere
        all_star_masses = self.sphere[('STAR', "MASS")].in_units("msun")
        self.stellar_mass = np.sum(all_star_masses)

