import utils

class EndOfGalsException(Exception):
    pass

def parse_line(line):
    objects = line.split()
    name = objects[0]
    # get rid of colon
    name = name.replace(":", "")
    values = objects[1:]  # will be a list always, even if just one item
    if len(values) == 1:  # if only one item, turn into a scalar
        values = float(values[0])
    else:
        values = [float(val) for val in values]  # turn to floats
    return name, values

class Minigal():
    """
    Class holding a shorted version of the data that is held in the galaxy
    class. This is done to make plotting go quicker, as we don't have to read
    in all the simulation objects, just the data we need.
    """
    def __init__(self, file_obj):
        """
        Initialize the Minigal object from a file.

        :param file_obj: Already opened file object that will have data about
                         this galaxy.
        """
        # initialize some stuff about the object
        self.value_dict = dict()

        # then read through the file
        self._read(file_obj)

        # Then allow things to be accessed by the dot operator
        for key in self.value_dict:
            setattr(self, key, self.value_dict[key])


    def _read(self, file_obj):
        """Reads a mini galaxy object from a file.

        The file has to be an already opened file object that is at the location
        of a galaxy object, written by the Galaxy.mini_write() function.
        This function will fill in the attributes of this instance.

        If the file is not in the right spot, a ValueError will be raised.

        :param file_obj: already opened file that is at the location of a new
                         galaxy object, as described above.
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
                raise EndOfGalsException

        # we are now at the right spot. Each line following this is a single
        # known value that is easy to grab.
        this_line = file_obj.readline()
        while this_line.strip() != "end_of_galaxy":
            if this_line.strip() == "":
                continue
            name, value = parse_line(this_line)
            self.value_dict[name] = value
            this_line = file_obj.readline()

        # there are a couple things I need to make sure are lists
        keys_to_check = ["kde_radii", "kde_densities", "kde_binned_radii",
                         "kde_binned_densities", "nsc_mass_err",
                         "nsc_r_half_err", "star_masses",
                         "star_al_on_fe", "star_mg_on_fe",
                         "star_o_on_fe", "star_na_on_fe"]
        for key in keys_to_check:
            if key in self.value_dict:
                this_value = self.value_dict[key]
                try:
                    utils.test_iterable(this_value, "")
                except TypeError:  # not iterable
                    self.value_dict[key] = [this_value]

