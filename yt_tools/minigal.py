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
    def __init__(self, file_obj):
        """

        :param file_obj:
        """
        # initialize some stuff about the object
        self.value_dict = dict()

        # then read through the file
        self._read(file_obj)

        # Then allow things to be accessed by the dot operator
        for key in self.value_dict:
            setattr(self, key, self.value_dict[key])


    def _read(self, file_obj):
        """Reads a galaxy object from a file.

                The file has to be an already opened file object that is at the location
                of a galaxy object, written by the Galaxy.write() function. This function
                will return a Galaxy object with all the attributes filled in.

                If the file is not in the right spot, a ValueError will be raised.

                :param file_obj: already opened file that is at the location of a new
                                 galaxy object, as described above.
                :returns: Minigal object with the data filled in.
                :rtype: Minigal
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
        this_line = file_obj.readline()
        while this_line.strip() != "end_of_galaxy":
            if this_line.strip() == "":
                continue
            name, value = parse_line(this_line)
            self.value_dict[name] = value
            this_line = file_obj.readline()

