import yt
from yt.analysis_modules.halo_analysis.api import HaloCatalog

def parse_sim_name_scale_factor(name):
    return name[-10:-4]

def find_correct_halo_file(halos_dir, ds):
    a = ds.scale_factor
    with open(halos_dir + "datasets.txt", "r") as datasets:
        for line in datasets:
            if line.startswith("#"):
                continue
            a_line = float(parse_sim_name_scale_factor(line.split()[0]))
            if abs(a_line - a) < 0.001:
                idx = int(line.split()[-1])

    return halos_dir + "halos_{}.0.bin".format(idx)


def make_halo_catalog(halo_file, data_ds):
    # load data
    halos_ds = yt.load(halo_file)

    # create halo catalog object
    hc = HaloCatalog(halos_ds=halos_ds, data_ds=data_ds)
    hc.add_callback("sphere", factor=0.5)
    hc.create(save_halos=True, save_catalog=False)
    return hc


def find_largest_halo(hc):
    max_halo_mass = 0
    max_halo_idx = None
    for idx, halo in enumerate(hc.halo_list):
        # get the halo mass
        this_halo_mass = halo.quantities["particle_mass"]

        # compare to max
        if this_halo_mass > max_halo_mass:
            max_halo_mass = this_halo_mass
            max_halo_idx = idx

    return hc.halo_list[max_halo_idx]

def get_halo_center(halo):
    return halo.data_object.center.in_units("kpc")