# Tools to handle some things using the ART code.
import yt

def code_density_to_cgs(field_name, data_obj):
    """
    Get a field with density units that yt does not handle well.

    This is for fields like the various fields used to hold the density of
    different species. These should be in code units, but yt says they are
    dimensionless.

    :param field_name: Name of the field. Ex: `('artio', 'RT_HVAR_HI')`
    :param data_obj: Data object to get the data from.
    :return: Yt array with the data in g/cm^3.
    """
    array = data_obj[field_name]
    # This array will be dimensionless, but should be in code units.
    code_array = data_obj.ds.arr(array.value, "code_mass/code_length**3")
    # then return this in cgs
    return code_array.to("g/cm**3")

def add_species_fields(ds):
    """
    This adds the various species fields that ART has to be full-fledged
    fields with proper units.

    :param ds: Dataset to add these fields to.
    :return: None, but adds fields
    """
    def _h2_density(field, data):
        return 2 * code_density_to_cgs(('artio', 'RT_HVAR_H2'), data)

    def _hI_density(field, data):
        return code_density_to_cgs(('artio', 'RT_HVAR_HI'), data)

    def _hII_density(field, data):
        return code_density_to_cgs(('artio', 'RT_HVAR_HII'), data)

    def _heI_density(field, data):
        return 4 * code_density_to_cgs(('artio', 'RT_HVAR_HeI'), data)

    def _heII_density(field, data):
        return 4 * code_density_to_cgs(('artio', 'RT_HVAR_HeII'), data)

    def _heIII_density(field, data):
        return 4 * code_density_to_cgs(('artio', 'RT_HVAR_HeIII'), data)

    ds.add_field(("gas", "H2_density"), function=_h2_density,
                 units="g/cm**3", sampling_type="cell")
    ds.add_field(("gas", "HI_density"), function=_hI_density,
                 units="g/cm**3", sampling_type="cell")
    ds.add_field(("gas", "HII_density"), function=_hII_density,
                 units="g/cm**3", sampling_type="cell")
    ds.add_field(("gas", "HeI_density"), function=_heI_density,
                 units="g/cm**3", sampling_type="cell")
    ds.add_field(("gas", "HeII_density"), function=_heII_density,
                 units="g/cm**3", sampling_type="cell")
    ds.add_field(("gas", "HeIII_density"), function=_heIII_density,
                 units="g/cm**3", sampling_type="cell")