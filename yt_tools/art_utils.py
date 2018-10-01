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

def add_species_fields():
    """
    This adds the various species fields that ART has to be full-fledged
    fields with proper units.
    :return: None, but adds fields
    """
    @yt.derived_field(("gas", "H2_density"), units="g/cm**3")
    def _h2_density(field, data):
        return 2 * code_density_to_cgs(('artio', 'RT_HVAR_H2'), data)

    @yt.derived_field(("gas", "HI_density"), units="g/cm**3")
    def _hI_density(field, data):
        return code_density_to_cgs(('artio', 'RT_HVAR_HI'), data)

    @yt.derived_field(("gas", "HII_density"), units="g/cm**3")
    def _hII_density(field, data):
        return code_density_to_cgs(('artio', 'RT_HVAR_HII'), data)

    @yt.derived_field(("gas", "HeI_density"), units="g/cm**3")
    def _heI_density(field, data):
        return 4 * code_density_to_cgs(('artio', 'RT_HVAR_HeI'), data)

    @yt.derived_field(("gas", "HeII_density"), units="g/cm**3")
    def _heII_density(field, data):
        return 4 * code_density_to_cgs(('artio', 'RT_HVAR_HeII'), data)

    @yt.derived_field(("gas", "HeIII_density"), units="g/cm**3")
    def _heIII_density(field, data):
        return 4 * code_density_to_cgs(('artio', 'RT_HVAR_HeIII'), data)