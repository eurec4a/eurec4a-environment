import pytest


from eurec4a_environment import get_field
from eurec4a_environment import nomenclature as nom


@pytest.mark.parametrize(
    "field_name",
    [
        nom.TEMPERATURE,
        nom.ALTITUDE,
        nom.POTENTIAL_TEMPERATURE,
        nom.PRESSURE,
        nom.RELATIVE_HUMIDITY,
        nom.SPECIFIC_HUMIDITY,
        nom.WIND_SPEED,
        nom.MERIDIONAL_WIND,
        nom.ZONAL_WIND,
    ],
)
def test_get_field_by_name(ds_joanne, field_name):
    get_field(ds=ds_joanne, name=field_name, units=ds_joanne[field_name].units)


@pytest.mark.parametrize(
    "field_name",
    [
        nom.TEMPERATURE,
        nom.ALTITUDE,
        nom.POTENTIAL_TEMPERATURE,
        nom.PRESSURE,
        nom.RELATIVE_HUMIDITY,
        nom.SPECIFIC_HUMIDITY,
        nom.WIND_SPEED,
        nom.MERIDIONAL_WIND,
        nom.ZONAL_WIND,
    ],
)
def test_get_field_by_standard_name(ds_joanne, field_name):
    # make a dataset where the variable name is doesn't match
    ds_copy = ds_joanne[[field_name]].rename({field_name: "_foobar_field_"})

    get_field(ds=ds_copy, name=field_name, units=ds_joanne[field_name].units)


@pytest.mark.parametrize(
    "cf_name",
    [
        nom.CF_STANDARD_NAMES[nom.TEMPERATURE],
        nom.CF_STANDARD_NAMES[nom.ALTITUDE],
        nom.CF_STANDARD_NAMES[nom.POTENTIAL_TEMPERATURE],
        nom.CF_STANDARD_NAMES[nom.PRESSURE],
        nom.CF_STANDARD_NAMES[nom.RELATIVE_HUMIDITY],
        nom.CF_STANDARD_NAMES[nom.SPECIFIC_HUMIDITY],
        nom.CF_STANDARD_NAMES[nom.WIND_SPEED],
        nom.CF_STANDARD_NAMES[nom.MERIDIONAL_WIND],
        nom.CF_STANDARD_NAMES[nom.ZONAL_WIND],
    ],
)
def test_get_field_by_cf_standard_name(ds_joanne, cf_name):
    get_field(ds=ds_joanne, name=cf_name)


def test_get_field_missing_no_standard_name(ds_joanne):
    field_name = nom.TEMPERATURE

    # make a dataset where the variable name is doesn't match and the
    # standard_name isn't set
    ds_copy = ds_joanne[[field_name]].rename({field_name: "_foobar_field_"})
    del ds_copy["_foobar_field_"].attrs["standard_name"]

    with pytest.raises(nom.FieldMissingException):
        get_field(ds=ds_copy, name=field_name, units=ds_joanne[field_name].units)


def test_get_field_missing_unknown_standard_name(ds_joanne):
    field_name = "_non_existing_field_"

    with pytest.raises(nom.FieldMissingException):
        get_field(ds=ds_joanne, name=field_name, units=None)


def test_get_field_multiple_by_standard_name(ds_joanne):
    """
    Tests that merging multiple variables on the same dataset works
    """
    field_name = nom.RELATIVE_HUMIDITY
    field_name_copy = field_name + "_copy"
    # XXX: currently if there's a field with correct name then the standard
    # names are ignored, is this ok?
    field_name_renamed = field_name + "_renamed"
    ds_copy = ds_joanne[[]]
    ds_copy[field_name_copy] = ds_joanne[field_name].copy()
    ds_copy[field_name_renamed] = ds_joanne[field_name].copy()

    da = get_field(ds=ds_copy, name=field_name, units=ds_joanne[field_name].units)
    assert list(da["var_name"]) == [field_name_copy, field_name_renamed]
