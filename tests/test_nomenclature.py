import pytest


from eurec4a_environment import nomenclature as nom
import eurec4a_environment.source_data


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
def test_get_field_by_name(field_name):
    ds = eurec4a_environment.source_data.open_joanne_dataset()

    nom.get_field(ds=ds, field_name=field_name, units=ds[field_name].units)


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
def test_get_field_by_standard_name(field_name):
    ds = eurec4a_environment.source_data.open_joanne_dataset()

    # JOANNE dataset is renaming some variables, so while we have the
    # version with old namings we'll just let this pass
    # TODO: remove this mapping once we've updated the JOANNE dataset in the
    # intake catalog
    if ds.attrs["JOANNE-version"] == "0.5.7-alpha+0.g45fe69d.dirty":
        if field_name == "ta":
            ds = ds.rename(dict(T="ta"))
        elif field_name == "alt":
            ds = ds.rename(dict(height="alt"))
        elif field_name == "theta":
            ds.theta.attrs["standard_name"] = "air_potential_temperature"

    # make a dataset where the variable name is doesn't match
    ds_copy = ds[[field_name]]
    ds_copy = ds[[field_name]].rename({field_name: "_foobar_field_"})

    nom.get_field(ds=ds_copy, field_name=field_name, units=ds[field_name].units)


def test_get_field_missing_no_standard_name():
    field_name = nom.TEMPERATURE
    ds = eurec4a_environment.source_data.open_joanne_dataset()

    # JOANNE dataset is renaming some variables, so while we have the
    # version with old namings we'll just let this pass
    # TODO: remove this mapping once we've updated the JOANNE dataset in the
    # intake catalog
    if ds.attrs["JOANNE-version"] == "0.5.7-alpha+0.g45fe69d.dirty":
        if field_name == "ta":
            ds = ds.rename(dict(T="ta"))
        elif field_name == "alt":
            ds = ds.rename(dict(height="alt"))
        elif field_name == "theta":
            ds.theta.attrs["standard_name"] = "air_potential_temperature"

    # make a dataset where the variable name is doesn't match and the
    # standard_name isn't set
    ds_copy = ds[[field_name]].rename({field_name: "_foobar_field_"})
    del ds_copy["_foobar_field_"].attrs["standard_name"]

    with pytest.raises(nom.FieldMissingException):
        nom.get_field(ds=ds_copy, field_name=field_name, units=ds[field_name].units)


def test_get_field_missing_unknown_standard_name():
    field_name = "_non_existing_field_"
    ds = eurec4a_environment.source_data.open_joanne_dataset()

    with pytest.raises(nom.FieldMissingException):
        nom.get_field(ds=ds, field_name=field_name, units=None)


def test_get_field_multiple_by_standard_name():
    """
    Tests that merging multiple variables on the same dataset works
    """
    field_name = nom.RELATIVE_HUMIDITY
    field_name_copy = field_name + "_copy"
    # XXX: currently if there's a field with correct name then the standard
    # names are ignored, is this ok?
    field_name_renamed = field_name + "_renamed"
    ds = eurec4a_environment.source_data.open_joanne_dataset()
    ds_copy = ds[[]]
    ds_copy[field_name_copy] = ds[field_name].copy()
    ds_copy[field_name_renamed] = ds[field_name].copy()

    da = nom.get_field(ds=ds_copy, field_name=field_name, units=ds[field_name].units)
    assert list(da["var_name"]) == [field_name_copy, field_name_renamed]
