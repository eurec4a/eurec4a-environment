"""
To ensure that all functions in eurec4a-environment can access the variables
required the module uses a common nomenclature for variable names throughout.
To add a new variable to the nomenclature simply a) add a new constant below
setting the assumed field name and b) add (if available) the CF-conventions
standard name mapping in `CF_STANDARD_NAMES` if the variable has a "standard
name" (See http://cfconventions.org/standard-names.html for the list of
"standard names")
"""
import inspect
import xarray as xr
try:
    import cfunits
    HAS_UDUNITS2 = True
except FileNotFoundError:
    import warnings
    HAS_UDUNITS2 = False


# TODO: update temperature to be called `ta` once JOANNE dataset is released
# which uses this definition
TEMPERATURE = "T"
# TODO: update altitude to be called `alt` once JOANNE dataset is released
# which uses this definition
ALTITUDE = "height"
RELATIVE_HUMIDITY = "rh"
POTENTIAL_TEMPERATURE = "theta"
PRESSURE = "p"

# From CF-conventions: "specific" means per unit mass. Specific humidity is
# the mass fraction of water vapor in (moist) air.
SPECIFIC_HUMIDITY = "q"

# From CF-conventions: Speed is the magnitude of velocity. Wind is defined as
# a two-dimensional (horizontal) air velocity vector, with no vertical
# component. (Vertical motion in the atmosphere has the standard name
# upward_air_velocity.) The wind speed is the magnitude of the wind velocity
WIND_SPEED = "wspd"

# From CF-conventions: "Eastward" indicates a vector component which is
# positive when directed eastward (negative westward). Wind is defined as a
# two-dimensional (horizontal) air velocity vector, with no vertical component.
# (Vertical motion in the atmosphere has the standard name
# upward_air_velocity.)
ZONAL_WIND = "u"

# From CF-conventions: "Northward" indicates a vector component which is
# positive when directed northward (negative southward). Wind is defined as a
# two-dimensional (horizontal) air velocity vector, with no vertical component.
# (Vertical motion in the atmosphere has the standard name
# upward_air_velocity.)
MERIDIONAL_WIND = "v"


CF_STANDARD_NAMES = {
    TEMPERATURE: "air_temperature",
    ALTITUDE: "geopotential_height",
    RELATIVE_HUMIDITY: "relative_humidity",
    POTENTIAL_TEMPERATURE: "air_potential_temperature",
    PRESSURE: "air_pressure",
    SPECIFIC_HUMIDITY: "specific_humidity",
    WIND_SPEED: "wind_speed",
    ZONAL_WIND: "eastward_wind",
    MERIDIONAL_WIND: "northward_wind",
}


def _get_calling_function_name():
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    return calframe[1][3]


class FieldMissingException(Exception):
    pass

class UDUNTS2_MissingException(Exception):
    pass


def get_field(ds, field_name, units=None):
    """
    Get field described by `field_name` in dataset `ds` and ensure it is in the
    units defined by `units` (if `units` != None). The units definition is any
    unit definition supported by
    [UDUNITS](https://www.unidata.ucar.edu/software/udunits/)
    """
    if field_name in ds:
        da = ds[field_name]
    else:

        calling_function_name = _get_calling_function_name()

        if field_name not in CF_STANDARD_NAMES:
            raise FieldMissingException(
                f"Couldn't find the variable `{field_name}` in the provided dataset."
                f" To use {calling_function_name} you need to provide a variable"
                f" with the name `{field_name}`"
            )
        else:
            standard_name = CF_STANDARD_NAMES[field_name]
            matching_dataarrays = {}
            vars_and_coords = list(ds.data_vars) + list(ds.coords)
            for v in vars_and_coords:
                if ds[v].attrs.get("standard_name", None) == standard_name:
                    matching_dataarrays[v] = ds[v]

            if len(matching_dataarrays) == 0:
                raise FieldMissingException(
                    f"Couldn't find the variable `{field_name}` in the provided dataset."
                    f" To use {calling_function_name} you need to provide a variable"
                    f" with the name `{field_name}` or set the `standard_name`"
                    f" attribute to `{standard_name}` for one or more of the variables"
                    " in the dataset"
                )
            elif len(matching_dataarrays) == 1:
                da = list(matching_dataarrays.values())[0]
            else:
                dims = [da.dims for da in matching_dataarrays.values()]
                if len(set(dims)) == 1:
                    # all variables have the same dims, so we can create a
                    # composite data-array out of these
                    var_names = list(matching_dataarrays.keys())
                    var_dataarrays = list(matching_dataarrays.values())

                    # TODO: should we check units here too?
                    da_combined = xr.concat(var_dataarrays, dim="var_name")
                    da_combined.coords["var_name"] = var_names
                    da = da_combined
                else:
                    raise FieldMissingException(
                        "More than one variable was found in the dataset with"
                        f" the standard name `{standard_name}`, but these couldn't"
                        " be merged to a single xarray.DataArray because they"
                        " don't exist in the same coordinate system"
                    )
                    pass

    if units is None:
        return da
    else:
        # ensure that output is in correct units
        if "units" not in da.attrs:
            field_name = da.name
            raise Exception(
                f"Units haven't been set on `{field_name}` field in dataset"
            )
        if da.attrs["units"] == units:
            return da

        if not HAS_UDUNITS2:
            raise UDUNTS2_MissingException(
                "To do correct unit conversion udunits2 is required, without"
                " it no unit conversion will be done. udunits2 can be installed"
                " with conda, `conda install -c conda-forge udunits2` or see"
                " https://stackoverflow.com/a/42387825 for general instructions"
            )

        old_units = cfunits.Units(da.attrs["units"])
        new_units = cfunits.Units(units)
        if old_units == new_units:
            return da
        else:
            values_converted = cfunits.Units.conform(da.values, old_units, new_units)
            attrs = dict(da.attrs)
            attrs["units"] = units
            da_converted = xr.DataArray(
                values_converted, coords=da.coords, dims=da.dims, attrs=attrs
            )
            return da_converted
