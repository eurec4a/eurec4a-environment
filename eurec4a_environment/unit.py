import xarray as xr

try:
    import cfunits

    HAS_UDUNITS2 = True
except FileNotFoundError:
    HAS_UDUNITS2 = False


class UDUNTS2MissingException(Exception):
    pass


def convert_units(da, units):
    """Convert the data array to the new units

    Args:
        da (xarray.DataArray):
        units (str):

    Returns:
        xarray.DataArray: The input dataset converted to the new units
    """
    if "units" not in da.attrs:
        field_name = da.name
        raise KeyError(f"Units haven't been set on `{field_name}` field in dataset")

    if da.attrs["units"] == units:
        return da

    if not HAS_UDUNITS2:
        raise UDUNTS2MissingException(
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
            values_converted, coords=da.coords, dims=da.dims, attrs=attrs, name=da.name
        )
        return da_converted
