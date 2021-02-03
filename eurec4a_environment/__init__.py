import itertools

import xarray as xr

from ._version import get_versions
from .nomenclature import get_field_by_name
from .unit import convert_units

__version__ = get_versions()["version"]
del get_versions


def get_field(ds, name, units=None):
    """
    Get field described by `name` in dataset `ds` and ensure it is in the
    units defined by `units` (if `units` != None). The units definition is any
    unit definition supported by
    [UDUNITS](https://www.unidata.ucar.edu/software/udunits/)
    """
    da = get_field_by_name(ds, name)

    if units is None:
        return da
    else:
        return convert_units(da, units)


def get_fields(ds, *names, **names_and_units):
    """
    Make sub-selection on dataset `ds` by extracting only the variables with
    names and units as defined in **names_and_units. For any fields names where
    the units aren't given the units will not be checked

    If any of the variables are coordinates these will be properly assigned as
    required by xarray
    """
    dataarrays = []
    coords = {}

    all_fields = itertools.chain(
        names_and_units.items(), zip(names, [None] * len(names))
    )

    for (name, units) in all_fields:
        da = get_field(ds=ds, name=name, units=units)
        if name in ds.coords:
            coords[name] = da
        else:
            dataarrays.append(da)

    ds_subset = xr.merge(dataarrays).assign_coords(**coords)
    return ds_subset
