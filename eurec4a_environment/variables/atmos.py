import numpy as np

from .. import get_field
from ..constants import cp_d, Rd
from .. import nomenclature as nom


def potential_temperature(
    ds, temperature=nom.TEMPERATURE, pressure=nom.PRESSURE, p_surf=None,
    vertical_coord=None
):
    """
    Calculate potential temperature from presure and temperature, using either
    `p_surf` as the reference pressure (in hPa) or the maximum pressure along the
    `vertical_coord` coordinate.

    NB: uses heat capacity and gas constant for dry air
    """
    da_temp = get_field(ds, name=temperature, units="K")
    da_p = get_field(ds, name=pressure, units="hPa")

    if not (p_surf or vertical_coord) or (p_surf and vertical_coord):
        raise Exception("Please provide one of `p_surf` or `vertical_coord`")

    if p_surf:
        p_ref = p_surf
        if isinstance(p_surf, xr.DataArray):
            # implement unit conversion here if necessary
            assert p_surf.attrs.get("units") == "hPa"
    else:
        p_ref = da_p.max(dim=vertical_coord)

    da_theta = da_temp*(p_ref/da_p)**(Rd/cp_d)
    da_theta.name = nom.POTENTIAL_TEMPERATURE
    da_theta.attrs['units'] = "K"
    da_theta.attrs['long_name'] = "potential temperature"

    return da_theta
