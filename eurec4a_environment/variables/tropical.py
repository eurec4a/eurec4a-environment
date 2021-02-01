import numpy as np
import xarray as xr

from .. import get_field
from ..constants import cp_d, g
from .. import nomenclature as nom
from .utils import apply_by_column


def lower_tropospheric_stability(
    ds,
    pot_temperature=nom.POTENTIAL_TEMPERATURE,
    pressure=nom.PRESSURE,
    vertical_coord=nom.ALTITUDE,
):
    """
    Lower tropospheric stability as defined by Klein & Hartmann 1993 is the difference
    in potential temperature between the surface and at 700hPa

    NB: currently picks the level closest to 700hPa
    TODO: add vertical linear interpolation for more accurate estimate

    reference: https://www.jstor.org/stable/26198436
    """

    def _calc_lts(ds_column):
        # swap from vertical coord so we can index by pressure
        ds_column = ds_column.swap_dims({vertical_coord: nom.PRESSURE})
        p_max = ds_column[nom.PRESSURE].max()
        p_ref = 700.0
        theta_surf = ds_column.sel(p=p_max)[nom.POTENTIAL_TEMPERATURE]
        theta_ref = ds_column.sel(p=p_ref, method="nearest")[nom.POTENTIAL_TEMPERATURE]

        return theta_ref - theta_surf

    da_theta = get_field(ds, name=pot_temperature, units="K")
    da_p = get_field(ds, name=pressure, units="hPa")

    if pressure in ds.coords:
        ds_derived = da_theta.to_dataset()
        ds_derived = ds_derived.assign_coords(**{pressure: da_p})
    else:
        ds_derived = xr.merge([da_theta, da_p])

    da_lts = apply_by_column(ds=ds_derived, vertical_coord=vertical_coord, fn=_calc_lts)
    da_lts.name = "d_theta__lts"
    da_lts.attrs["units"] = "K"
    da_lts.attrs["long_name"] = "lower tropospheric stability"
    da_lts.attrs["definition"] = "Klein & Hartmann 1993"
    return da_lts
