import numpy as np

from ... import get_fields
from ...constants import cp_d, g
from ... import nomenclature as nom
from ..utils import apply_by_column


def find_LCL_Bolton(
    ds,
    temperature=nom.TEMPERATURE,
    rh=nom.RELATIVE_HUMIDITY,
    altitude=nom.ALTITUDE,
    vertical_coord=nom.ALTITUDE,
):
    """
    Calculates distribution of LCL from RH and T at different vertical levels
    returns mean LCL from this distribution

    Anna Lea Albright (refactored by Leif Denby) July 2020
    """

    def _calc_LCL_Bolton(ds_column):
        """
        Returns lifting condensation level (LCL) [m] calculated according to
        Bolton (1980).
        Inputs: da_temperature is temperature in Kelvin
                da_rh is relative humidity (0 - 1, dimensionless)
        Output: LCL in meters
        """
        # the calculation doesn't work where the relative humidity is zero, so
        # we drop those levels
        ds_column = ds_column.where(ds_column[rh] > 0.0, drop=True)
        da_temperature = ds_column[temperature]
        da_rh = ds_column[rh]
        da_altitude = ds_column[altitude]
        assert da_temperature.units == "K"
        assert da_rh.units == "1"
        assert da_altitude.units == "m"

        tlcl = 1.0 / ((1.0 / (da_temperature - 55.0)) - (np.log(da_rh) / 2840.0)) + 55.0
        zlcl = da_altitude - (cp_d * (tlcl - da_temperature) / g)
        mean_zlcl = np.mean(zlcl)
        return mean_zlcl

    ds_selected = get_fields(ds, **{temperature: "K", rh: "1", altitude: "m"})

    da_LCL = apply_by_column(
        ds=ds_selected, fn=_calc_LCL_Bolton, vertical_coord=vertical_coord
    )
    da_LCL.attrs["long_name"] = "lifting condensation level"
    da_LCL.attrs["reference"] = "Bolton 1980"
    da_LCL.attrs["units"] = "m"
    da_LCL.name = find_LCL_Bolton.name
    return da_LCL


find_LCL_Bolton.name = "z_LCL"
