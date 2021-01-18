import numpy as np

from ... import get_field
from ...constants import cp_d, g
from ... import nomenclature as nom


def lower_tropospheric_stability(
    ds, temperature=nom.TEMPERATURE, rh=nom.RELATIVE_HUMIDITY, altitude=nom.ALTITUDE
):
    """

    """

    def _calc_LCL_Bolton(da_temperature, da_rh, da_altitude):
        """
        Returns lifting condensation level (LCL) [m] calculated according to
        Bolton (1980).
        Inputs: da_temperature is temperature in Kelvin
                da_rh is relative humidity (0 - 1, dimensionless)
        Output: LCL in meters
        """
        assert da_temperature.units == "K"
        assert da_rh.units == "1"
        assert da_altitude.units == "m"

        tlcl = 1.0 / ((1.0 / (da_temperature - 55.0)) - (np.log(da_rh) / 2840.0)) + 55.0
        zlcl = da_altitude - (cp_d * (tlcl - da_temperature) / g)
        mean_zlcl = np.mean(zlcl)
        return mean_zlcl

    da_temperature = get_field(ds=ds, name=temperature, units="K")
    da_rh = get_field(ds=ds, name=rh, units="1")
    da_altitude = get_field(ds=ds, name=altitude, units="m")

    return _calc_LCL_Bolton(
        da_temperature=da_temperature, da_rh=da_rh, da_altitude=da_altitude
    )


def lts()
