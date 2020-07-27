import numpy as np

from ...constants import cp_d, g


def find_LCL_Bolton(ds, temperature="T", rh="RH", altitude="alt"):
    """
    Calculates distribution of LCL from RH and T at different vertical levels
    returns mean LCL from this distribution

    Anna Lea Albright (refactored by Leif Denby) July 2020
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

    # Celsius to Kelvin
    if ds[temperature].units == "C":
        da_temperature = ds[temperature] + 273.15
        da_temperature.attrs["units"] = "K"
    else:
        da_temperature = ds[temperature]
    # RH from % to [0,1] value
    if ds[rh].max().values > 2:
        da_rh = ds[rh] / 100
        da_rh.attrs["units"] = "1"
    else:
        da_rh = ds[rh]

    da_altitude = ds[altitude]

    return _calc_LCL_Bolton(
        da_temperature=da_temperature, da_rh=da_rh, da_altitude=da_altitude
    )
