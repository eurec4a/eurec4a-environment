"""
Routines for constructing reference profiles

# TODO: could be nice to refactor this into a separate python package
"""
import numpy as np
import xarray as xr

from eurec4a_environment.constants import cp_d, g, Rd
from eurec4a_environment import nomenclature as nom


def make_height_grid(z_max, dz, n_timesteps=3):
    """
    Make a (space, time)-grid with the height being in fixed intervals (`dz`)
    of height in meters.
    """
    z = np.arange(0.0, z_max, dz)
    da_timesteps = xr.DataArray(np.arange(0, n_timesteps), dims=("timestep"))
    da_altitude = xr.DataArray(
        z,
        dims=(nom.ALTITUDE),
        attrs=dict(standard_name=nom.CF_STANDARD_NAMES[nom.ALTITUDE], units="m"),
    )
    ds = xr.Dataset(coords={nom.ALTITUDE: da_altitude, "timestep": da_timesteps})
    return ds


def make_pressure_grid(p_min, dp, n_timesteps=3, p0=101325.0):
    """
    Make a (space, time)-grid with the height being in fixed intervals (`dp`)
    of pressure in Pa.
    """
    p = np.arange(p0, p_min, -dp)
    da_timesteps = xr.DataArray(np.arange(0, n_timesteps), dims=("timestep"))
    da_altitude = xr.DataArray(
        p,
        dims=(nom.PRESSURE),
        attrs=dict(standard_name=nom.CF_STANDARD_NAMES[nom.PRESSURE], units="Pa"),
    )
    ds = xr.Dataset(coords={nom.PRESSURE: da_altitude, "timestep": da_timesteps})
    return ds


def make_fixed_lapse_rate_dry_profile(
    z_max=3000.0, dz=10.0, dTdz=0.0, T0=300.0, p0=101325.0
):
    ds = make_height_grid(z_max=z_max, dz=dz)

    if type(dTdz) == str:
        if dTdz == "isothermal":
            dTdz = 0.0
        elif dTdz == "isentropic":
            dTdz = -g / cp_d
        else:
            raise NotImplementedError(dTdz)

    # (multiplying by ds.timesteps is just a quick hack to get ds.T have shape
    # of (alt, timestep)
    ds[nom.TEMPERATURE] = T0 - dTdz * ds[nom.ALTITUDE] + 0.0 * ds.timestep
    ds[nom.TEMPERATURE].attrs["units"] = "K"
    ds[nom.TEMPERATURE].attrs["long_name"] = "absolute temperature"
    ds[nom.TEMPERATURE].attrs["standard_name"] = nom.CF_STANDARD_NAMES[nom.TEMPERATURE]

    ds[nom.SPECIFIC_HUMIDITY] = xr.zeros_like(ds[nom.TEMPERATURE])
    ds[nom.SPECIFIC_HUMIDITY].attrs["units"] = "g/kg"
    ds[nom.SPECIFIC_HUMIDITY].attrs["long_name"] = "specific humidity"
    ds[nom.SPECIFIC_HUMIDITY].attrs["standard_name"] = nom.CF_STANDARD_NAMES[
        nom.SPECIFIC_HUMIDITY
    ]

    # calculate density (not yet saved in profile variables as it's not part of
    # nomenclature)
    rho0 = p0 / T0 * 1.0 / Rd
    T = ds[nom.TEMPERATURE]
    z = ds[nom.ALTITUDE]
    if dTdz == 0.0:
        rho = rho0 * np.exp(-z * g / (Rd * T))
    else:
        alpha = g / (Rd * dTdz)
        rho = rho0 * np.power(T0, alpha + 1.0) * np.power(T, -alpha - 1.0)

    p = rho * Rd * T
    ds[nom.PRESSURE] = p
    ds[nom.PRESSURE].attrs["units"] = "Pa"
    ds[nom.PRESSURE].attrs["long_name"] = "Pressure"
    ds[nom.PRESSURE].attrs["standard_name"] = nom.CF_STANDARD_NAMES[nom.PRESSURE]

    return ds
