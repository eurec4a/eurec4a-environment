"""
Functions in here provide "prefixes" for the tests (see pytest prefixes)
"""
import pytest
import numpy as np
import xarray as xr

import eurec4a_environment.source_data
from eurec4a_environment.constants import cp_d, g
from eurec4a_environment import nomenclature as nom


@pytest.fixture
def ds_joanne():
    ds = eurec4a_environment.source_data.open_joanne_dataset()

    # JOANNE dataset is renaming some variables, so while we have the
    # version with old namings we'll just let this pass
    # TODO: remove this mapping once we've updated the JOANNE dataset in the
    # intake catalog
    if ds.attrs["JOANNE-version"] == "0.5.7-alpha+0.g45fe69d.dirty":
        ds = ds.rename(dict(T=nom.TEMPERATURE, height=nom.ALTITUDE))
        ds.theta.attrs["standard_name"] = "air_potential_temperature"

    return ds


@pytest.fixture
def ds_isentropic_test_profiles():
    z = np.arange(0.0, 3000.0, 10.0)
    da_timesteps = xr.DataArray(np.arange(0, 3), dims=("timestep"))
    da_altitude = xr.DataArray(
        z,
        dims=(nom.ALTITUDE),
        attrs=dict(standard_name=nom.CF_STANDARD_NAMES[nom.ALTITUDE], units="m"),
    )
    ds = xr.Dataset(coords={nom.ALTITUDE: da_altitude, "timestep": da_timesteps})

    # isentropic atmosphere with surface temperature at 300K
    # (multiplying by ds.timesteps is just a quick hack to get ds.T have shape
    # of (alt, timestep)
    ds[nom.TEMPERATURE] = 300.0 - g / cp_d * ds[nom.ALTITUDE] + 0.2 * ds.timestep
    ds[nom.TEMPERATURE].attrs["units"] = "K"
    ds[nom.TEMPERATURE].attrs["long_name"] = "absolute temperature"
    ds[nom.TEMPERATURE].attrs["standard_name"] = nom.CF_STANDARD_NAMES[nom.TEMPERATURE]

    ds[nom.RELATIVE_HUMIDITY] = 0.95 * xr.ones_like(ds[nom.TEMPERATURE])
    ds[nom.RELATIVE_HUMIDITY].attrs["units"] = "1"
    ds[nom.RELATIVE_HUMIDITY].attrs["long_name"] = "relative humidity"
    ds[nom.RELATIVE_HUMIDITY].attrs["standard_name"] = nom.CF_STANDARD_NAMES[
        nom.RELATIVE_HUMIDITY
    ]

    return ds
