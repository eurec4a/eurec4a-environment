"""
Functions in here provide "prefixes" for the tests (see pytest prefixes)
"""
import pytest
import numpy as np
import xarray as xr

from eurec4a_environment.constants import cp_d, g


@pytest.fixture
def ds_isentropic_test_profiles():
    z = np.arange(0.0, 3000.0, 10.0)
    da_timesteps = xr.DataArray(np.arange(0, 3), dims=("timestep"))
    da_altitude = xr.DataArray(
        z, dims=("alt"), attrs=dict(long_name="altitude", units="m")
    )
    ds = xr.Dataset(coords=dict(alt=da_altitude, timestep=da_timesteps))

    # isentropic atmosphere with surface temperature at 300K
    # (multiplying by ds.timesteps is just a quick hack to get ds.T have shape
    # of (alt, timestep)
    ds["T"] = 300.0 - g / cp_d * ds.alt + 0.2 * ds.timestep
    ds.T.attrs["units"] = "K"
    ds.T.attrs["long_name"] = "absolute temperature"

    ds["RH"] = 0.95 * xr.ones_like(ds.T)
    ds.RH.attrs["units"] = "1"
    ds.RH.attrs["long_name"] = "relative humidity"

    return ds


@pytest.fixture
def ds_test_levels():
    z = np.arange(0.0, 3000.0, 10.0)
    da_sounding = xr.DataArray(np.arange(0, 3), dims=("sounding"))
    da_altitude = xr.DataArray(
        z, dims=("height"), attrs=dict(long_name="altitude", units="m")
    )
    ds = xr.Dataset(coords=dict(height=da_altitude, sounding=da_sounding))

    # isentropic atmosphere with surface temperature at 300K
    # (multiplying by ds.timesteps is just a quick hack to get ds.T have shape
    # of (alt, timestep)
    ds["T"] = 300.0 - g / cp_d * ds.height + 0.2 * ds.sounding
    ds.T.attrs["units"] = "K"
    ds.T.attrs["long_name"] = "absolute temperature"

    ds["RH"] = 0.95 * xr.ones_like(ds.T)
    ds.RH.attrs["units"] = "1"
    ds.RH.attrs["long_name"] = "relative humidity"

    return ds
