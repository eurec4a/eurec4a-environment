import numpy as np
import xarray as xr


from eurec4a_environment.variables import boundary_layer
from eurec4a_environment.constants import cp_d, g


def test_LCL_Bolton():
    z = np.arange(0.0, 3000.0, 10.0)
    timesteps = np.arange(0, 3)
    da_altitude = xr.DataArray(z, attrs=dict(long_name="altitude", units="m"))
    ds = xr.Dataset(coords=dict(alt=da_altitude, timesteps=timesteps))

    # isentropic atmosphere with surface temperature at 300K
    ds["T"] = 300.0 * np.ones_like(ds.alt) - g / cp_d * da_altitude
    ds.T.attrs["units"] = "K"
    ds.T.attrs["long_name"] = "absolute temperature"

    ds["RH"] = 0.95 * np.ones_like(ds.alt)
    ds.RH.attrs["units"] = "1"
    ds.RH.attrs["long_name"] = "relative humidity"

    da_lcl = boundary_layer.lcl.find_LCL_Bolton(ds=ds)
    assert da_lcl.mean() > 0.0
    assert da_lcl.mean() < 2000.0
