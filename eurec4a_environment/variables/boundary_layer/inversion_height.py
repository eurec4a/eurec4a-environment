import xarray as xr

def find_inversion_height_grad_RH(
    ds, altitude="alt", rh="rh", smoothing_win_size=None, z_min=1500, z_max=4000.0
):

    """
        Returns inversion height defined as the maximum in the vertical gradient of RH
        """
    ds_lowertroposhere = ds.sel({altitude: slice(z_min, z_max)})
    if smoothing_win_size:
        RH = (
            ds_lowertroposhere[rh]
            .rolling(alt=smoothing_win_size, min_periods=smoothing_win_size, center=True)
            .mean(skipna=True)
        )
    else:
        RH = ds_lowertroposhere[rh]

    RHg = RH.differentiate(coord=altitude)
    ix = RHg.argmax(dim=altitude, skipna=True)
    da_z = RHg.isel({ altitude: ix }).alt
    da_z.attrs["long_name"] = "inversion layer height (from RH gradient)"
    da_z.attrs["units"] = "m"

    return da_z
