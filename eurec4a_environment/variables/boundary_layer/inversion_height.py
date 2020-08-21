import xarray as xr

from ... import nomenclature as nom


def find_inversion_height_grad_RH(
    ds,
    altitude=nom.ALTITUDE,
    rh=nom.RELATIVE_HUMIDITY,
    smoothing_win_size=None,
    z_min=1500,
    z_max=4000.0,
):
    """
        Returns inversion height defined as the maximum in the vertical gradient of RH
    """

    ds_lowertroposhere = ds.sel({altitude: slice(z_min, z_max)})
    da_rh = nom.get_field(ds=ds_lowertroposhere, field_name=rh)
    da_z = nom.get_field(ds=ds_lowertroposhere, field_name=altitude)

    if smoothing_win_size:
        RH = da_rh.rolling(
            alt=smoothing_win_size, min_periods=smoothing_win_size, center=True
        ).mean(skipna=True)
    else:
        RH = da_rh

    RHg = RH.differentiate(coord=altitude)
    ix = RHg.argmin(dim=altitude, skipna=True)
    da_z = RHg.isel({altitude: ix})[altitude]
    da_z.attrs["long_name"] = "inversion layer height (from RH gradient)"
    da_z.attrs["units"] = da_z.units

    return da_z
