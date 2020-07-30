def find_inversion_height_grad_RH(
    ds, altitude="alt", rh="rh", smooth=False, z_min=1500, z_max=4000.0
):

    """
        Returns inversion height defined as the maximum in the vertical gradient of RH
        """
    ds_lowertroposhere = ds.sel({altitude: slice(z_min, z_max)})
    window = 10  # window for convolution
    if smooth:
        RH = (
            ds_lowertroposhere[rh]
            .rolling(alt=window, min_periods=window, center=True)
            .mean(skipna=True)
        )
    else:
        RH = ds_lowertroposhere[rh]

    RHg = RH.differentiate(coord=altitude)
    ix = RHg.argmax(dim=altitude, skipna=True)
    da_z = RHg.alt[ix]
    da_z.attrs["long_name"] = "inversion layer height (from RH gradient)"
    da_z.attrs["units"] = "m"

    return da_z
