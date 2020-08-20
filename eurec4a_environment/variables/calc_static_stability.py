#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr


def calc_static_stability(ds, pres="p", temp="T", theta="theta", altitude="height"):

    """ Equation from: https://www.ncl.ucar.edu/Document/Functions/Contributed/static_stability.shtml
        S = -(T/theta)*d(theta)/dp. From Bluestein (1992), pg 197, eqn 4.3.8
        
        Static stability measures the gravitational resistance of an atmosphere to vertical displacements. 
        It results from fundamental buoyant adjustments, and so it is determined by the vertical stratification 
        of density or potential temperature.
    """

    # !! edit these lines based on units function !!
    # Celsius to Kelvin
    if ds[temp].max().values < 100:
        temp_K = ds[temp] + 273.15
    else:
        temp_K = ds[temp]

    # convert Pa to hPa
    if ds[pres].max().values > 1400:
        ds[pres] = ds[pres] / 100

    static_stability = -(temp_K / ds[theta]) * (
        ds[theta].diff(dim=altitude) / ds[pres].diff(dim=altitude)
    )
    static_stability = static_stability.transpose(
        transpose_coords=True
    )  # same dimension order as original dataset

    dims = list(static_stability.dims)
    da = xr.DataArray(
        static_stability, dims=dims, coords={d: static_stability[d] for d in dims}
    )
    da.attrs["long_name"] = "static stability"
    da.attrs["units"] = "K/hPa"

    return da
