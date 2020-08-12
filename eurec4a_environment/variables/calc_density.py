#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
from ..constants import Rd, Rv, eps
#from constants import Rd, Rv, eps


def calc_density(ds, pres="p", temp="T", specific_humidity="q", altitude="height"):

    # equation: rho = P/(Rd * Tv), where Tv = T(1 + mr/eps)/(1+mr)

    # convert pressure from hPa to Pa
    if ds[pres].max().values < 1200:
        pressure = ds[pres] * 100
    else:
        pressure = ds[pres]
    # convert temperature from Celsius to Kelvin
    if ds[temp].max().values < 100:
        temp_K = ds[temp] + 273.15
    else:
        temp_K = ds[temp]
    # convert specific humidity from g/kg to kg/kg
    if ds[specific_humidity].max().values > 10:
        q = ds[specific_humidity] / 1000
    else:
        q = ds[specific_humidity]

    mixing_ratio = q / (1 - q)

    density = (pressure) / (
        Rd * (temp_K) * (1 + (mixing_ratio / eps)) / (1 + mixing_ratio)
    )
    density = density.transpose(transpose_coords=True)

    dims = list(ds.dims.keys())
    da = xr.DataArray(density, dims=dims, coords={d: ds[d] for d in dims})
    da.attrs["long_name"] = "density of air"
    da.attrs["units"] = "kg/m3"

    return da
