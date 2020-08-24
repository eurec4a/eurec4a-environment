#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr

from .. import get_field
from ..constants import Rd, Rv, eps
from .. import nomenclature as nom


def calc_density(
    ds,
    pres=nom.PRESSURE,
    temp=nom.TEMPERATURE,
    specific_humidity=nom.SPECIFIC_HUMIDITY,
    altitude=nom.ALTITUDE,
):
    # equation: rho = P/(Rd * Tv), where Tv = T(1 + mr/eps)/(1+mr)

    pressure = get_field(ds=ds, field_name=pres, units="Pa")
    temp_K = get_field(ds=ds, field_name=temp, units="K")
    q = get_field(ds=ds, field_name=specific_humidity, units="g/kg")

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
