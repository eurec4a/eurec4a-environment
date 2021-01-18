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

    da_p = get_field(ds=ds, name=pres, units="Pa")
    da_temp = get_field(ds=ds, name=temp, units="K")
    da_qv = get_field(ds=ds, name=specific_humidity, units="g/kg")

    da_rv = da_qv / (1 - da_qv)

    da_rho = (da_p) / (
        Rd * (da_temp) * (1 + (da_rv / eps)) / (1 + da_rv)
    )
    da_rho.attrs["long_name"] = "density of air"
    da_rho.attrs["units"] = "kg/m3"

    return da_rho
