#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find mixed layer height as layer over which x(z+1) - x(z) < threshold

Inputs:
    -- ds: Dataset
    -- var: variable name
    -- threshold: in units of variable (i.e. g/kg or K). Maximum difference allowed in one vertical step, dz=10m, i.e. threshold_q = 0.4 g/kg,
        or vertical gradient dq/dz < 0.04 g/kg/m

Outputs: DataArray containing mixed layer height from gradient method

Note that function is quite slow and should be optimized

"""

import numpy as np
import xarray as xr
from scipy.signal import find_peaks
import statsmodels.api as sm
from constants import Rd, Rv, eps
import seaborn as sns

# import eurec4a_environment.source_data
# ds = eurec4a_environment.source_data.open_joanne_dataset()


def calc_from_gradient(
    ds, var, threshold, z_min=200, altitude="height", time="sounding"
):
    def calculateHmix_var(
        density_profile,
        var_profile,
        threshold,
        z_min,
        altitude="height",
        time="sounding",
    ):

        var_diff = 0
        numer = 0
        denom = 0
        var_mix = 0

        k = int(z_min / 10)  # enforce lower limit = 200m. k is index of lower limit

        # var(z) - weighted_mean(z-1) < threshold
        while abs(var_diff) < threshold:

            numer += 0.5 * (
                density_profile[k + 1] * var_profile[k + 1]
                + density_profile[k] * var_profile[k]
            )
            denom += 0.5 * (density_profile[k + 1] + density_profile[k])
            var_mix = numer / denom
            k += 1
            var_diff = (var_profile[k] - var_mix).values

        hmix = var_profile[altitude].values[k]
        return hmix

    # call inner function
    from eurec4a_environment.variables.calc_density import calc_density

    da_density = calc_density(ds)

    hmix_vec = np.zeros(len(ds[var]))
    for i in range((len(ds[var]))):
        density_profile = da_density.isel({time: i})
        var_profile = ds[var].isel({time: i})
        hmix_vec[i] = calculateHmix_var(
            density_profile,
            var_profile,
            threshold,
            z_min,
            altitude="height",
            time="sounding",
        )

    dims = list(ds.dims.keys())
    # drop the height coord
    del dims[dims.index(altitude)]
    da = xr.DataArray(hmix_vec, dims=dims, coords={d: ds[d] for d in dims})
    da.attrs["long_name"] = f"mixed layer height (from gradient method using {var})"
    da.attrs["units"] = "m"

    return da
