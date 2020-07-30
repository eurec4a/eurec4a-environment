#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:13:02 2020

"""

#%%
# import os
# import xarray as xr
#
# load JOANNE dropsondes
input_dir = "/Users/annaleaalbright/Dropbox/EUREC4A/Dropsondes/Data/"
fp_dropsondes = os.path.join(
    input_dir, "EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v0.5.7-alpha+0.g45fe69d.dirty.nc"
)
all_sondes = xr.open_dataset(fp_dropsondes)
all_sondes = xr.open_dataset(fp_dropsondes).swap_dims({"sounding": "launch_time"})

#%%

import numpy as np
import xarray as xr
from scipy.signal import find_peaks
import statsmodels.api as sm


def calc_peakRH_linearize(
    ds,
    altitude="height",
    rh="rh",
    time_dim="sounding",
    z_min=200.0,
    z_max=900.0,
    z_min_lin=50.0,
    z_max_lin=400.0,
):
    """
     Calculate maximum in relative humidity (RH) profile that minimizes difference between 
     observed RH profile and RH profile linearized around lower levels (i.e. 200-400m). 
     Assume moisture is well-mixed at these lower levels, such that RH would increase and 
     assume linear profile as an idealization
     Inputs:
         -- ds: dataset 
         -- altitude, rh, time_dim: variable names
         -- z_min and z_max: lower and upper bounds for mixed layer height
         -- z_min_lin and z_max_lin: bounds for linearization of observed RH profile
     Outputs: 
         -- da: datarray containing h_peakRH_linfit
     """
    h_peakRH_linfit = np.zeros(len(ds[rh]))

    dz = int(ds[altitude].diff(dim=altitude)[1])  # dz=10m

    mixed_layer = np.logical_and(
        ds[altitude] >= z_min, ds[altitude] <= 1500
    )  # enforce z_max later
    ml_linfit = np.logical_and(
        ds[altitude] >= z_min_lin, ds[altitude] <= z_max_lin
    )  # for linearized RH profile

    for i in range(len(ds[rh])):

        rh_profile = ds[rh].isel({time_dim: i}).interpolate_na(dim=altitude)
        X_height = rh_profile[ml_linfit][altitude]
        X = sm.add_constant(X_height.values)  # add intercept
        model = sm.OLS(
            rh_profile[ml_linfit].values, X
        ).fit()  # instantiate linear model
        linearized_RH = model.predict(sm.add_constant(ds[altitude]))

        # 'find_peaks' is scipy function
        idx_peaks_RH_raw, _ = find_peaks(rh_profile[mixed_layer])
        # add +X index from mixed layer (ml) lower bound (lb), divided by dz=10m
        add_to_index_ml_lb = int(z_min / dz)
        idx_peaks_RH = idx_peaks_RH_raw + add_to_index_ml_lb
        obs_RH_peaks = rh_profile[idx_peaks_RH]  # RH values at the local maxima
        linearized_RH_values = linearized_RH[
            idx_peaks_RH
        ]  # linearized RH values at the observed RH maxima
        min_diff_RH = min(
            abs(obs_RH_peaks - linearized_RH_values)
        )  # find minimum RH difference between observed peaks and linearized profile
        h_peakRH_linfit[i] = min_diff_RH[altitude]

    # set 'anomalously' high values > z_max to nan
    # ... somewhat arbitrary choice
    h_peakRH_linfit[h_peakRH_linfit > z_max] = np.nan

    dims = list(ds.dims.keys())
    # drop the height coord
    del dims[dims.index(altitude)]
    da = xr.DataArray(h_peakRH_linfit, dims=dims, coords={d: ds[d] for d in dims})
    da.attrs["long_name"] = "mixed layer height (from RH peak and linearization)"
    da.attrs["units"] = "m"
    return da


#%% callfunction

z_peakRH_linfit = calc_peakRH_linearize(
    all_sondes,
    altitude="height",
    rh="rh",
    time_dim="sounding",
    z_min=200.0,
    z_max=900.0,
    z_min_lin=50.0,
    z_max_lin=200.0,
)

#%% basic plots

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(
    context="notebook",
    style="whitegrid",
    palette="deep",
    font="sans-serif",
    font_scale=2,
    color_codes=True,
    rc=None,
)
plt.figure(figsize=(15, 10))
sns.distplot(z_peakRH_linfit)
plt.ylabel("pdf")
plt.xlabel("mixed layer height from RHmax linearize (m)")

plt.figure(figsize=(15, 10))
plt.scatter(np.arange(len(z_peakRH_linfit)), z_peakRH_linfit)
plt.xlabel("sonde number")
plt.ylabel("mixed layer height from RHmax linearize (m)")
