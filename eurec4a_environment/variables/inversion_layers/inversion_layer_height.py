#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr


def calc_inversion_static_stability(
    ds, time="sounding", altitude="height", z_min=1000, z_max=3000
):

    """
    Returns inversion height defined as the height where static stability profiles
    first exceed 0.1 K/hPa between given heights
    Adapted from Bony and Stevens 2018, Measuring Area-Averaged Vertical Motions with Dropsondes, p 772
    To do: combine static stability criterion with one for moisture (i.e. hydrolapse)
    """

    from eurec4a_environment.variables.calc_static_stability import (
        calc_static_stability,
    )

    ds_inversion = ds.sel({altitude: slice(z_min, z_max)})
    da_static_stability = calc_static_stability(ds_inversion)

    inversion_height_stability = np.zeros(len(da_static_stability[time]))
    for i in range(len(da_static_stability[time])):
        stability_sounding = da_static_stability.isel({time: i}).values
        idx_stability = np.argmax(stability_sounding > 0.1)
        inversion_height_stability[i] = da_static_stability[altitude][idx_stability]

    dims = list(da_static_stability.dims)
    # drop the height coord
    del dims[dims.index(altitude)]

    da = xr.DataArray(
        inversion_height_stability,
        dims=dims,
        coords={d: da_static_stability[d] for d in dims},
    )
    da.attrs["long_name"] = "inversion base height, static stability > 0.1 K/hPa"
    da.attrs["units"] = "m"

    return da


def calc_inversion_grad_RH_T(
    ds,
    time="sounding",
    altitude="height",
    temperature="T",
    rh="rh",
    z_min=1000,
    z_max=2500,
):

    """
    Inversion base following Grindinger et al, 1992; Cao et al 2007
    dT/dz > 0 and dRH/dz < 0
    To do: find all inversion bases and heights, not just first inversion base
    """

    ds_inversion = ds.sel({altitude: slice(z_min, z_max)})
    gradient_RH = ds_inversion[rh].differentiate(coord=altitude)
    gradient_T = ds_inversion[temperature].differentiate(coord=altitude)
    inversion_levels = gradient_T[altitude].where(
        (gradient_RH <= 0) & (gradient_T >= 0)
    )

    num_iters = len(inversion_levels[time])
    inv_base_height_gradients = np.zeros(num_iters)
    for i in range(num_iters):
        alt_inv_da = inversion_levels.isel({time: i})
        inversion_alts = alt_inv_da[~np.isnan(alt_inv_da)]
        if inversion_alts[altitude].size == 0:
            print(f"sounding {i} has no inversion below {z_max} m")
            continue
        inv_base_height_gradients[i] = inversion_alts[0]  # find first inversion base
    inv_base_height_gradients[inv_base_height_gradients == 0] = np.nan

    dims = list(ds.dims.keys())
    # drop the height coord
    del dims[dims.index(altitude)]
    da = xr.DataArray(
        inv_base_height_gradients, dims=dims, coords={d: ds[d] for d in dims}
    )
    da.attrs["long_name"] = "inversion base height from RH and T gradient"
    da.attrs["units"] = "m"
    return da

    # sns.distplot(inv_base_height_gradients)

    # plot vertical profiles dRH/dz and dT/dz

    # fig, ax = plt.subplots(1,2,figsize=(20,8))
    # for i in range(5): # len(gradient_RH)
    #     ax[0].plot(gradient_RH.isel({time: i}), gradient_RH[altitude].values,color="lightgrey", linewidth=2,alpha=0.5)
    # ax[0].plot(gradient_RH.mean(dim=time),gradient_RH[altitude].values,linewidth=4, color='black')
    # ax[0].spines['right'].set_visible(False)
    # ax[0].spines['top'].set_visible(False)
    # ax[0].set_xlabel('$\partial$RH/$\partial$z')
    # ax[0].set_ylabel('Altitude / m')
    # ax[0].set_ylim([0,z_max])
    # ax[0].axvline(x=0, color='black')
    # ax[0].set_xlim([ -1, 1])

    # for i in range(5):
    #     ax[1].plot(gradient_T.isel({time: i}), gradient_T[altitude].values,color="lightgrey", linewidth=2,alpha=0.5)
    # ax[1].plot(gradient_T.mean(dim=time),gradient_T[altitude].values,linewidth=4, color='black')
    # ax[1].spines['right'].set_visible(False)
    # ax[1].spines['top'].set_visible(False)
    # ax[1].set_xlabel('$\partial$T/$\partial$z (dz=10m)' )
    # #ax[1].set_ylabel('Altitude / m')
    # ax[1].set_ylim([0,z_max])
    # ax[1].axvline(x=0, color='black')
    # ax[1].set_xlim([-0.03, 0.03])
