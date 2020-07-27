#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anna Lea Albright
26 July 2020

-- Script to pre-process circle-mean quantities for HALO dropsondes from EUREC4A field campaign
    -- vertical structure: mixed layer, cloud layer, inversion base height
    -- some large-scale external conditions like LTS and EIS, layer-mean wind speed and direction
    
-- edit to include flag for calculating quantities for individual sondes as well (i.e. radiosondes, P3 dropsondes) 
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import seaborn as sns
import glob
import os
import matplotlib.cm as cm
from datetime import datetime
from pytz import timezone
import matplotlib.dates as mdates
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stat
from scipy.signal import find_peaks
import typhon
from scipy.interpolate import interp1d

sns.set(
    context="notebook",
    style="whitegrid",
    palette="deep",
    font="sans-serif",
    font_scale=2,
    color_codes=True,
    rc=None,
)

#%%


def preprocess_circle_means(fp_flight, fp_dropsondes):

    # =============================================================================
    #             load flight partitioning data
    # =============================================================================

    with open(fp_flight) as f:
        flightinfo = yaml.load(f, Loader=yaml.FullLoader)
    circle_time_stamps = [
        (c["start"], c["end"]) for c in flightinfo["segments"] if c["kind"] == "circle"
    ]
    circle_num = np.arange(len(circle_time_stamps))

    # =============================================================================
    #      convert circle start times from UTC to local/Barbados time
    # =============================================================================
    circle_start_list_local = []
    for i in range(len(circle_num)):
        fmt = "%Y-%m-%d %H:%M"
        dt_UTC = circle_time_stamps[i][0]
        # add UTC localization
        dt_UTC = timezone("UTC").localize(dt_UTC)
        # convert from UTC to local time
        dt_Barbados = dt_UTC.astimezone(timezone("America/Barbados"))
        str_local = dt_Barbados.strftime(fmt)
        circle_start_list_local.append(str_local)
        circle_start_time_local = np.asarray(circle_start_list_local)

    # =============================================================================
    #      load dropsonde data for corresponding research flight
    # =============================================================================
    all_sondes = xr.open_dataset(fp_dropsondes).swap_dims({"sounding": "launch_time"})
    day_str = circle_time_stamps[0][0].strftime("%Y-%m-%d")
    sondes_oneday = all_sondes.sel(launch_time=day_str)
    # remove sondes with large numbers of NaNs
    # threshold : require this many non-NA values
    subset_sondes = sondes_oneday.dropna(
        dim="launch_time", subset=["rh"], how="any", thresh=300
    )

    sondes = subset_sondes
    print(repr(len(circle_num)), "circles performed on", repr(day_str))

    # =============================================================================
    #      load measured variables
    # =============================================================================

    theta = sondes["theta"]  # K
    temp = sondes["T"]  # degC

    wspd = sondes["wspd"]  # m/s
    u_wind = sondes["u"]
    v_wind = sondes["v"]
    wdir = sondes["wdir"]  # degrees

    specific_humidity = sondes["q"]  # kg/kg
    RH = sondes["rh"]  # %
    PW = sondes["PW"]  # kg/m2

    pressure = sondes["p"]  # hPa
    alt_vec = temp.height.values

    # =============================================================================
    #      calculate mixing ratio and theta_v
    # =============================================================================

    # define parameters
    Rd = typhon.constants.gas_constant_dry_air  # J/kg/K
    Rv = typhon.constants.gas_constant_water_vapor
    eps = Rd / Rv

    mixing_ratio = specific_humidity / (1 - specific_humidity)  # kg/kg
    theta_v = theta * ((1 + mixing_ratio / eps) / (1 + mixing_ratio))

    # =============================================================================
    #        function to calculate circle-mean values
    # =============================================================================

    def calc_circle_mean_var(var, circle_time_stamps):
        num_circles = len(circle_time_stamps)
        var_circle_mean = np.zeros([num_circles, len(alt_vec)])
        for i in range(num_circles):
            start_time = np.datetime64(np.datetime64(circle_time_stamps[i][0]), "ns")
            end_time = np.datetime64(np.datetime64(circle_time_stamps[i][1]), "ns")
            c_bool = var["launch_time"] > start_time
            c_bool &= var["launch_time"] < end_time
            var_circle_mean[i, :] = var.sel(launch_time=c_bool).mean(
                dim="launch_time", skipna=True
            )

        return var_circle_mean

    # =============================================================================
    #        call function
    # =============================================================================

    mixing_ratio_circle_mean = calc_circle_mean_var(mixing_ratio, circle_time_stamps)
    specific_humidity_circle_mean = calc_circle_mean_var(
        specific_humidity, circle_time_stamps
    )
    rh_circle_mean = calc_circle_mean_var(RH, circle_time_stamps)
    PW_circle_mean = calc_circle_mean_var(PW, circle_time_stamps)

    theta_circle_mean = calc_circle_mean_var(theta, circle_time_stamps)
    temp_circle_mean = calc_circle_mean_var(temp, circle_time_stamps)
    theta_v_circle_mean = calc_circle_mean_var(theta_v, circle_time_stamps)

    wind_circle_mean = calc_circle_mean_var(wspd, circle_time_stamps)
    u_wind_circle_mean = calc_circle_mean_var(u_wind, circle_time_stamps)
    v_wind_circle_mean = calc_circle_mean_var(v_wind, circle_time_stamps)
    wind_dir_circle_mean = calc_circle_mean_var(wdir, circle_time_stamps)

    pres_circle_mean = calc_circle_mean_var(pressure, circle_time_stamps)

    # =============================================================================
    #             construct a xarray Dataset from circle-mean data
    # =============================================================================

    circle_means = xr.Dataset(
        data_vars={
            "mixing_ratio": (("circle_num", "alt"), mixing_ratio_circle_mean),
            "specific_humidity": (("circle_num", "alt"), specific_humidity_circle_mean),
            "RH": (("circle_num", "alt"), rh_circle_mean),
            "PW": (("circle_num", "alt"), PW_circle_mean),
            "theta": (("circle_num", "alt"), theta_circle_mean),
            "temp": (("circle_num", "alt"), temp_circle_mean),
            "theta_v": (("circle_num", "alt"), theta_v_circle_mean),
            "wspd": (("circle_num", "alt"), wind_circle_mean),
            "u_wind": (("circle_num", "alt"), u_wind_circle_mean),
            "v_wind": (("circle_num", "alt"), v_wind_circle_mean),
            "wind_dir": (("circle_num", "alt"), wind_dir_circle_mean),
            "pressure": (("circle_num", "alt"), pres_circle_mean),
            "circle_start_local": (("circle_num"), circle_start_time_local),
        },
        coords={"circle_num": circle_num, "alt": alt_vec},
    )

    # =============================================================================
    #             Density
    # =============================================================================

    def calc_density(ds, pres_str, temp_str, mixing_ratio_str):

        # equation: rho = P/(Rd * Tv), where Tv = T(1 + mr/eps)/(1+mr)

        # convert pressure from hPa to Pa
        if ds[pres_str].max().values < 1200:
            pressure = ds[pres_str] * 100
        else:
            pressure = ds[pres_str]
        # convert temperature from Celsius to Kelvin
        if ds[temp_str].max().values < 100:
            temp_K = ds[temp_str] + 273.15
        else:
            temp_K = ds[temp_str]
        # convert mixing ratio from g/kg to kg/kg
        if ds[mixing_ratio_str].max().values > 10:
            mixing_ratio = ds[mixing_ratio_str] / 1000
        else:
            mixing_ratio = ds[mixing_ratio_str]

        Rd = typhon.constants.gas_constant_dry_air  # J/kg/K
        Rv = typhon.constants.gas_constant_water_vapor
        eps = Rd / Rv
        ds = ds.assign(
            density=(pressure)
            / (Rd * (temp_K) * (1 + (mixing_ratio / eps)) / (1 + mixing_ratio))
        )

        return ds

    circle_means = calc_density(circle_means, "pressure", "temp", "mixing_ratio")

    # =============================================================================
    #            Characterizing vertical structure
    #       mixed layer, LCL, and inversion base height
    # =============================================================================
    """
    Functions to estimate MIXED LAYER height:
    (1) height of maximum RH for each circle-mean profile
        -- 'calc_peak_RH' calculates the local maximum below 900m 
        -- 'calc_peak_RH_linearize' finds local RH maximum closest to the extrapolated RH profile at low levels (i.e. 200-400m)
            -- removes anomalously high values
    (2, 3) zero vertical gradients in mixing ratio and theta_v, given threshold (i.e. 10% of mean difference between surface and cloud layer values)
    
    ---> idea to choose most conservative estimate min(1, 2, 3)
    """

    def calc_peak_RH(ds, rh_str):
        alt_peakRH = np.zeros(len(ds[rh_str]))
        val_peakRH = np.zeros(len(ds[rh_str]))
        alt = ds.alt
        mixed_layer = np.logical_and(alt >= 400, alt <= 900)

        # find RHmax for each sonde
        for i in range(len(ds.circle_num)):
            rh_profile = ds[rh_str].isel(circle_num=i)
            peakRH_ind = np.argmax(rh_profile[mixed_layer])  # index of first maximum
            alt_peakRH[i] = alt[mixed_layer][peakRH_ind]
            val_peakRH[i] = rh_profile.values[mixed_layer][peakRH_ind]

        # assign heights to dataset
        ds["mixed_layer_height_RHmax"] = alt_peakRH
        # ds['RHmax_value'] = val_peakRH

        return ds

    circle_means = calc_peak_RH(circle_means, rh_str="RH")

    def calc_peak_RH_linearize(ds, rh_str):

        alt_peakRH = np.zeros(len(ds[rh_str]))
        val_peakRH = np.zeros(len(ds[rh_str]))
        alt_vec = ds.alt
        lower_bound_mixed_layer = 400
        upper_bound_mixed_layer = 900  # bounds for mixed layer height
        mixed_layer = np.logical_and(
            alt_vec >= lower_bound_mixed_layer, alt_vec <= upper_bound_mixed_layer
        )
        lower_ml = np.logical_and(
            alt_vec >= 200, alt_vec <= 400
        )  # for linearized RH profile
        first_900_m = np.logical_and(alt_vec >= 0, alt_vec <= 900)  # for plotting

        for i in range(len(ds.circle_num)):
            rh_profile = ds[rh_str].isel(circle_num=i)

            # fit linear model to lower mixed layer and plot linearized profile
            X_alt_values = rh_profile[lower_ml].alt.values
            X = sm.add_constant(X_alt_values)  # add intercept
            model = sm.OLS(
                rh_profile[lower_ml].values, X
            ).fit()  # instantiate linear model
            predicted_RH_line = model.predict(sm.add_constant(alt_vec[first_900_m]))

            # find local maxima in RH and altitude of these maxima below 900m
            # choose peak that minimizes horizontal distance btween actual RH value and linearized lower mixed layer RH profile
            # 'find_peaks' is scipy function
            idx_peaks_RH, _ = find_peaks(
                rh_profile[mixed_layer]
            )  # returns indices of local maxima
            # add +X index from mixed layer (ml) lower bound (lb), divided by 10 (step size = 10m)
            add_to_index_ml_lb = int(lower_bound_mixed_layer / 10)
            RH_peaks = rh_profile[
                idx_peaks_RH + add_to_index_ml_lb
            ]  # RH values at the local maxima
            RH_extrapolated = predicted_RH_line[
                idx_peaks_RH + add_to_index_ml_lb
            ]  # extrapolated RH values at the observed, local RH maxima
            difference_peak_extrap = min(
                abs(RH_peaks - RH_extrapolated)
            )  # find minimum horizontal difference (RH peak - RH extrapolated)

            alt_peakRH[
                i
            ] = difference_peak_extrap.alt  # altitude where distance is minimized
            val_peakRH[i] = rh_profile.values[
                int(alt_peakRH[i] / 10)
            ]  # RH value at this altitude, index from height / 10

        # plot?
        if False:

            # if RH units [0,1] convert to [0,100] for plotting
            if ds[rh_str].max().values < 10:
                RH = ds[rh_str] * 100
            else:
                RH = ds[rh_str]

            plt.figure(figsize=(8, 10))
            colors = [
                "navy",
                "deepskyblue",
                "crimson",
                "mediumaquamarine",
                "darkmagenta",
                "goldenrod",
            ]
            for i in range(len(ds.circle_num)):
                rh_profile = RH.isel(circle_num=i)
                # rh_profile = ds[rh_str].rolling(alt=window, min_periods = window, center=True).mean(skipna=True).isel(circle_num=i)

                plt.plot(
                    rh_profile[first_900_m],
                    alt_vec[first_900_m],
                    color=colors[i],
                    alpha=0.6,
                    linewidth=4,
                )
                # fit linear model to lower mixed layer, i.e. 50-300m and plot line
                X_alt_values = rh_profile[lower_ml].alt.values
                X = sm.add_constant(X_alt_values)
                model = sm.OLS(rh_profile[lower_ml].values, X).fit()
                # predicted RH extrapolation from 10-800m
                predicted_RH_line = model.predict(sm.add_constant(alt_vec[first_900_m]))
                plt.plot(
                    predicted_RH_line,
                    alt_vec[first_900_m],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=3,
                )
                plt.gca().spines["right"].set_visible(False)
                plt.gca().spines["top"].set_visible(False)
                plt.xlabel("RH")
                plt.ylabel("Altitude / m")
                plt.ylim([0, 1000])
                plt.xlim([60, 95])

        # assign heights to dataset
        ds["mixed_layer_height_RHmax_2"] = alt_peakRH
        # ds['RHmax_value'] = val_peakRH # save peak RH value as well?

        return ds

    circle_means = calc_peak_RH_linearize(circle_means, rh_str="RH")

    # find mixed layer height from constant vertical gradients
    def find_ml_height_from_gradient(ds, var_str, threshold):
        def calculateHmix_var(density_profile, var_profile, threshold):

            # size = len(sonde.alt)
            var_diff = 0
            numer = 0
            denom = 0
            var_mix = 0

            k = 40  # lower limit = 400m

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

            hmix = var_profile.alt.values[k]
            return hmix

        # call inner function
        hmix_vec = np.zeros(len(ds.density))
        for i in range(len(ds.density)):
            density_profile = ds["density"].isel(circle_num=i)
            var_profile = ds[var_str].isel(circle_num=i)
            hmix_vec[i] = calculateHmix_var(density_profile, var_profile, threshold)

        # assign variable to dataset
        ds[f"mixed_layer_height_grad_{var_str}"] = hmix_vec

        return ds

    # dq/dz
    # thresholds chosen as 10% of difference between mean surface and cloud layer values
    circle_means = find_ml_height_from_gradient(
        circle_means, "mixing_ratio", threshold=0.3 / 1000
    )
    # dthetav/dz
    circle_means = find_ml_height_from_gradient(circle_means, "theta_v", threshold=0.05)

    # =============================================================================
    #     LCL Bolton
    # =============================================================================

    def find_LCL_Bolton(ds, temp_str, rh_str, levels):

        # calculates distribution of LCL from RH and T at different vertical levels
        # returns mean LCL from this distribution
        def calc_LCL_Bolton(temperature, relative_humidity, altitude, levels):
            """
            Returns lifting condensation level (LCL) [m] calculated according to Bolton (1980).
            Inputs: T is temperature in Kelvin
                    RH is relative humidity (0 - 1, dimensionless)
            Output: LCL in meters
            """
            z0 = altitude.values[levels]
            Cp = 1004  # J/kg/K
            g = 9.81  # m/s2
            tlcl = (
                1
                / (
                    (1 / (temperature[levels] - 55))
                    - (np.log(relative_humidity[levels]) / 2840.0)
                )
                + 55
            )
            zlcl = z0 - (Cp * (tlcl - temperature[levels]) / g)
            mean_zlcl = np.mean(zlcl)
            return mean_zlcl

        # Celsius to Kelvin
        if ds[temp_str].max().values < 100:
            temp_K = ds[temp_str] + 273.15
        else:
            temp_K = ds[temp_str]
        # RH from % to [0,1] value
        if ds[rh_str].max().values > 2:
            RH = ds[rh_str] / 100
        else:
            RH = ds[rh_str]
        nt = len(ds.circle_num)  # number of circles
        LCL_Bolton = np.zeros([nt])
        for i in range(nt):
            LCL_Bolton[i] = calc_LCL_Bolton(
                temp_K[i, :], RH[i, :], ds["alt"], levels=levels
            )

        # set anomalously high values to nan
        LCL_Bolton[LCL_Bolton > 1500] = np.nan
        ds["LCL_Bolton"] = LCL_Bolton

        return ds

    # define levels of air masses from which to find mean LCL
    # i.e. 10-200m, 10m intervals, np.arange(1,20)
    levels = np.arange(1, 20)
    circle_means = find_LCL_Bolton(circle_means, "temp", "RH", levels)

    # =============================================================================
    #            Inversion base height
    #   -- static stability > 0.1 K/hPa following Bony and Stevens 2018
    #   -- maximum dRH/dz
    #   -- area where dT/dz > 0 and dRH/dz < 0 following Grindinger et al, 1992; Cao et al 2007

    #  Keep results from both raw and smoothed (100m window) profiles
    # =============================================================================

    # =============================================================================
    #             Static stability
    # =============================================================================

    def calc_static_stability(ds, pres_str, temp_str):

        """ Equation source: https://www.ncl.ucar.edu/Document/Functions/Contributed/static_stability.shtml
              Inputs:
             -- dataset (i.e. circle_means)
             -- string for pressure and temperature
              Outputs:
             -- static stability in K / hPa
        
            Static stability measures the gravitational resistance of an atmosphere to vertical displacements. 
            It results from fundamental buoyant adjustments, and so it is determined by the vertical stratification 
            of density or potential temperature.
        """
        # Celsius to Kelvin
        if ds[temp_str].max().values < 100:
            temp_K = ds[temp_str] + 273.15
        else:
            temp_K = ds[temp_str]

        # convert Pa to hPa
        if ds[pres_str].max().values > 1400:
            ds[pres_str] = ds[pres_str] / 100

        ds["static_stability"] = -(temp_K / ds["theta"]) * (
            ds["theta"].diff(dim="alt") / ds[pres_str].diff(dim="alt")
        )

        # smoothed
        # window = 10 # window for convolution
        # ds['static_stability_smooth'] = ds['static_stability'].rolling(alt=window, min_periods = window, center=True).mean(skipna=True)

        return ds

    circle_means = calc_static_stability(circle_means, "pressure", "temp")

    # search for inversion values between 1.5 - 5km altitude
    max_alt_inv = 5000

    # =============================================================================
    #           from peak in low-level static stability
    # =============================================================================

    def find_inversion_height_static_stability(ds, max_alt):

        """
        Returns inversion height defined as the height where static stability profiles
        first exceed 0.1 K/hPa
        
        """
        static_stability = ds.static_stability[:, 10:]
        alt = static_stability.alt

        window = 10  # window for convolution
        static_stability_smoothed = static_stability.rolling(
            alt=window, min_periods=window, center=True
        ).mean(skipna=True)[:, 10:]

        def calc_inversion_height_static_stability(static_stability, smooth):
            inversion_height_stability = np.zeros(len(static_stability))
            for i in range(len(static_stability)):
                stability_circle = static_stability.isel(circle_num=i).values
                if smooth == False:
                    idx_stability = np.argmax(stability_circle > 0.1)
                else:
                    idx_stability = np.argmax(stability_circle > 0.08)

                inversion_height_stability[i] = alt[idx_stability]
            return inversion_height_stability

        # call function to calculate
        inversion_height_stability = calc_inversion_height_static_stability(
            static_stability, smooth=False
        )
        inversion_height_stability_smoothed = calc_inversion_height_static_stability(
            static_stability_smoothed, smooth=True
        )

        # assign to xarray
        ds = ds.assign(inversion_height_stability=inversion_height_stability)
        ds = ds.assign(
            inversion_height_stability_smoothed=inversion_height_stability_smoothed
        )

        return ds

    # call outer function
    circle_means = find_inversion_height_static_stability(
        circle_means, max_alt=max_alt_inv
    )

    # =============================================================================
    #            maximum dRH/dz (raw or smoothed)
    # =============================================================================

    def find_inversion_height_grad_RH(ds, max_alt):

        """
        Returns inversion height defined as the maximum in the vertical gradient of RH
        """

        window = 10  # window for convolution
        RH_smooth = (
            ds["RH"]
            .rolling(alt=window, min_periods=window, center=True)
            .mean(skipna=True)
        )
        RH = ds["RH"]

        def calc_inversion_height_grad_RH(RH):
            inversion_height_grad_RH = np.zeros(len(RH))
            for i in range(len(RH)):
                # load data
                RH_circle = RH[i, :]
                alt = ds.alt[:]

                # calculate RH gradient
                gradient_RH = np.diff(RH_circle) / np.diff(alt)
                gradient_RH = np.append(gradient_RH, gradient_RH[-1])
                lower_tropo = np.logical_and(
                    circle_means.alt > 1500, circle_means.alt < max_alt
                )
                inversion_ind = np.argmax(
                    gradient_RH[lower_tropo]
                )  # index of first maximum in this domain
                inversion_height_grad_RH[i] = alt[lower_tropo][inversion_ind]
            return inversion_height_grad_RH

        inversion_height_grad_RH = calc_inversion_height_grad_RH(RH)
        inversion_height_grad_RH_smoothed = calc_inversion_height_grad_RH(RH_smooth)

        ds = ds.assign(inversion_height_grad_RH=inversion_height_grad_RH)
        ds = ds.assign(
            inversion_height_grad_RH_smoothed=inversion_height_grad_RH_smoothed
        )

        return ds

    # call function
    circle_means = find_inversion_height_grad_RH(circle_means, max_alt=max_alt_inv)

    # =============================================================================
    #          inversion base and top, following Grindinger et al, 1992; Cao et al 2007
    #          dT/dz > 0 and dRH/dz < 0
    # =============================================================================

    # needs some editing to find all inversions, but just first inversion
    def find_inversion_height_grad_RH_and_T(ds, max_alt):

        RH_var = ds["RH"]
        T_var = ds["temp"]

        gradient_RH = RH_var.differentiate(coord="alt")
        gradient_T = T_var.differentiate(coord="alt")
        inversion_levels = gradient_T.alt.where((gradient_RH <= 0) & (gradient_T >= 0))
        num_iters = len(inversion_levels.circle_num)
        inv_base_height_gradients = np.zeros(num_iters)

        for i in range(num_iters):
            alt_inv_da = inversion_levels.isel(circle_num=i)
            idx_base = np.argmax(
                inversion_levels.isel(circle_num=i) < 3000
            )  # first maximum below 3 km
            inv_base_height_gradients[i] = alt_inv_da[idx_base].alt.values

        ds = ds.assign(inversion_height_grad_RH_T=inv_base_height_gradients)

        # plot?
        if 0:
            fig, ax = plt.subplots(1, 2, figsize=(20, 8))

            for i in range(len(gradient_RH)):
                ax[0].plot(
                    gradient_RH.isel(circle_num=i),
                    gradient_RH.alt.values,
                    color="lightgrey",
                    linewidth=2,
                    alpha=0.5,
                )
            ax[0].plot(
                gradient_RH.mean(dim="circle_num"),
                gradient_RH.alt.values,
                linewidth=4,
                color="black",
            )
            ax[0].spines["right"].set_visible(False)
            ax[0].spines["top"].set_visible(False)
            ax[0].set_xlabel("$\partial$RH/$\partial$z")
            ax[0].set_ylabel("Altitude / m")
            ax[0].set_ylim([0, 5000])
            ax[0].axvline(x=0, color="black")
            ax[0].set_xlim([-0.004, 0.004])

            for i in range(len(gradient_T)):
                ax[1].plot(
                    gradient_T.isel(circle_num=i),
                    gradient_T.alt.values,
                    color="lightgrey",
                    linewidth=2,
                    alpha=0.5,
                )
            ax[1].plot(
                gradient_T.mean(dim="circle_num"),
                gradient_T.alt.values,
                linewidth=4,
                color="black",
            )
            ax[1].spines["right"].set_visible(False)
            ax[1].spines["top"].set_visible(False)
            ax[1].set_xlabel("$\partial$T/$\partial$z (dz=10m)")
            # ax[1].set_ylabel('Altitude / m')
            ax[1].set_ylim([0, 5000])
            ax[1].axvline(x=0, color="black")
            ax[1].set_xlim([-0.03, 0.03])

            inversion_levels.plot(figsize=(10, 7))
            plt.ylim([1200, 5000])

            # first peak
            plt.figure(figsize=(12, 8))
            plt.plot(
                range(len(inv_base_height_gradients)),
                inv_base_height_gradients,
                linewidth=4,
                color="black",
            )
            plt.xlim(
                [
                    range(len(inv_base_height_gradients))[0],
                    range(len(inv_base_height_gradients))[-1],
                ]
            )
            # plt.xlim([x1, x2])
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["top"].set_visible(False)
            plt.xlabel("circle-mean")
            plt.ylabel("Estimate for height of inversion base / m")
            plt.ylim([1000, 3000])

        return ds

    # call function
    circle_means = find_inversion_height_grad_RH_and_T(
        circle_means, max_alt=max_alt_inv
    )

    # draft to find all inversion levels (not just first inversion)
    # step forward until difference > 10, split
    # inversion_levels.dropna(dim="alt", thresh=2).sel(circle_num=0).alt
    # inversion_levels.dropna(dim="alt", thresh=2).sel(circle_num=0).alt.diff(dim="alt")
    # inversion_levels.dropna(dim="alt", thresh=2).sel(circle_num=0)[0]

    # compare different estimates of inversion base height
    if False:
        plt.figure(figsize=(25, 10))
        plt.plot(
            range(len(circle_means.circle_num)),
            circle_means.inversion_height_stability,
            linewidth=4,
            color="grey",
            label="static stability criterion",
        )
        plt.plot(
            range(len(circle_means.circle_num)),
            circle_means.inversion_height_stability_smoothed,
            linewidth=4,
            color="midnightblue",
            label="static stability smoothed",
        )
        plt.plot(
            range(len(circle_means.circle_num)),
            circle_means.inversion_height_grad_RH,
            linewidth=4,
            color="blue",
            label="RH",
        )
        plt.plot(
            range(len(circle_means.circle_num)),
            circle_means.inversion_height_grad_RH_T,
            linewidth=4,
            color="red",
            label="RH + temp criterion",
        )
        # plt.xlim([range(len(inv_base_height_vec))[0], range(len(inv_base_height_vec))[-1]])
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.xlabel("circle mean")
        plt.ylabel("Estimate for height of inversion base / m")
        plt.ylim([1200, 3000])
        plt.legend(loc="best")

    # static stability and T and RH gradients seem to agree relatively well

    # =============================================================================
    #             calculate other predictor variables
    #               ex. EIS, LTS, mean layer wind and direction
    # =============================================================================

    def find_lower_tropospheric_stability(ds):
        """
        Adapted from Marc Prange's LTS and EIS functions
        Returns lower tropospheric stability (LTS) [K] calculated following Wood and Hartmann (2006).
        """

        def calc_lts(ds):

            # convert Pa to hPa
            if ds["pressure"].max().values > 1400:
                pressure = ds["pressure"] / 100
            else:
                pressure = ds["pressure"]

            if pressure[0, -3] < pressure[0, 1]:  # ignore NaNs at end of
                surf_ind = 1
            else:
                surf_ind = -1
            potential_temperature = ds["theta"]
            num_iters = len(ds["circle_num"])

            lts = np.zeros(num_iters)
            for i in range(num_iters):
                iter_pres = pressure[i, :]
                iter_theta = potential_temperature[i, :]
                pot_temp_interp = interp1d(
                    iter_pres, iter_theta, fill_value="extrapolate", bounds_error=False
                )
                lts[i] = pot_temp_interp(700) - potential_temperature[i, surf_ind]
            return lts

        # call inner function and assign to xarray
        lts = calc_lts(ds)
        ds = ds.assign(LTS=lts)

        return ds

    # call outer function
    circle_means = find_lower_tropospheric_stability(circle_means)

    def find_eis(ds):
        """
        Returns estimated inversion strength (EIS) [K] calculated according to Wood and Bretherton (2006).
        """

        def calc_eis(ds):

            # Celsius to Kelvin
            if ds["temp"].max().values < 100:
                temp_K = ds["temp"] + 273.15
            else:
                temp_K = ds["temp"]
            # convert hPa to Pa
            if ds["pressure"].max().values < 1400:
                pressure_Pa = ds["pressure"] * 100
            else:
                pressure_Pa = ds["pressure"]

            altitude = ds.alt.values
            num_iters = len(ds["circle_num"])

            EIS = np.zeros(num_iters)
            for i in range(num_iters):
                # moist lapse rate units K/m
                moist_lapse_rate = typhon.physics.moist_lapse_rate(
                    pressure_Pa[i, :], temp_K[i, :]
                )
                moist_lapse_rate_850hPa = interp1d(pressure_Pa[i, :], moist_lapse_rate)(
                    850e2
                )
                lts = circle_means["LTS"][i]
                lcl = circle_means["LCL_Bolton"][i]
                z700hPa = interp1d(pressure_Pa[i, :], altitude)(700e2)
                EIS[i] = lts - moist_lapse_rate_850hPa * (z700hPa - lcl)

            return EIS

        # call inner function and assign to xarray
        EIS = calc_eis(ds)
        ds = ds.assign(EIS=EIS)

        return ds

    # call outer function
    circle_means = find_eis(circle_means)

    def layer_mean_wind_speed_dir(ds):

        wind_speed = ds["wspd"]
        # u_wind = ds['u_wind']
        # v_wind = ds['v_wind']
        wind_dir = ds["wind_dir"]
        mixed_layer_height = circle_means["mixed_layer_height_RHmax"]
        LCL = circle_means["LCL_Bolton"]
        inversion_base = circle_means["inversion_height_stability"]
        altitude = circle_means.specific_humidity.alt
        num_iters = len(circle_means["circle_num"])

        # pre-allocate
        wind_speed_mean_ml = np.zeros(num_iters)  # mixed layer
        # u_mean_ml = np.zeros(num_iters)
        # v_mean_ml = np.zeros(num_iters)
        wind_dir_ml = np.zeros(num_iters)

        wind_speed_mean_cl = np.zeros(num_iters)  # cloud layer
        # u_mean_cl = np.zeros(num_iters)
        # v_mean_cl = np.zeros(num_iters)
        wind_dir_cl = np.zeros(num_iters)

        for i in range(num_iters):
            mixed_layer_ind = np.logical_and(
                altitude >= 10, altitude <= mixed_layer_height[i].values
            )
            wind_speed_mean_ml[i] = wind_speed[i, mixed_layer_ind].mean(dim="alt")
            # u_mean_ml[i] = u_wind[i,mixed_layer_ind].mean(dim="alt")
            # v_mean_ml[i] = v_wind[i,mixed_layer_ind].mean(dim="alt")
            wind_dir_ml[i] = wind_dir[i, mixed_layer_ind].mean(dim="alt")

            cloud_layer_ind = np.logical_and(
                altitude >= LCL[i].values, altitude <= inversion_base[i].values
            )
            wind_speed_mean_cl[i] = wind_speed[i, cloud_layer_ind].mean(dim="alt")
            # u_mean_cl[i] = u_wind[i,cloud_layer_ind].mean(dim="alt")
            # v_mean_cl[i] = v_wind[i,cloud_layer_ind].mean(dim="alt")
            wind_dir_cl[i] = wind_dir[i, cloud_layer_ind].mean(dim="alt")

        # just save layer-mean wind speed (not u- and v-components) for now
        ds = ds.assign(wind_speed_ml=wind_speed_mean_ml)
        ds = ds.assign(wind_dir_ml=wind_dir_ml)

        ds = ds.assign(wind_speed_cl=wind_speed_mean_cl)
        ds = ds.assign(wind_dir_cl=wind_dir_cl)

        return ds

    # =============================================================================
    #             example plots
    # =============================================================================
    def plot_all_sondes(circle_means, var, xlabel, xlim1, xlim2):

        plt.figure(figsize=(8, 10))
        for i in range(len(circle_means.circle_num)):
            plt.plot(
                var.isel(circle_num=i),
                var.alt.values,
                color="lightgrey",
                linewidth=2,
                alpha=1,
            )
        plt.plot(var.mean(dim="circle_num"), var.alt.values, linewidth=4, color="black")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.xlabel(xlabel)
        plt.ylabel("Altitude / m")
        plt.ylim([0, 4000])
        plt.xlim([xlim1, xlim2])
        if False:
            plt.axvline(x=0, color="black")
        plt.title(day_str)

    # =============================================================================
    #             display plots?
    # =============================================================================
    if False:
        plot_all_sondes(
            circle_means,
            circle_means.mixing_ratio * 1000,
            xlabel="Mixing ratio / g/kg ",
            xlim1=0,
            xlim2=20,
        )
        plot_all_sondes(
            circle_means, circle_means.theta, xlabel="theta / K", xlim1=295, xlim2=320
        )
        plot_all_sondes(
            circle_means,
            circle_means.theta_v,
            xlabel="theta_v / K",
            xlim1=297,
            xlim2=320,
        )
        plot_all_sondes(
            circle_means, circle_means.temp, xlabel="temperature / K", xlim1=5, xlim2=30
        )
        plot_all_sondes(
            circle_means, circle_means.RH, xlabel="RH / %", xlim1=0, xlim2=100
        )
        plot_all_sondes(
            circle_means,
            circle_means.wspd,
            xlabel="wind speed / m/s",
            xlim1=0,
            xlim2=15,
        )

        # different estimates of mixed layer heights
        plt.figure(figsize=(15, 10))
        plt.plot(
            circle_means["mixed_layer_height_RHmax_2"].values,
            linewidth=3,
            label="RHmax",
        )
        plt.plot(
            circle_means["mixed_layer_height_grad_theta_v"].values,
            linewidth=3,
            label="thetav",
        )
        plt.plot(
            circle_means["mixed_layer_height_grad_mixing_ratio"].values,
            linewidth=3,
            label="mixing ratio",
        )
        plt.legend()
        plt.title(day_str)

        # plot vertical RH profiles in different colors
        rh_str = "RH"
        plt.figure(figsize=(8, 10))
        colors = iter(cm.rainbow(np.linspace(0, 1, len(circle_means[rh_str]))))
        for i in range(len(circle_means[rh_str])):
            alt_vec = circle_means[rh_str].alt
            # convert fraction to [0,100]
            if circle_means[rh_str].max().values < 10:
                plt.plot(
                    circle_means[rh_str][i, :] * 100,
                    alt_vec,
                    color=next(colors),
                    alpha=0.5,
                    linewidth=4,
                )
            else:
                plt.plot(
                    circle_means[rh_str][i, :],
                    alt_vec,
                    color=next(colors),
                    alpha=0.5,
                    linewidth=4,
                )
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.ylabel("Altitude / m")
        plt.xlabel("RH (%)")
        plt.ylim([0, 1500])
        plt.xlim([60, 95])
        plt.title(day_str)

        # plot level of max RH per circle
        plt.figure(figsize=(12, 8))
        plt.plot(
            range(len(circle_means[rh_str])),
            circle_means["mixed_layer_height_RHmax"],
            linewidth=4,
            color="black",
        )
        # plt.xlim([x1, x2])
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.xlabel("Sounding circle ")
        plt.ylabel("Level of max. relative humidity / m")
        plt.ylim([380, 920])
        plt.title(day_str)

        # LCL height distribution
        plt.figure()
        sns.set(rc={"figure.figsize": (10, 12)})
        sns.set(
            context="notebook",
            style="white",
            palette="deep",
            font="sans-serif",
            font_scale=1.8,
            color_codes=True,
            rc=None,
        )
        sns.distplot(
            circle_means["LCL_Bolton"],
            bins=20,
            norm_hist=True,
            hist=False,
            kde=True,
            vertical=True,
            kde_kws={
                "color": "royalblue",
                "lw": 3,
                "label": f"HALO LCL= {int(np.ceil(circle_means.LCL_Bolton.mean().values))} $\pm$ {int(np.ceil(circle_means.LCL_Bolton.std().values))} ({len(circle_means.LCL_Bolton)} sondes)",
            },
            hist_kws={"alpha": 0.5, "color": "royalblue"},
        )
        plt.xlabel("density")
        plt.ylabel("LCL / m")
        plt.title(day_str)

    return circle_means


#%%
# =============================================================================
#      load data and call pre-processing function
# =============================================================================

# load JOANNE dropsondes
input_dir = "/Users/annaleaalbright/Dropbox/EUREC4A/Dropsondes/Data/"
fp_dropsondes = os.path.join(
    input_dir, "EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v0.5.7-alpha+0.g45fe69d.dirty.nc"
)

# load HALO .yaml flight partitioning files into dictionary (RF02, RF03, etc)
def load_flight_files(fp_flights):
    d = {}
    count = 2  # starts with second research flight, RF02
    for filename in sorted(glob.glob(os.path.join(fp_flights, "*.yaml"))):
        with open(filename, "r") as f:
            text = f.read()
            d["rf{0}".format(count)] = filename
            count += 1
    return d


fp_flights = os.path.join(
    input_dir, "HALO_flight_partitioning"
)  # folder with individual partitioning files
d = load_flight_files(fp_flights)  # call functions

# calculate circle means
circle_means_RF02 = preprocess_circle_means(d["rf2"], fp_dropsondes)
circle_means_RF03 = preprocess_circle_means(d["rf3"], fp_dropsondes)
circle_means_RF04 = preprocess_circle_means(d["rf4"], fp_dropsondes)
circle_means_RF05 = preprocess_circle_means(d["rf5"], fp_dropsondes)
# circle_means_RF06 = calc_circle_means(d["rf6"], fp_dropsondes) # just a short flight, no circles
circle_means_RF07 = preprocess_circle_means(d["rf7"], fp_dropsondes)
circle_means_RF08 = preprocess_circle_means(d["rf8"], fp_dropsondes)
circle_means_RF09 = preprocess_circle_means(d["rf9"], fp_dropsondes)
circle_means_RF10 = preprocess_circle_means(d["rf10"], fp_dropsondes)
circle_means_RF11 = preprocess_circle_means(d["rf11"], fp_dropsondes)
circle_means_RF12 = preprocess_circle_means(d["rf12"], fp_dropsondes)
circle_means_RF13 = preprocess_circle_means(d["rf13"], fp_dropsondes)
circle_means_RF14_raw = preprocess_circle_means(d["rf14"], fp_dropsondes)
circle_means_RF14 = circle_means_RF14_raw.dropna(
    dim="circle_num", subset=["temp"], how="any", thresh=100
)

#%%
# =============================================================================
# # load data from flights
# =============================================================================


def load_data_circles(var_str):

    flight_ids = [
        "02",
        "03",
        "04",
        "05",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
    ]
    var_rs = np.zeros((len(flight_ids), 6))
    # loop over flights
    ct_flight = 0
    for i in flight_ids:
        # load flight data
        key = "circle_means_RF{}".format(i)
        circle_means = eval(key)
        var = circle_means[var_str]
        for ct_circle in range(len(var)):
            var_rs[ct_flight, ct_circle] = var[ct_circle]
        ct_flight += 1
    # 0 --> nan
    var_rs[var_rs == 0] = np.nan
    var_rs[
        -1, 2
    ] = (
        np.nan
    )  # third circle in last flight not performed within HALO circle (by NTAS buoy)
    return var_rs


#%%
# =============================================================================
# load data where there is one value per flight
# i.e. mixed layer height, LCL, inversion base height
# =============================================================================

# mixed layer height estimates
mixed_layer_height_RHmax = load_data_circles(
    var_str="mixed_layer_height_RHmax_2"
)  # or, mixed_layer_height_RHmax
mixed_layer_height_grad_mixing_ratio = load_data_circles(
    var_str="mixed_layer_height_grad_mixing_ratio"
)
mixed_layer_height_grad_theta_v = load_data_circles(
    var_str="mixed_layer_height_grad_theta_v"
)

LCL_Bolton = load_data_circles(var_str="LCL_Bolton")
inversion_height_stability = load_data_circles(var_str="inversion_height_stability")


#%%

# =============================================================================
#           how well do these various definitions agree?
# =============================================================================

sns.set(
    context="notebook",
    style="whitegrid",
    palette="deep",
    font="sans-serif",
    font_scale=2,
    color_codes=True,
    rc=None,
)
g = sns.jointplot(
    x=mixed_layer_height_RHmax,
    y=mixed_layer_height_grad_theta_v,
    kind="kde",
    color="m",
    height=8,
    xlim=(320, 880),
    ylim=(320, 880),
)  #                   marginal_kws=dict(bw=40),bw=40
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels(
    "relative humidity maximum", "$\partial\Theta_v/\partial z$ = 0 $\mid$ threshold"
)
x0, x1 = g.ax_joint.get_xlim()
y0, y1 = g.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
lims = [max(x0, y0), min(x1, y1)]
g.ax_joint.plot(lims, lims, ":k", linewidth=2)
g.plot_joint(plt.scatter, c="w", s=40, linewidth=1.5, marker="+")

g = sns.jointplot(
    x=mixed_layer_height_RHmax,
    y=mixed_layer_height_grad_mixing_ratio,
    kind="kde",
    color="m",
    height=8,
    xlim=(320, 880),
    ylim=(320, 880),
)
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels(
    "relative humidity maximum", "$\partial q/\partial z$ = 0 $\mid$ threshold"
)
x0, x1 = g.ax_joint.get_xlim()
y0, y1 = g.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
lims = [max(x0, y0), min(x1, y1)]
g.ax_joint.plot(lims, lims, ":k", linewidth=2)
g.plot_joint(plt.scatter, c="w", s=40, linewidth=1.5, marker="+")

g = sns.jointplot(
    x=mixed_layer_height_grad_theta_v,
    y=mixed_layer_height_grad_mixing_ratio,
    kind="kde",
    color="m",
    height=8,
)
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels(
    "$\partial\Theta_v/\partial z$ = 0 $\mid$ threshold",
    "$\partial q/\partial z$ = 0 $\mid$ threshold",
)
x0, x1 = g.ax_joint.get_xlim()
y0, y1 = g.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
lims = [max(x0, y0), min(x1, y1)]
g.ax_joint.plot(lims, lims, ":k", linewidth=2)
g.plot_joint(plt.scatter, c="w", s=40, linewidth=1.5, marker="+")

# alternate plot
# g = sns.JointGrid(x=mixed_layer_height_RHmax, y=mixed_layer_height_grad_theta_v, space=0)
# g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
# sns.kdeplot(mixed_layer_height_RHmax.flatten(), color="b", shade=True, bw=0.1, ax=g.ax_marg_x)
# sns.kdeplot(mixed_layer_height_grad_theta_v.flatten(), color="r", shade=True, bw=0.01, vertical=True, ax=g.ax_marg_y)

# list the correlations for methods for 70 circle-means, with 0.05, 0.4 thresholds
# RH and q: r=0.84
# RH and theta_v: r=0.56
# q and theta_v: 0.59

# list the correlations for methods for 70 circle-means, with 0.1K, 0.4 g/kg thresholds
# RH and q: r=0.84
# RH and theta_v: r=0.62
# q and theta_v: 0.68


import numpy.ma as ma

# mixed_layer_height_RHmax, mixed_layer_height_grad_mixing_ratio, mixed_layer_height_grad_theta_v
A = mixed_layer_height_grad_mixing_ratio.flatten()
B = mixed_layer_height_RHmax.flatten()
print(ma.corrcoef(ma.masked_invalid(A), ma.masked_invalid(B)))

# np.nanstd(mixed_layer_height_grad_theta_v)
# np.nanmean(mixed_layer_height_grad_mixing_ratio)

# mean (one sigma)
# RHmax: 614 (97.1)
# q  622 (66.1)

plt.figure(figsize=(10, 10))
sns.distplot(mixed_layer_height_RHmax - mixed_layer_height_grad_mixing_ratio)
plt.xlim([-200, 200])
plt.title("RHmax - dq/dz")
plt.figure(figsize=(10, 10))
sns.distplot(mixed_layer_height_RHmax - mixed_layer_height_grad_theta_v)
plt.xlim([-200, 200])
plt.title("RHmax - dthetav/dz")
plt.figure(figsize=(10, 10))
sns.distplot(mixed_layer_height_grad_mixing_ratio - mixed_layer_height_grad_theta_v)
plt.xlim([-200, 200])
plt.title("dq/dz - dthetav/dz")

# what is mean of residuals
residuals_RH_q = mixed_layer_height_RHmax - mixed_layer_height_grad_mixing_ratio
residuals_RH_thetav = mixed_layer_height_RHmax - mixed_layer_height_grad_theta_v
residuals_thetav_q = (
    mixed_layer_height_grad_theta_v - mixed_layer_height_grad_mixing_ratio
)


def print_mean_sigma_residuals(residuals):
    print("mean of residuals", np.round(np.nanmean(residuals), 2))
    print("standard deviation of residuals", np.round(np.nanstd(residuals), 2))


print_mean_sigma_residuals(residuals_thetav_q)


#%%

# =============================================================================
#                   define various dictionaries of attributes
# =============================================================================

listofDates = [
    "2020-01-22",
    "2020-01-24",
    "2020-01-26",
    "2020-01-28",
    "2020-01-31",
    "2020-02-02",
    "2020-02-05",
    "2020-02-07",
    "2020-02-09",
    "2020-02-11",
    "2020-02-13",
    "2020-02-15",
]

days = [
    "01-22",
    "01-24",
    "01-26",
    "01-28",
    "01-31",
    "02-02",
    "02-05",
    "02-07",
    "02-09",
    "02-11",
    "02-13",
    "02-15",
]

flights = [
    "RF02_01_22",
    "RF03_01_24",
    "RF04_01_26",
    "RF05_01_28",
    "RF07_01_31",
    "RF08_02_02",
    "RF09_02_05",
    "RF10_02_07",
    "RF11_02_09",
    "RF12_02_11",
    "RF13_02_13",
    "RF14_02_15",
]

# research flights short-hand
rfs = [
    "RF02",
    "RF03",
    "RF04",
    "RF05",
    "RF07",
    "RF08",
    "RF09",
    "RF10",
    "RF11",
    "RF12",
    "RF13",
    "RF14",
]

dictOfDays = dict(zip(rfs, days))

dict_attrs = {"listofDates": listofDates, "days": days, "flights": flights}
df_attrs = pd.DataFrame(dict_attrs)

# labels for flight circles (for dataframe)
x1 = [2, 3, 4, 5, 7, 8, 9, 10, 11]
research_flight = np.repeat(x1, 6)
research_flight_12 = np.repeat(12, 5)  # RF12 only has 5 circles
research_flight13 = np.repeat(13, 6)
research_flight14 = np.repeat(14, 5)
research_flight = np.hstack(
    (research_flight, research_flight_12, research_flight13, research_flight14)
)
