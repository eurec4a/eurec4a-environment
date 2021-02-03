import numpy as np
import xarray as xr
from scipy.signal import find_peaks
import statsmodels.api as sm

from ... import get_field, get_fields, nomenclature as nom
from ..utils import apply_by_column

DEBUG = False


def calc_peak_RH(
    ds, altitude=nom.ALTITUDE, rh=nom.RELATIVE_HUMIDITY, z_min=200.0, z_max=900.0
):
    """
    Calculate height at maximum relative humidity values
    """
    ds_mixed_layer = ds.sel({altitude: slice(z_min, z_max)})

    da_rh = get_field(ds=ds_mixed_layer, name=rh, units="%")
    dims = list(ds.dims.keys())
    # drop the height coord
    del dims[dims.index(altitude)]

    peakRH_idx = da_rh.argmax(dim=altitude)
    da = da_rh.isel({altitude: peakRH_idx})[altitude]

    da.attrs["long_name"] = "mixed layer height (from RH peak)"
    da.attrs["units"] = "m"

    return da


def calc_peakRH_linearize(
    ds,
    altitude=nom.ALTITUDE,
    rh=nom.RELATIVE_HUMIDITY,
    time="sounding",
    z_min=200.0,
    z_max=1500.0,
    z_min_lin=50.0,
    z_max_lin=400.0,
    vertical_coord=":altitude",
):
    """
    Calculate maximum in relative humidity (RH) profile that minimizes
    difference between observed RH profile and RH profile linearized around
    lower levels (i.e. 200-400m).  Assume moisture is well-mixed at these lower
    levels, such that RH would increase and assume linear profile as an
    idealization.

    Inputs:
        - ds: dataset
        - altitude, rh, time: variable names
        - z_min and z_max: lower and upper bounds for mixed layer height
        - z_min_lin and z_max_lin: bounds for linearization of observed RH profile

    Outputs:
        - da: datarray containing h_peakRH_linfit
    """
    # if `vertical_coord` hasn't been changed we'll assume that the vertical
    # coordinate has name of the altitude variable
    if vertical_coord == ":altitude":
        vertical_coord = altitude

    import matplotlib.pyplot as plt

    def _calc_mixed_layer_depth(ds_column):
        da_alt = ds_column[altitude]

        mixed_layer = np.logical_and(da_alt >= z_min, da_alt <= z_max)
        ml_linfit = np.logical_and(
            da_alt >= z_min_lin, da_alt <= z_max_lin
        )  # for linearized RH profile

        dz = int(da_alt.diff(dim=altitude)[1])  # dz=10m

        rh_profile = ds_column[rh].interpolate_na(dim=altitude)

        # fit a straight line to the relative humidity values in the
        # "ml-linfit" region
        X_height = rh_profile[ml_linfit][altitude]
        X = sm.add_constant(X_height.values)  # add intercept
        model = sm.OLS(
            rh_profile[ml_linfit].values, X
        ).fit()  # instantiate linear model
        linearized_RH = model.predict(sm.add_constant(da_alt))

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

        h_peakRH_linfit = min_diff_RH[altitude]

        if DEBUG:
            ds_column[rh].plot(y=altitude, label="raw profile")
            rh_profile.plot(y=altitude, label="raw profile interpolated")
            plt.plot(linearized_RH, da_alt, label="RH interpolation in linfit region")

            rh_profile.isel({altitude: idx_peaks_RH}).plot(
                marker="x", linestyle="--", label="interpolated profile peaks", y=altitude
            )

            plt.axhline(h_peakRH_linfit, label="h_peakRH_linfit")

            plt.legend()
            plt.show()

        # set 'anomalously' high values > z_max to nan
        # ... somewhat arbitrary choice
        if h_peakRH_linfit > z_max:
            h_peakRH_linfit = np.nan

        import ipdb

        ipdb.set_trace()

        return h_peakRH_linfit

    ds_subset = get_fields(ds=ds, **{rh: "%", altitude: "m"})
    da_ml_depth = apply_by_column(
        ds=ds_subset, vertical_coord=vertical_coord, fn=_calc_mixed_layer_depth
    )

    da_ml_depth.attrs[
        "long_name"
    ] = "mixed layer height (from RH peak and linearization)"
    da_ml_depth.attrs["units"] = "m"
    return da_ml_depth


def calc_from_gradient(
    ds, var, threshold, z_min=200, altitude=nom.ALTITUDE, time="sounding"
):
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

    def calculateHmix_var(
        density_profile,
        var_profile,
        threshold,
        z_min,
        altitude=nom.ALTITUDE,
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
            altitude=altitude,
            time="sounding",
        )

    dims = list(ds.dims.keys())
    # drop the height coord
    del dims[dims.index(altitude)]
    da = xr.DataArray(hmix_vec, dims=dims, coords={d: ds[d] for d in dims})
    da.attrs["long_name"] = f"mixed layer height (from gradient method using {var})"
    da.attrs["units"] = "m"

    return da
