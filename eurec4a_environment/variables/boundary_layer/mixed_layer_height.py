import xarray as xr


def calc_peak_RH(ds, altitude="alt", rh="RH", z_min=200.0, z_max=900.0):
    """
    Calculate height at maximum relative humidity values
    """
    ds_mixed_layer = ds.sel({altitude: slice(z_min, z_max)})

    da_rh = ds_mixed_layer[rh]
    dims = list(ds.dims.keys())
    # drop the height coord
    del dims[dims.index(altitude)]

    peakRH_idx = da_rh.argmax(dim=altitude)
    h_peakRH = ds[altitude].isel({altitude: peakRH_idx})

    da = xr.DataArray(h_peakRH, dims=dims, coords={d: ds[d] for d in dims})
    da.attrs["long_name"] = "mixed layer height (from RH peak)"
    da.attrs["units"] = "m"

    return da
