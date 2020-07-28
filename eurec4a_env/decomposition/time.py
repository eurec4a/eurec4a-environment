import xarray as xr

from ..source_data import get_halo_flight_legs


def annotate_with_halo_flight_legs(ds, time="time", legtype="circle"):
    """
    Add halo flight-leg information to the dataset provided by matching on
    `time` coordinate, HALO flight leg variables start with `halo_`. Once added
    these extra variables can be used to aggregate over flight legs, for
    example:

    >> ds_joanne = open_joanne_dataset()
    >> ds = annotate_with_halo_flight_legs(ds_joanne, time="launch_time", legtype="circle")
    >> ds.groupby('halo_segment_id')

    See [xarray.groupby](http://xarray.pydata.org/en/stable/groupby.html) on
    how to calculate e.g. mean per-leg or apply other functions. You can also
    simply call `list(annotate_with_halo_flight_legs(...))` to get individual
    datasets per leg
    """
    ds_legs = get_halo_flight_legs(legtype=legtype)

    # ensure we have time as a coordinate
    org_dim = None
    try:
        ds.isel(**{time: 0})
    except ValueError:
        org_dim = ds[time].dims[0]
        ds = ds.swap_dims({org_dim: time})

    annotated = []
    for segment_id in ds_legs.segment.values:
        ds_segment = ds_legs.sel(segment=segment_id)
        ds_ = ds.sel(**{time: slice(ds_segment.start, ds_segment.end)})
        # only keep segments which contain data
        if ds_[time].count() == 0:
            continue
        for v in ds_segment.data_vars:
            ds_[f"halo_{v}"] = ds_segment[v]
        annotated.append(ds_)

    ds_annotated = xr.concat(annotated, dim="launch_time")

    if org_dim is not None:
        ds_annotated = ds_annotated.swap_dims({time: org_dim})
    return ds_annotated
