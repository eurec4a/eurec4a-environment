import numpy as np

from eurec4a_environment.decomposition import time as time_decomposition
from eurec4a_environment.decomposition import space as space_decomposition
from eurec4a_environment.source_data import open_joanne_dataset


def test_time_decomposition():
    ds_joanne = open_joanne_dataset()
    ds = time_decomposition.annotate_with_halo_flight_legs(
        ds_joanne, time="launch_time", legtype="circle"
    )

    ds_mean_per_segment = ds.groupby("halo_segment_id").mean()
    assert ds_mean_per_segment.halo_segment_id.count() > 0


def test_halo_circle_space_decomposition():
    circle = space_decomposition.HALOCircle()

    # check for single set point
    assert circle.contains(lat=circle.lat, lon=circle.lon)
    # check for array of points
    assert np.all(
        circle.contains(
            lat=np.array([circle.lat + circle.r_degrees, circle.lat]),
            lon=np.array([circle.lon, circle.lon + circle.r_degrees]),
        )
    )
    assert not circle.contains(
        lat=circle.lat + circle.r_degrees, lon=circle.lon + circle.r_degrees
    )

    ds_joanne = open_joanne_dataset()
    # take mean per-sounding so we find if the mean lat/lon position is inside
    # the cirlce or not (just to make things a bit faster)
    ds_per_sounding = ds_joanne.mean(dim="height")
    ds_joanne["inside_circle"] = space_decomposition.inside_halo_circle(
        ds=ds_per_sounding
    )

    # there should be points both inside and outside the circle for the dropsondes
    assert ds_joanne.where(ds_joanne.inside_circle, drop=True).count() > 0
    assert ds_joanne.where(~ds_joanne.inside_circle, drop=True).count() > 0
