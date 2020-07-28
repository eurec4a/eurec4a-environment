from eurec4a_environment.decomposition import time as time_decomposition
from eurec4a_environment.source_data import open_joanne_dataset


def test_time_decomposition():
    ds_joanne = open_joanne_dataset()
    ds = time_decomposition.annotate_with_halo_flight_legs(
        ds_joanne, time="launch_time", legtype="circle"
    )

    ds_mean_per_segment = ds.groupby("halo_segment_id").mean()
    assert ds_mean_per_segment.halo_segment_id.count() > 0
