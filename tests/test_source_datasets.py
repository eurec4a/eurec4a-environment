import eurec4a_environment.source_data
import eurec4a_environment.nomenclature as nom


def test_download_joanne_level3():
    ds = eurec4a_environment.source_data.open_joanne_dataset()
    assert ds[nom.ALTITUDE].count() > 0
    assert ds.sounding.count() > 0


def test_download_halo_flight_segments():
    ds = eurec4a_environment.source_data.get_halo_flight_legs()
    N_segments = ds.flight_num.count()
    assert N_segments > 0

    ds = eurec4a_environment.source_data.get_halo_flight_legs(legtype="circle")
    ds.flight_num.count() < N_segments
