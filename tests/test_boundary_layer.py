from eurec4a_environment.variables import boundary_layer
from eurec4a_environment.source_data import open_joanne_dataset


def test_LCL_Bolton(ds_isentropic_test_profiles):
    ds = ds_isentropic_test_profiles
    da_lcl = boundary_layer.lcl.find_LCL_Bolton(ds=ds)
    assert da_lcl.mean() > 0.0
    assert da_lcl.mean() < 2000.0


def test_mixed_layer_height_RHmax():
    ds = open_joanne_dataset()
    da_lcl = boundary_layer.mixed_layer_height.calc_peak_RH(
        ds=ds, altitude="height", rh="rh"
    )
    assert da_lcl.mean() > 0.0
    assert da_lcl.mean() < 2000.0
