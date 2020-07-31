import numpy as np


from eurec4a_environment.variables import boundary_layer
import eurec4a_environment.source_data


def test_LCL_Bolton(ds_isentropic_test_profiles):
    ds = ds_isentropic_test_profiles
    da_lcl = boundary_layer.lcl.find_LCL_Bolton(ds=ds)
    assert da_lcl.mean() > 0.0
    assert da_lcl.mean() < 2000.0


def test_mixed_layer_height_RHmax(ds_isentropic_test_profiles):
    ds = ds_isentropic_test_profiles
    # set the RH profile so we know where the peak should be
    z0 = 600.
    z = ds.alt
    ds['rh'] = 1.0 - np.maximum(np.abs(z0 - z), 0) / z0
    da_rh_peak = boundary_layer.mixed_layer_height.calc_peak_RH(
        ds=ds, altitude="alt", rh="rh"
    )
    assert np.allclose(da_rh_peak, z0)

def test_inversion_height_gradient_RH(ds_isentropic_test_profiles):
    ds = ds_isentropic_test_profiles
    da_inv = boundary_layer.inversion_height.find_inversion_height_grad_RH(ds=ds)
    assert da_inv.mean() > 1500.0
    assert da_inv.mean() < 4000.0

def test_mixed_layer_height_RH_lin():
    ds = eurec4a_environment.source_data.open_joanne_dataset()
    ds = ds.isel(sounding=slice(0, 10))
    da_lin = boundary_layer.mixed_layer_height.calc_peakRH_linearize(ds)
    assert len(da_lin) == 10
    assert np.all(da_lin < 1500.)
