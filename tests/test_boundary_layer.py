import numpy as np

from eurec4a_environment.variables import boundary_layer
from eurec4a_environment.variables.boundary_layer import inversion_height
from eurec4a_environment import nomenclature as nom
import eurec4a_environment.source_data


def test_LCL_Bolton(ds_isentropic_test_profiles):
    ds = ds_isentropic_test_profiles
    da_lcl = boundary_layer.lcl.find_LCL_Bolton(ds=ds)
    assert da_lcl.mean() > 0.0
    assert da_lcl.mean() < 2000.0


def test_mixed_layer_height_RHmax(ds_isentropic_test_profiles):
    ds = ds_isentropic_test_profiles
    # set the RH profile so we know where the peak should be
    z0 = 600.0
    z = ds[nom.ALTITUDE]
    ds[nom.RELATIVE_HUMIDITY] = 1.0 - np.maximum(np.abs(z0 - z), 0) / z0
    ds[nom.RELATIVE_HUMIDITY].attrs["units"] = "1"
    da_rh_peak = boundary_layer.mixed_layer_height.calc_peak_RH(
        ds=ds, altitude=nom.ALTITUDE, rh=nom.RELATIVE_HUMIDITY
    )
    assert np.allclose(da_rh_peak, z0)


def test_mixed_layer_height_RH_lin():
    ds = eurec4a_environment.source_data.open_joanne_dataset()
    ds = ds.isel(sounding=slice(0, 10))
    da_lin = boundary_layer.mixed_layer_height.calc_peakRH_linearize(ds)
    assert len(da_lin) == 10
    assert np.all(da_lin < 1500.0)


def test_mixed_layer_height_gradient():
    ds = eurec4a_environment.source_data.open_joanne_dataset()
    ds = ds.isel(sounding=slice(0, 10))
    # adjust threshold based on units 0.4 g/kg/10m or 0.0004 kg/kg/10m
    da_gradient = boundary_layer.mixed_layer_height.calc_from_gradient(
        ds, var=nom.SPECIFIC_HUMIDITY, threshold=0.4 / 1000
    )
    assert len(da_gradient) == 10
    assert np.all(da_gradient < 1500.0)
    assert da_gradient.units == "m"


def test_inversion_height_gradient_RH(ds_isentropic_test_profiles):
    ds = ds_isentropic_test_profiles.copy()
    z_INV = 2000.0
    z = ds[nom.ALTITUDE]
    ds[nom.RELATIVE_HUMIDITY] = ds[nom.RELATIVE_HUMIDITY].where(z < z_INV, other=0.5)
    ds[nom.RELATIVE_HUMIDITY].attrs["units"] = "1"
    da_inv = boundary_layer.inversion_height.find_inversion_height_grad_RH(
        ds=ds, rh=nom.RELATIVE_HUMIDITY
    )
    assert np.allclose(da_inv, z_INV, atol=20)  ## within 20m
