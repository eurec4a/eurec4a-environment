import numpy as np
import xarray as xr

import eurec4a_environment.variables.tropical as tropical_variables
import eurec4a_environment.variables.boundary_layer as bl_variables
from eurec4a_environment import nomenclature as nom
from eurec4a_environment.variables import atmos
from eurec4a_environment.constants import g, cp_d
import eurec4a_environment.source_data

from reference_profiles import make_fixed_lapse_rate_dry_profile, make_pressure_grid


def test_lts_isentropic_profile():
    ds = make_fixed_lapse_rate_dry_profile(dTdz="isentropic")
    ds[nom.POTENTIAL_TEMPERATURE] = atmos.potential_temperature(
        ds=ds, vertical_coord=nom.ALTITUDE
    )
    lts = tropical_variables.lower_tropospheric_stability(ds=ds, pressure=nom.PRESSURE)
    assert lts.units == "K"
    assert np.allclose(lts, 0.0)


def test_lts_linear_profile_pressure_coord():
    dtheta = 2.0  # [K]
    theta0 = 300.0
    theta1 = theta0 + dtheta
    p0 = 100000.0  # [Pa], surface pressure
    p1 = 70000.0  # [Pa], pressure at height of theta1

    ds = make_pressure_grid(p0=p0, p_min=p1, dp=10.0)

    da_theta = (
        theta0
        + (ds[nom.PRESSURE] - p0) * (theta1 - theta0) / (p1 - p0)
        + 0.0 * ds.timestep
    )
    da_theta.attrs["units"] = "K"
    da_theta.attrs["long_name"] = "potential temperature"
    ds[nom.POTENTIAL_TEMPERATURE] = da_theta

    lts = tropical_variables.lower_tropospheric_stability(
        ds=ds, pressure=nom.PRESSURE, vertical_coord=nom.PRESSURE
    )
    assert lts.units == "K"
    # TODO: after adding linear interpolation inside the LTS routine this
    # should become more accurate
    assert np.allclose(lts, dtheta, rtol=1.0e-3)


def test_eis_isentropic_profile():
    # make an atmosphere which cools only half as fast with height as
    # isentropic, making it very stable
    dTdz = -0.5 * g / cp_d
    ds = make_fixed_lapse_rate_dry_profile(dTdz=dTdz, z_max=5.0e3)
    ds[nom.POTENTIAL_TEMPERATURE] = atmos.potential_temperature(
        ds=ds, vertical_coord=nom.ALTITUDE
    )
    # add a moist layer need the surface (z < 500m)
    ds[nom.RELATIVE_HUMIDITY] = 0.9 * xr.ones_like(ds[nom.TEMPERATURE]).where(
        ds[nom.ALTITUDE] < 500.0, 0.0
    )
    ds[nom.RELATIVE_HUMIDITY].attrs["units"] = "1"
    ds[nom.RELATIVE_HUMIDITY].attrs["long_name"] = "relative humidity"
    ds[nom.RELATIVE_HUMIDITY].attrs["standard_name"] = nom.CF_STANDARD_NAMES[
        nom.RELATIVE_HUMIDITY
    ]

    LCL_name = bl_variables.lcl.find_LCL_Bolton.name
    LTS_name = tropical_variables.lower_tropospheric_stability.name

    da_lts = tropical_variables.lower_tropospheric_stability(
        ds=ds, pressure=nom.PRESSURE
    )
    ds[LTS_name] = da_lts

    da_lcl = bl_variables.lcl.find_LCL_Bolton(ds=ds)
    ds[LCL_name] = da_lcl

    da_eis = tropical_variables.estimated_inversion_strength(
        ds=ds, LCL=LCL_name, LTS=LTS_name
    )

    assert da_eis.units == "K"
    assert np.all(da_eis > 0.0)


def test_eis_joanne_profile():
    ds = eurec4a_environment.source_data.open_joanne_dataset()
    ds = ds.isel(sounding=slice(0, 10))

    LCL_name = bl_variables.lcl.find_LCL_Bolton.name
    da_lcl = bl_variables.lcl.find_LCL_Bolton(ds=ds)
    ds[LCL_name] = da_lcl

    LTS_name = tropical_variables.lower_tropospheric_stability.name
    da_lts = tropical_variables.lower_tropospheric_stability(
        ds=ds, pressure=nom.PRESSURE
    )
    ds[LTS_name] = da_lts

    da_eis = tropical_variables.estimated_inversion_strength(ds=ds, LCL=LCL_name, LTS=LTS_name)

    assert da_eis.units == "K"
    assert np.all(da_eis > 0.0)
