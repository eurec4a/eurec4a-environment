import numpy as np

import eurec4a_environment.variables.tropical as tropical_variables
from eurec4a_environment import nomenclature as nom
from eurec4a_environment.variables import atmos
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
