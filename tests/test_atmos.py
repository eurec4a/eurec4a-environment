"""
Test routines for routines calculating standard atmospheric variables, e.g.
potential temperature, hydrostatic pressure, etc
"""
import numpy as np

from eurec4a_environment import nomenclature as nom
from eurec4a_environment.variables import atmos
from reference_profiles import make_fixed_lapse_rate_dry_profile


def test_hydrostatic_pressure():
    T0 = 300.0
    ds = make_fixed_lapse_rate_dry_profile(dTdz="isothermal", T0=T0)
    p0 = ds.isel(**{nom.ALTITUDE: 0})[nom.PRESSURE]

    ds_no_pressure = ds.drop(nom.PRESSURE)

    da_p = atmos.hydrostatic_pressure(ds=ds_no_pressure, p_surf=p0)

    MAX_ERROR = 0.001

    assert da_p.units == "Pa"
    assert np.allclose(da_p, ds.p, rtol=MAX_ERROR)


def test_density_calculation_dry_profile():
    ds = make_fixed_lapse_rate_dry_profile(dTdz="isothermal")
    da_density = atmos.density(ds)
    # check that we get some sensible numbers out for the density
    assert da_density.units == "kg/m3"
    assert da_density.max() < 2.5
    assert da_density.min() > 0.0
