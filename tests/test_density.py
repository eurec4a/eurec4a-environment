# -*- coding: utf-8 -*-

import numpy as np

from eurec4a_environment.variables.calc_density import calc_density
import eurec4a_environment.source_data


def test_density_calculation():
    ds = eurec4a_environment.source_data.open_joanne_dataset()
    ds = ds.isel(sounding=slice(0, 10))
    da_density = calc_density(ds)
    # check that we get some sensible numbers out for the density
    assert da_density.units == "kg/m3"
    assert da_density.max() < 2.5
    assert da_density.min() > 0.0
