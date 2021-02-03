#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from eurec4a_environment.variables.inversion_layers import inversion_layer_height
import eurec4a_environment.source_data


def test_inversion_static_stability():
    ds = eurec4a_environment.source_data.open_joanne_dataset()
    ds = ds.isel(sounding=slice(0, 10))
    da_ss = inversion_layer_height.calc_inversion_static_stability(ds)
    assert len(da_ss) == 10
    assert np.all(da_ss < 5000.0)
    assert da_ss.units == "m"


def test_inversion_gradient_RH_T():
    ds = eurec4a_environment.source_data.open_joanne_dataset()
    ds = ds.isel(sounding=slice(0, 10))
    # adjust threshold based on units 0.4 g/kg/10m or 0.0004 kg/kg/10m
    da_gradRH_T = inversion_layer_height.calc_inversion_grad_RH_T(ds)
    assert len(da_gradRH_T) == 10
    assert np.all(da_gradRH_T < 5000.0)
    assert da_gradRH_T.units == "m"
