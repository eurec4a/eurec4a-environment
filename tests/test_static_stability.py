#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from eurec4a_environment.variables.calc_static_stability import calc_static_stability
import eurec4a_environment.source_data


def test_static_stability_calculation():
    ds = eurec4a_environment.source_data.open_joanne_dataset()
    da_static_stability = calc_static_stability(ds)
    # check that we get some sensible numbers
    assert da_static_stability.units == "K/hPa"
    assert da_static_stability.max() < 3
    assert da_static_stability.min() > -3
