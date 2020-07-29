import numpy as np

from eurec4a_environment.source_data import open_joanne_dataset
from eurec4a_environment.interface import calculate


def test_calc():
    ds_joanne = open_joanne_dataset()

    result = calculate("thingness", ds_joanne)
    expected = ds_joanne.u*ds_joanne.v*ds_joanne.T

    expected = _mask_nans(expected.values)
    result = _mask_nans(result.values)

    assert (result == expected).all()


def _mask_nans(array):
    # NaNs mess up array-wise equality checks
    return np.ma.masked_where(np.isnan(array), array)
