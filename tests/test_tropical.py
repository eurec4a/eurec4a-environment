import numpy as np

import eurec4a_environment.variables.tropical import tropical_variables
from eurec4a_environment import nomenclature as nom


def test_eis():
    pass


def test_lts_isentropic_profile(ds_isentropic_test_profiles):
    ds = ds_isentropic_test_profiles
    lts = tropical_variables.lts(ds=ds)
    assert lts.units == "K"
    assert np.all_close(lts, 0.0)
