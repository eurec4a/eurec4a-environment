import pytest

from eurec4a_environment import unit
from eurec4a_environment import nomenclature as nom


@pytest.mark.parametrize(
    "var_name,new_units",
    [
        (nom.TEMPERATURE, "celsius")
    ]
)
def test_convert_units(ds_joanne, var_name, new_units):
    unit.convert_units(ds_joanne[var_name], new_units)


def test_convert_units_missing(ds_joanne):
    var_name = nom.TEMPERATURE

    da = ds_joanne[var_name].copy()
    del da.attrs["units"]

    with pytest.raises(KeyError):
        unit.convert_units(da, "celsius")
