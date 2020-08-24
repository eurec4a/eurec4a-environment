import pytest

from eurec4a_environment import unit, nomenclature


@pytest.mark.parametrize(
    "var_name,new_units",
    [
        (nomenclature.TEMPERATURE, "celsius")
    ]
)
def test_convert_units(ds_joanne, var_name, new_units):
    unit.convert_units(ds_joanne[var_name], new_units)
