from eurec4a_environment.decomposition import levels
import numpy as np


def test_height_specified_quantity(ds_test_levels):
    ds = ds_test_levels
    assert (
        np.round(
            np.mean(
                levels.height_specified_quantity(
                    ds,
                    variable="RH",
                    level_name="mixed_layer",
                    level_definition="max_RH",
                    bounds=40,
                ).values
            ),
            2,
        )
        == 0.95
    )

    assert (
        np.round(
            levels.height_specified_quantity(
                ds,
                variable="T",
                level_name="cloud_layer",
                level_definition="max_RH",
                cell_type="point",
                drop_nan=False,
            ),
            2,
        ).values
        == np.array([293.16, 293.36, 293.56])
    ).all()

