from eurec4a_environment.decomposition import levels


def test_scalar(ds_test_levels):
    ds = ds_test_levels
    assert (
        int(
            levels.scalar(
                ds.T, level_name="mixed_layer", level_definition="max_RH", bounds=40
            )
        )
        == 292
    )
