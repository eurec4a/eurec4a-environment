import eurec4a_env.source_data


def test_download_joanne_level3():
    ds = eurec4a_env.source_data.open_joanne_dataset()
    assert ds.height.count() > 0
    assert ds.sounding.count() > 0
