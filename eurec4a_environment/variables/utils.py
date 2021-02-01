def apply_by_column(ds, vertical_coord, fn):
    if not vertical_coord in ds.dims:
        raise Exception(
            f"`{vertical_coord}` is not a dimension of the provided " "dataset"
        )

    dims = set(ds.dims)
    dims.remove(vertical_coord)
    return ds.stack(column=dims).groupby("column").apply(fn).unstack("column")
