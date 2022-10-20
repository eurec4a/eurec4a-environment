def apply_by_column(ds, vertical_coord, fn):
    if not vertical_coord in ds.dims:
        raise Exception(
            f"`{vertical_coord}` is not a dimension of the provided " "dataset"
        )

    # use wrapped call here to ensure the column dimension is squeezed out.
    # Allows for simpler indexing in the column-wise functions
    def _wrapped_fn(da):
        return fn(da.squeeze())

    dims = set(ds.dims)
    dims.remove(vertical_coord)
    return ds.stack(column=dims).groupby("column").apply(_wrapped_fn).unstack("column")
