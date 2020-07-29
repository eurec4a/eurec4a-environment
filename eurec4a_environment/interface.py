# Example function for testing calculate
def _thingness(u, v, T):
    return u*v*T


def calculate(name, ds):
    """Calculate a variable from the given dataset by CF standard name

    Args:
        name (str): The CF standard name of the variable to be calculated
        ds (xarray.DataSet): The dataset to calculate the variable from

    Returns:
        xarray.DataArray:

    Raises:
        ValueError: If the requested variable (or a variable required to calculate it)
            is not available in the dataset
    """
    # If the variable is in the dataset, just return it
    if name in ds:
        return ds[name]

    # Also check if the name matches a CF standard name
    # Create a mapping of CF standard names in the dataset to their variable name
    translation_table = dict()
    for varname in ds:
        try:
            translation_table[ds[varname].attrs["standard_name"]] = varname
        except KeyError:
            # Skip variables without a standard_name attribute
            pass

    # Check whether the requested variable matches a standard name
    if name in translation_table:
        result = ds[translation_table[name]].copy()
        _rename_xarray(result, name)
        return result

    # Otherwise calculate the requested variable using variables in the dataset
    if name in available:
        arguments = []
        for argument_name in available[name]["arguments"]:
            arguments.append(calculate(argument_name, ds))

        result = available[name]["function"](*arguments)
        _rename_xarray(result, name)
        return result

    # If you get this far without returning a variable then something has gone wrong
    raise ValueError("Can not calculate {} from dataset".format(name))


def _rename_xarray(array, name):
    array.rename(name)
    array.attrs["standard_name"] = name
    array.attrs["long_name"] = name


# A dictionary mapping variables that can be calculated to the functions to calculate
# them and the required input arguments (by CF standard names)
available = dict(
    thingness=dict(
        function=_thingness,
        arguments=["eastward_wind", "northward_wind", "air_temperature"],
    )
)
