# %% IMPORTING MODULES
######################################################
import sys
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# %% FUNCTION DICTIONARIES
######################################################
default_functions = {
    "near_surface": "default_function_here",
    "lcl": "default_function_here",
    "mixed_layer": "default_function_here",
    "cloud_layer": "default_function_here",
    "inversion": "default_function_here",
    "free_lower_troposphere": "default_function_here",
    "middle_troposphere": "default_function_here",
    "upper_troposphere": "default_function_here",
}


def estimate_level():  # Temporary function till actual functions are linked in to default_functions dictionary
    return 700


dictionary_of_functions_here = {
    "max_RH": estimate_level,
    "max_thetav_gradient": "function_here",
    "and_so_on": "for_individual_levels",
}  # temporary example filler till actual functions are linked

level_functions = {
    "near_surface": dictionary_of_functions_here,
    "lcl": dictionary_of_functions_here,
    "mixed_layer": dictionary_of_functions_here,
    "cloud_layer": dictionary_of_functions_here,
    "inversion": dictionary_of_functions_here,
    "free_lower_troposphere": dictionary_of_functions_here,
    "middle_troposphere": dictionary_of_functions_here,
    "upper_troposphere": dictionary_of_functions_here,
}

# %% TEMPLATE FUNCTION
######################################################


def pass_through_tests():
    return


def scalar(
    profile,
    level_name,  # e.g. "mixed_layer","near_surface"
    dim="height",  # Could also be "pressure"
    level_definition="default",  # preferred definition of the level (default value provided for all level_names)
    cell_type="bin",  # bin or point
    bounds=None,  # in case the upper and lower bounds are the same; in m / Pa
    upper=None,  # in m / Pa
    lower=None,  # in m / Pa
    cell_method="mean",
    drop_na=True,
):
    """
    Function takes a profile(s) and estimates a scalar quantity based on the specified level and method.
    User can specify by which definition they want the level to be calculated. Ideally, the profile is provided
    as a DataArray with altitude as dimension. 
    """

    try:
        default_functions[level_name]
    except KeyError as err:
        print(
            f"{err} : This level_name is not valid. Do you want to use custom_level instead?"
        )
        return

    try:
        level_functions[level_name][level_definition]
    except KeyError as err:
        print(
            f"{err} : This is not recognised as a level definition. Definitions available for {level_name} are {list(level_functions[level_name].keys)}"
        )
        return
    else :
        level_value = level_functions[level_name][level_definition]()


    if cell_type == "bin":
        if bounds is None :
            if upper is None or lower is None:
                raise Exception(
                    "if cell_type is bin, you must specify either upper and lower limits or provide bounds as a single value"
                )
                return 
            elif upper <= 0 or lower <= 0:
                raise Exception(
                    "upper and lower bounds for a cell have to be positive values"
                )
                return
            else :
                upper = level_value + upper
                lower = level_value - lower

        elif bounds is not None :
            
            if upper is not None or lower is not None:
                raise Exception(
                    "either provide the upper and lower limits or provide bounds as a single values"
                )
                return
            else :
                upper = level_value + bounds
                lower = level_value - bounds
        else :
            
    if cell_type == "point":
        if bounds is not None or upper is not None or lower is not None:
            raise Exception(
                "if cell type is point, there can be no specified bounds"
            )

    if drop_na is True:
        cell_method = "nan" + cell_method

    method = getattr(np, cell_method)

    if cell_type == "bin":
        scalar = method(profile.sel(height=slice(level - lower, level + upper)))
    elif cell_type == "point":
        scalar = method(profile.sel(height=level))

    return scalar


# %% TO DO LIST
######################################################

# 1. Link functions to estimate the vertical levels
# 2. Include space for arguments needed for functions mentioned in 1. to work
# 3. Include all tests as part of a different function and keep 'scalar' clean

# %%
