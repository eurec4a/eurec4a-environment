import xarray as xr

from .. import get_field
from ..constants import cp_d, Rd, g, eps
from .. import nomenclature as nom
from .utils import apply_by_column


def density(
    ds,
    pres=nom.PRESSURE,
    temp=nom.TEMPERATURE,
    specific_humidity=nom.SPECIFIC_HUMIDITY,
    altitude=nom.ALTITUDE,
):
    """
    Calculate the mixture density (for dry air with water vapour)

    equation: rho = P/(Rd * Tv), where Tv = T(1 + mr/eps)/(1+mr)
    """

    da_p = get_field(ds=ds, name=pres, units="Pa")
    da_temp = get_field(ds=ds, name=temp, units="K")
    da_qv = get_field(ds=ds, name=specific_humidity, units="g/kg")

    da_rv = da_qv / (1 - da_qv)

    da_rho = (da_p) / (Rd * (da_temp) * (1 + (da_rv / eps)) / (1 + da_rv))
    da_rho.attrs["long_name"] = "density of air"
    da_rho.attrs["units"] = "kg/m3"

    return da_rho


def potential_temperature(
    ds,
    temperature=nom.TEMPERATURE,
    pressure=nom.PRESSURE,
    p_surf=None,
    vertical_coord=None,
):
    """
    Calculate potential temperature from presure and temperature, using either
    `p_surf` as the reference pressure (in hPa) or the maximum pressure along the
    `vertical_coord` coordinate.

    NB: uses heat capacity and gas constant for dry air
    """
    da_temp = get_field(ds, name=temperature, units="K")
    da_p = get_field(ds, name=pressure, units="hPa")

    if not (p_surf or vertical_coord) or (p_surf and vertical_coord):
        raise Exception("Please provide one of `p_surf` or `vertical_coord`")

    if p_surf:
        p_ref = p_surf
        if isinstance(p_surf, xr.DataArray):
            # implement unit conversion here if necessary
            assert p_surf.attrs.get("units") == "hPa"
    else:
        p_ref = da_p.max(dim=vertical_coord)

    da_theta = da_temp * (p_ref / da_p) ** (Rd / cp_d)
    da_theta.name = nom.POTENTIAL_TEMPERATURE
    da_theta.attrs["units"] = "K"
    da_theta.attrs["long_name"] = "potential temperature"

    return da_theta


def hydrostatic_pressure(
    ds,
    p_surf,
    temperature=nom.TEMPERATURE,
    vertical_coord=nom.ALTITUDE,
    specific_humidity=nom.SPECIFIC_HUMIDITY,
):
    """
    Calculate potential temperature from presure and temperature, using either
    `p_surf` as the reference pressure (in hPa) or the maximum pressure along the
    `vertical_coord` coordinate.

    NB: uses heat capacity and gas constant for dry air
    """
    # pressure variable will be named with standard nomenclature
    pressure = nom.PRESSURE

    if not isinstance(p_surf, xr.DataArray):
        raise Exception(
            "Please provide the surface reference pressure (`p_surf`) with the "
            "`units` attribute set"
        )

    assert p_surf.attrs.get("units") == "Pa"

    def _integrate_hydrostatic_pressure(ds_column):
        z_levels = ds_column.sortby(vertical_coord)[vertical_coord]
        z0 = z_levels.min()

        ds_surf = ds_column.sel(alt=z0)
        ds_surf[pressure] = ds_column.p_ref

        levels = [ds_surf]
        for z_ in z_levels.values[1:]:
            ds_prev = levels[-1]
            dz = z_ - ds_prev[vertical_coord]

            rho = density(
                ds=ds_prev,
                pres=pressure,
                temp=temperature,
                specific_humidity=specific_humidity,
            )

            # dp/dz = - rho * g
            dp = -rho * g * dz
            ds_level = ds_column.sel(**{vertical_coord: z_})
            ds_level[pressure] = ds_prev[pressure] + dp
            ds_level[pressure].attrs["units"] = "Pa"
            levels.append(ds_level)

        ds_column_integrated = xr.concat(levels, dim=vertical_coord)
        da_p = ds_column_integrated[pressure]
        da_p.attrs["long_name"] = "pressure"
        da_p.attrs["units"] = "Pa"
        return da_p

    ds_subset = ds[[temperature, specific_humidity]]
    # ensure that the vertical coordinate measures height in meters and is available
    ds_subset[vertical_coord] = get_field(ds=ds, name=vertical_coord, units="m")
    # XXX: need to set this value for all columns
    ds_subset["p_ref"] = p_surf

    # calculate pressure at every height level
    da_pressure = apply_by_column(
        ds=ds_subset, vertical_coord=vertical_coord, fn=_integrate_hydrostatic_pressure
    )

    return da_pressure
