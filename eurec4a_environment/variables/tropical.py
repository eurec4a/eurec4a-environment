import xarray as xr

from .. import get_field
from .. import nomenclature as nom
from .utils import apply_by_column
from . import atmos


def lower_tropospheric_stability(
    ds,
    pot_temperature=nom.POTENTIAL_TEMPERATURE,
    pressure=nom.PRESSURE,
    vertical_coord=nom.ALTITUDE,
):
    """
    Lower tropospheric stability as defined by Klein & Hartmann 1993 is the difference
    in potential temperature between the surface and at 700hPa

    NB: currently picks the level closest to 700hPa
    TODO: add vertical linear interpolation for more accurate estimate

    reference: https://www.jstor.org/stable/26198436
    """

    def _calc_lts(ds_column):
        # swap from vertical coord so we can index by pressure
        ds_column = ds_column.swap_dims({vertical_coord: pressure})
        # and skip all the levels where pressure isn't given
        ds_column = ds_column.where(~ds_column.p.isnull(), drop=True)

        p_max = ds_column[pressure].max()
        p_ref = 700.0
        theta_surf = ds_column.sel(p=p_max)[pot_temperature]
        theta_ref = ds_column.sel(p=p_ref, method="nearest")[pot_temperature]

        return theta_ref - theta_surf

    da_theta = get_field(ds, name=pot_temperature, units="K")
    da_p = get_field(ds, name=pressure, units="hPa")

    if pressure in ds.coords:
        ds_derived = da_theta.to_dataset()
        ds_derived = ds_derived.assign_coords(**{pressure: da_p})
    else:
        ds_derived = xr.merge([da_theta, da_p])

    da_lts = apply_by_column(ds=ds_derived, vertical_coord=vertical_coord, fn=_calc_lts)
    da_lts.name = lower_tropospheric_stability.name
    da_lts.attrs["units"] = "K"
    da_lts.attrs["long_name"] = "lower tropospheric stability"
    da_lts.attrs["definition"] = "Klein & Hartmann 1993"
    return da_lts


lower_tropospheric_stability.name = "d_theta__lts"


def estimated_inversion_strength(
    ds,
    temperature=nom.TEMPERATURE,
    pressure=nom.PRESSURE,
    altitude=nom.ALTITUDE,
    vertical_coord=nom.ALTITUDE,
    LCL="z_LCL",
    LTS="dtheta_LTS",
):
    """
    Estimated inversion strength (EIS) as defined by Wood & Bretherton 2006.
    The expression is a refinement on the lower-tropospheric stability (LTS)
    taking into account the lapse-rate in the free troposphere and the
    decoupled (broken cumulus or stratiform covered) "decoupled-layer"

    reference: https://doi.org/10.1175/JCLI3988.1
    """
    # reference pressures used for interpolation
    da_p700 = xr.DataArray(700.0, attrs=dict(units="hPa"))
    da_p850 = xr.DataArray(850.0, attrs=dict(units="hPa"))

    def _calc_eis(ds_column):
        # swap from vertical coord so we can index by pressure
        ds_column_by_p = ds_column.swap_dims({vertical_coord: pressure})
        # and skip all the levels where pressure isn't given
        ds_column_by_p = ds_column_by_p.where(~ds_column_by_p.p.isnull(), drop=True)

        # altitude of LCL
        z_LCL = ds_column[LCL]
        # lower-tropospheric stability
        dtheta_LTS = ds_column[LTS]

        # altitude of 700hPa pressure level
        ds_700 = ds_column_by_p.interp(**{pressure: da_p700})
        z_700 = ds_700[altitude]

        ds_850 = ds_column_by_p.interp(**{pressure: da_p850})
        dTdz_850 = atmos.moist_adiabatic_lapse_rate(
            ds=ds_850, temperature=temperature, pressure=pressure
        )

        dtheta_EIS = dtheta_LTS - dTdz_850 * (z_700 - z_LCL)
        return dtheta_EIS

    da_T = get_field(ds, name=temperature, units="K")
    da_p = get_field(ds, name=pressure, units="hPa")
    if LCL not in ds.data_vars:
        raise Exception(
            "The LCL height is required to calculate EIS. Please provide it "
            f"in the dataset as `{LCL}` (or provide another name by setting "
            "the `LCL` kwarg). LCL can be calculated with the routines in "
            "`eurec4a_environment.variables.boundary_layer.lcl`"
        )
    da_LCL = ds[LCL]
    assert da_LCL.units == "m"

    if LTS not in ds.data_vars:
        raise Exception(
            "LTS is required to calculate EIS. Please provide it "
            f"in the dataset as `{LTS}` (or provide another name by setting "
            "the `LTS` kwarg). LTS can be calculated with "
            "`eurec4a_environment.variables.tropical.lower_tropospheric_stability`"
        )
    da_LTS = ds[LTS]
    assert da_LTS.units == "K"

    ds_derived = xr.merge([da_T, da_p, da_LCL, da_LTS])

    if pressure in ds.coords:
        ds_derived = ds_derived.assign_coords(**{pressure: da_p})
    else:
        ds_derived[da_p.name] = da_p

    # this will happend when `altitude != vertical_coord`
    if not altitude in ds_derived:
        ds_derived[altitude] = ds[altitude]

    da_eis = apply_by_column(ds=ds_derived, vertical_coord=vertical_coord, fn=_calc_eis)
    da_eis.name = "d_theta__eis"
    da_eis.attrs["units"] = "K"
    da_eis.attrs["long_name"] = "estimated inversion strength"
    da_eis.attrs["definition"] = "Wood & Bretherton 2006"
    return da_eis
