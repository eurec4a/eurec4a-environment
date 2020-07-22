# Characterising the EUREC4A field campaign environment

The aim of this git repository is collect tools and ideas for characterising
the atmospheric environment during the EUREC4A field campaign.

# Variables, definitions and source data

## Column-based scalars
(could be averaged over flight, full-circle, single sounding etc)

| variable | short-hand | definition | data sources |
| --- | --- | --- | --- |
| h_BL | boundary layer depth | numerous definitions (**add links**) | thermodynamic profiles |
| h_CB | cloud-base height | numerous definitions (**add links**) | thermodynamic profiles |
| SST | sea surface temperature | - | ship observations (?), ERA reanalysis |
| EIS | estimated inversion strength | ? | thermodynamic profiles |
| LTS | lower tropospheric stability | ? | thermodynamic profiles |
| FT humidity | ? | ? | ? |
| "wind speed" | lower tropospheric wind magnitude? | - | horizontal wind profiles |
| "wind shear" | lower tropospheric wind shear magnitude? | - | horizontal wind profiles |
| z_INV | "inversion height" (multiple?) | ? | thermodynamic profiles |

## Profile variables

| variable | short-hand | observation sources |
| --- | --- | --- |
| qt(z) | total water vertical profile | HALO dropsondes, ... |
| dQdt_r(z) | radiative cooling profiles | radiative transfer calculations based on moisture and temperature profiles (`Products/Radiative Profiles/all_rad_profiles.nc` on AERIS) |
| theta_v(z) | virtual potential temperature profile | HALO dropsondes, ... |
| u(z) | zonal wind profile | HALO dropsondes? |
| v(z) | meridonal wind profile | HALO dropsondes? |
| w(z) | vertical wind profile | HALO dropsondes? |
