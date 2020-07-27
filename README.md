# Characterising the EUREC4A field campaign environment

![eurec4a_env](https://github.com/leifdenby/eurec4a-environment/workflows/eurec4a_env/badge.svg)

The aim of this git repository is collect tools and ideas for characterising
the atmospheric environment during the EUREC4A field campaign.

# Variables, definitions and source data

> **NOTE**: The tables below are likely out-of-date (but will be updated at the end of the hackathon), see the [project document](https://docs.google.com/document/d/17zO3mNVzYluToaUERtfwHpta0YG_HF_Zpyul8ldbhXY/edit#) for a more up-to-date list and discussion.

## Column-based scalars
(could be averaged over flight, full-circle, single sounding etc)

| variable | description | definition | data sources | implementation |
| --- | --- | --- | --- | --- |
| h_BL | boundary layer depth | numerous definitions (**add links**) | thermodynamic profiles | |
| h_CB | cloud-base height | numerous definitions (**add links**) | thermodynamic profiles | |
| SST | sea surface temperature | - | ship observations (?), ERA reanalysis | |
| EIS | estimated inversion strength | ? | thermodynamic profiles | |
| LTS | lower tropospheric stability | ? | thermodynamic profiles | |
| PW | precipitable water | ? | thermodynamic profiles | |
| FT humidity | free tropospheric humidity | ? | thermodynamic profiles | |
| "wind speed" | lower tropospheric wind magnitude? | - | horizontal wind profiles | |
| "wind shear" | lower tropospheric wind shear magnitude? | - | horizontal wind profiles | |
| z_INV | "inversion height" (multiple?) | ? | thermodynamic profiles | |
| ? | "mesoscale organisation category" | fish/flower/sugar/gravel | satellite observations | |
| I_org | mesoscale organisation | Tompkins & Semie 2017 | thresholded cloud-field measurement, for example cloud-top height | https://github.com/leifdenby/convorg |
| SCAI | mesoscale organisation | Tobin et al 2012 | thresholded cloud-field measurement, for example cloud-top height | https://github.com/leifdenby/convorg |
| LHF | latent heat flux | Modified COARE algorithm | thermodynamic profiles + surface instruments | |
| SHF | sensible heat flux | Modified COARE algorithm | thermodynamic profiles + surface instruments | |
| SBF | surface buoyancy flux | Modified COARE algorithm | thermodynamic profiles + surface instruments | |

## Profile variables

| variable | short-hand | observation sources |
| --- | --- | --- |
| qt(z) | total water vertical profile | JOANNE + radiosondes |
| dQdt_r(z) | radiative cooling profiles | radiative transfer calculations based on moisture and temperature profiles (`Products/Radiative Profiles/all_rad_profiles.nc` on AERIS) |
| theta_v(z) | virtual potential temperature profile | JOANNE + radiosondes |
| u(z) | zonal wind profile | JOANNE + radiosondes, wind lidars |
| v(z) | meridonal wind profile | JOANNE + radiosondes, wind lidars |
| w(z) | vertical wind profile | Lidars, radars (?) |
| W(z) | large-scale vertical velocity profile | JOANNE |
