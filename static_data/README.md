# Static data

| File | What it is | Source |
|---|---|---|
| `penmanshiel_turbines.csv` | Turbine identifiers, coordinates, ground elevation, model, rating, hub height, rotor diameter and commercial operation date | Tidy copy of `Penmanshiel_WT_static.csv` from Zenodo record 5946808 (Cubico Sustainable Investments, CC BY 4.0) |
| `mm82_power_curve.csv` | Senvion (REpower) MM82 power curve at standard air density, 0.5 m/s steps, scaled to the 2,050 kW rating | Normalised curve `REpower.MM82.2000` from the renewables.ninja virtual wind farm model (github.com/renewables-ninja/vwf, Iain Staffell, BSD 3-Clause), unsmoothed version |
| `era5_golden_monthly.csv` | Monthly means of hourly ERA5 global horizontal irradiance and 2 m temperature at Golden, Colorado, 2000 to 2024 (long-term reference for the independent solar estimate) | Copernicus Climate Change Service ERA5, through the Open-Meteo archive API (latitude 39.742, longitude -105.1727, `models=era5`), retrieved 25 October 2026 and averaged by month |
| `pvdaq_system50_daily_sample.csv` | Every third day of 2012 and 2013 for NREL PVDAQ system 50: minutes recorded, daily plane-of-array irradiation from NREL's on-site sensor and daily energy of output channel 2 (the stream analysed here) | NREL PVDAQ public data lake on OEDI (`pvdaq/csv/pvdata/system_id=50/`, CC BY 4.0), daily sums of the 1-minute files, retrieved 25 October 2026 |

