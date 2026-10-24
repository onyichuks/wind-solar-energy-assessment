# The datasets

This project uses three public datasets. None of them is stored in the repository; `python scripts/get_data.py` downloads them into `data/raw/` and `python scripts/prepare_data.py` turns the large turbine files into compact Parquet files in `data/processed/`.

| Dataset | What it is used for | Source and licence | Size |
|---|---|---|---|
| Penmanshiel wind farm data | Turbine SCADA, revenue meter and wind farm controller meter: the "virtual met mast", the actual production and the measured losses | Zenodo record [5946808](https://doi.org/10.5281/zenodo.5946808), published by Cubico Sustainable Investments, Creative Commons Attribution 4.0 (CC BY 4.0) | about 1.05 GB zipped |
| ERA5 reanalysis at the site | The 25-year long-term wind reference, air density and the month-by-month wind index | ERA5 from the Copernicus Climate Change Service, downloaded through the [Open-Meteo](https://open-meteo.com) historical weather API (CC BY 4.0) | 9 MB |
| NREL PVDAQ system 50 | The solar PV performance module | NREL Photovoltaic Data Acquisition (PVDAQ) public dataset through the U.S. Department of Energy Open Energy Data Initiative, as packaged in the example data of the open-source [PVAnalytics](https://github.com/pvlib/pvanalytics) library (BSD 3-Clause) | 1 MB |

Attribution: *Contains Penmanshiel wind farm data published by Cubico Sustainable Investments under CC BY 4.0; ERA5 reanalysis data (Copernicus Climate Change Service) accessed through Open-Meteo; and NREL PVDAQ data via PVAnalytics.*

Every check described below can be repeated with `python scripts/check_data.py`, which writes `outputs/tables/q01_scada_checks_by_turbine.csv`, `q02_meter_checks_by_month.csv`, `q03_meter_events.csv` and a summary in `outputs/data_quality.json`.

---

## 1. Penmanshiel wind farm

### Why this wind farm

Penmanshiel is one of very few operating wind farms whose full SCADA record has been published. It has 14 turbines, a revenue grid meter, a wind farm controller meter with the power setpoint, and turbine status logs. Together these allow a real test of a pre-construction estimate: most losses an energy yield assessment assumes can be measured.

This project uses version 0.0.2 of the Zenodo record (February 2022). The downloads are checked against the MD5 checksums published on the record page. A later version (v3, August 2025, record [16807304](https://doi.org/10.5281/zenodo.16807304)) adds data up to 2024. Its 2017 and 2018 SCADA files are identical (same checksums); its meter files are extended.

### The site

| Item | Value |
|---|---|
| Location | Scottish Borders, United Kingdom, a few kilometres inland from the North Sea coast (centre about 55.90 N, 2.29 W) |
| Turbines | 14 Senvion MM82 (numbered T01, T02, T04 to T15; there is no T03) |
| Rated power | 2,050 kW each, 28.7 MW in total |
| Hub height and rotor | 59 m hub, 82 m rotor diameter |
| Ground elevation | 180 m (T07) to 228 m (T15) |
| Commercial operation | 1 September 2016 |
| Turbine type notes | Pitch-regulated, variable speed. This matters for the air density correction of the power curve and for the thrust curve used in the wake models |

### Files used

| File (inside the zip) | Time step | Period used | What it holds |
|---|---|---|---|
| `Turbine_Data_Penmanshiel_<NN>_<year>...csv` (four SCADA zips, 2017 and 2018) | 10 minutes | 1 January 2017 to 31 December 2018 | About 300 signals per turbine. 20 are kept (listed below) |
| `Status_Penmanshiel_<NN>_<year>...csv` | Event log | 2017 and 2018 | Start, end, code, message and IEC availability category of every turbine state change (165,619 events). Used to check the turbine health flags: stop, warning and partial-performance messages on flagged days |
| `Device_Data_Penmanshiel_Grid_Meter...csv` (Grid zip) | 30 minutes, written as three equal 10-minute rows | July 2016 to September 2019 | Revenue meter energy export. No data after September 2019 |
| `Device_Data_Penmanshiel_PMU...csv` (PMU zip) | 10 minutes | April 2018 to June 2021 | Wind farm controller meter energy and power, and the wind farm power setpoint (from December 2018) |
| `Penmanshiel_WT_static.csv` | n/a | n/a | Turbine coordinates, ground elevation, model, rated power, hub height and rotor diameter (a tidy copy is in `static_data/penmanshiel_turbines.csv`) |

### Turbine signals kept (10-minute averages unless stated)

| Greenbyte signal | Name in this project | Unit | Used for |
|---|---|---|---|
| Wind speed | `wind_ms` | m/s | Data checks, virtual met mast, terrain factors, power curves, turbine health model |
| Wind speed, Standard deviation | `wind_std_ms` | m/s | Turbulence intensity for the wake models and the turbine health model |
| Wind direction | `wind_dir_deg` | degrees | Data checks, wind farm direction (nacelle position plus vane), turbine health model |
| Power | `power_kw` | kW | Data checks, power curves, wake efficiency, downtime check, turbine health model (target) |
| Energy Export | `energy_kwh` | kWh per interval | Turbine energy, electrical losses |
| Lost Production to Downtime | `lost_downtime_kwh` | kWh | Availability (operator figure); filters normal operation |
| Lost Production to Performance | `lost_performance_kwh` | kWh | Reported alongside the power curve result; check on the turbine health flags |
| Nacelle ambient temperature | `temp_c` | C | Not used as a model input: missing for weeks of 2018 (all of March) but never in 2017, so the turbine health model uses ERA5 air temperature instead |
| Lost Production to Curtailment (Total, Grid) | `lost_curtailment_kwh`, `lost_curtail_grid_kwh` | kWh | Curtailment step of the gap analysis (all zero in 2017 and 2018) |
| Turbine Power setpoint | `setpoint_kw` | kW | Inspected; it is a controller value, not a curtailment flag |

Also kept for context and future work, not used in the results: wind speed sensors 1 and 2, nacelle position, power minimum and maximum, potential power (default curve), energy theoretical, pitch angle (only from April 2018) and rotor speed.

### Known data issues and how they are handled

| Issue | What was found | Handling |
|---|---|---|
| Timestamps | 10-minute rows stamped at the start of each interval, in UTC all year: every day has 144 rows per turbine, and the best match with ERA5 comes at the same lag in summer and winter (20 to 30 minutes) | Hourly averages are centred on the hour, to match ERA5 values, which are instantaneous at the hour |
| Missing intervals | 0.2 to 1.1 percent per turbine (T02 highest); no duplicate timestamps | Kept as empty rows and excluded; 99.6 percent of intervals pass the checks |
| Out-of-range readings | No impossible power or wind values | Rules kept in `operational.apply_qc` so other periods are checked the same way |
| Stuck wind vane | T13 reports a wind direction of exactly 172.05 degrees for 17 hours on 16 and 17 August 2017, while stopped, with its second anemometer showing an error code (-15 m/s). No frozen wind speed anywhere | The frozen-reading check (wind speed or direction unchanged for an hour or more) flags it and those intervals are excluded |
| Energy counters | Turbine energy per 10 minutes is a counter difference: single intervals can show more than rated energy (up to 444 kWh against 342 kWh at rated power), but totals agree with average power to within 0.1 percent for every turbine | Used only in totals over a day or longer |
| Anemometers and vanes | The two nacelle anemometers on each turbine agree to within about 3.5 percent, with the same seasonal pattern in both years and no drift. Each turbine's median direction is within 2 degrees of the farm mean, and its monthly value moves by at most 5 degrees with no step changes | No correction needed |
| No met mast | The open data have no meteorological mast | A virtual mast from the nacelle anemometers of undisturbed turbines (IEC 61400-12-1 Annex A sectors). Recovery 99.7 percent of hours |
| Nacelle anemometry | Measured behind the rotor; the turbine controller applies its own transfer function | Accepted for the resource and the terrain factors, with a larger measurement uncertainty (3 percent of wind speed) than a met mast |
| Direction offset | Turbine wind direction reads anticlockwise of ERA5: median 7.7 degrees, mean 6.5 degrees | Site direction used for the wind rose and wakes (as with a site mast); ERA5 direction shifted by the median offset where it stands in for the site |
| Corrupt revenue meter day | 20 May 2017: the meter records 121.6 MWh against 93.3 MWh from the turbine meters. Every other day is within about 6 percent of the median ratio | Found by a daily check against the turbine meters (`operational.meter_day_check`). The day's export is replaced by turbine energy less the median electrical loss, and the day is left out of the electrical loss measurement |
| Electrical loss step | The meter-to-turbine ratio shifts around May 2017: monthly electrical loss is 0.6 to 0.8 percent before, 1.0 to 1.3 percent after | Not corrected; the measured electrical loss is the 2017 and 2018 average |
| Grid meter gap | No revenue meter data from October 2019 | Filled from the wind farm controller meter, which agrees with the grid meter to 0.2 percent where both report (scaling factor 1.0020) |
| Controller meter counter gap | 21 April to 16 May 2018: the controller meter repeats one energy value (1,309 kWh per 10 minutes), a counter gap spread evenly | Left out of the scaling factor; the grid meter covers these dates |
| Whole-farm outages | 26 September to 9 October 2020 (14 days) and 9 to 10 June 2020: controller meter energy and plant power are zero while ERA5 shows wind (7.1 m/s on average in the long outage) | Real outages, kept in the actual production. The long one cost roughly 2,500 MWh and explains most of the shortfall in September and October 2020 |
| Turbine power setpoint | Tracks power most of the time | Not used for curtailment; the wind farm setpoint from the controller meter is used instead |
| Open-ended status events | Some events have no end time ("-") | Read as missing end times |

### What the data can and cannot support

They support an operational assessment: availability, electrical losses, curtailment, power curves, terrain speed-up between turbine positions, wake losses by direction (estimated from nacelle anemometry) and long-term corrected production. With the virtual mast they also support a pre-construction style estimate for comparison.

They cannot support warranty power curve testing (no calibrated met mast or lidar) or a true pre-construction campaign (the virtual mast comes from operating turbines). This study used turbine SCADA for 2017 and 2018 only. From 2019 the meters alone show the 2020 outages, but the rest of the lower ratio in those years is not explained here (later turbine data are in the Zenodo record).

---

## 2. ERA5 reanalysis (long-term reference)

Reanalysis products are global weather models fitted to historical observations. They give a long, consistent record that puts one year of site measurements into long-term context.

| Item | Value |
|---|---|
| Grid point | 55.75 N, 2.25 W (the nearest land point; Open-Meteo reports its model elevation as 181 m) |
| Period | 1 January 2000 to 31 December 2024, hourly, UTC |
| Variables | Wind speed at 10 m and 100 m (m/s), wind direction at 100 m (degrees), air temperature at 2 m (C), surface pressure (hPa) |
| Request | See `OPEN_METEO` in `scripts/get_data.py` |

Notes:
* The 10 m and 100 m speeds give an hourly shear exponent (median 0.19), used only for the reanalysis-only comparison.
* Air density at hub height (1.213 kg/m3 long-term) comes from the 2 m temperature and surface pressure, moved to the mean turbine elevation plus hub height with the standard lapse rate.
* Year-to-year variability of the annual mean wind is 3.9 percent, used for the 1, 10 and 20-year P values.
* ERA5 data are occasionally revised, so a fresh download can differ very slightly from the file used here.

---

## 3. NREL PVDAQ system 50 (solar PV, a separate site in Golden, Colorado)

Penmanshiel has no solar, so the solar module uses this public, well-documented system instead.

| Item | Value |
|---|---|
| System | Fixed-tilt PV array at NREL, Golden, Colorado (39.7406 N, 105.1775 W) |
| Orientation | Tilt 45 degrees, azimuth 158 degrees (south-south-east), as published with the PVAnalytics example |
| Size | NREL lists the whole system at 6.0 kW DC (Siemens M55 modules, installed March 1996). The stream used here is one of two output channels and carries 50.0 to 50.7 percent of the system's AC energy (eight days checked, 2011 to 2013), so its rated size is taken as 3.0 kW DC. The clear-sky fit gives an effective 2.79 kW |
| Power data | `system_50_ac_power_2_full_DST.parquet`: AC power of output channel 2 in watts, every 15 minutes, 15 April 2011 to 31 December 2013 (matches NREL's 1-minute channel 2 data to within 0.01 percent in daily energy) |
| Weather data | `system_50_ac_power_2_full_DST_psm3.parquet`: NSRDB PSM3 satellite irradiance (global horizontal and clear-sky components) and air temperature, every 30 minutes, 2011 to 2013 |
| Long-term irradiance | `static_data/era5_golden_monthly.csv`: ERA5 monthly mean irradiance and temperature at the site, 2000 to 2024 (Open-Meteo archive), for the independent estimate |
| On-site sensor sample | `static_data/pvdaq_system50_daily_sample.csv`: NREL's plane-of-array irradiance sensor and channel 2 energy, daily sums of 1-minute data on every third day of 2012 and 2013 (NREL PVDAQ data lake on OEDI). The sensor often reads zero in 2012, so 171 of 243 days are usable |

### Known data issues and how they are handled

| Issue | What was found | Handling |
|---|---|---|
| Daylight saving clock | Production is centred an hour late every summer: the record starts on daylight time (15 April 2011) and switches on 6 November 2011, 12 March 2012, 4 November 2012, 10 March 2013 and 3 November 2013 | Detected with PVAnalytics (ruptures change points, production midpoint against modelled solar noon) and corrected |
| Timestamp convention | Values appear to be stamped at the end of each 15-minute interval | Moved back one step: the clear-sky fit improves from R squared 0.810 to 0.825 |
| Missing values | 3 percent of intervals, including part of April and May 2012 | Excluded from both actual and expected energy |
| Stale values | Checked in daylight with PVAnalytics | Excluded |
| Satellite irradiance | 4 km, 30-minute satellite estimates, not a plane-of-array sensor | Expected power has wider scatter than with on-site sensors. Checked against NREL's on-site sensor on sampled days: the sensor reads 1.5 percent more overall and 0.5 percent less on clear days |
| Snow on the ground | Likely raises light reflected onto the steep array in winter (no snow records were used to confirm it) | Not modelled (fixed ground reflectance); the most likely reason the winter performance index sits above one |
| Air temperature floor | Air temperature never goes below 0 C in the weather file: 23.5 percent of values are exactly 0 (about two-thirds of December to February values, none from June to September), so sub-zero readings appear to have been set to zero | Kept. It barely matters: setting those readings to -5 C would raise expected energy by about 0.1 percent |
| Peak power above fitted capacity | The highest AC readings (up to 3.37 kW, against 2.79 kW fitted DC capacity) all come between October and April, most in February and March | Consistent with sun near normal incidence on the steep array, cold modules and light reflected from snow; kept, and the capacity fit uses clear-sky periods across all seasons |

### What the data can and cannot support

They support availability, outage and snow loss analysis, clipping checks, time-shift detection, an expected-energy comparison and an independent energy estimate. The rated size is inferred from NREL's listing and the channel split, so it carries some uncertainty; performance ratios in the performance review are relative to the fitted capacity. At under three years, the record is too short for a reliable degradation rate. This is a small research system; the same methods apply to utility-scale plants at inverter level.
