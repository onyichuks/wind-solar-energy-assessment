# Testing a pre-construction energy yield assessment against operation at the Penmanshiel wind farm, with machine learning turbine health monitoring, and a separate solar PV performance review in Golden, Colorado

Onyi Chuks, Calgary, Alberta. October 2026.

Code, data pipeline and interactive dashboard: [github.com/onyichuks/wind-solar-energy-assessment](https://github.com/onyichuks/wind-solar-energy-assessment). A one-page [executive summary](EXECUTIVE_SUMMARY.md) gives the findings and what they mean. Every number in this report is produced by the code in that repository and stored in `outputs/`. The datasets, and every data issue found, are described in [DATASET.md](DATASET.md). The machine learning is examined in more depth (training curves, learning curve, baselines, error breakdown and a fault injection test) in the [machine learning technical note](ML_TECHNICAL_NOTE.md).

---

## Abstract

Pre-construction energy yield assessments decide whether wind farms are financed, yet they are rarely checked against what the farm later produces, and almost never in public. This study builds a pre-construction style assessment for the Penmanshiel wind farm (14 Senvion MM82 turbines, 28.7 MW, Scottish Borders) from information a developer would have had: one year of site wind, 25 years of ERA5 reanalysis, a library power curve and engineering wake models. It then tests the estimate against 58 months of metered production and uses turbine SCADA to explain the difference. The estimate gives a net P50 of 70.3 GWh per year. The farm produced 9.1 percent more than predicted for the same weather, which sits at about the P18 level of the estimate's own uncertainty. For 2017 and 2018, measured losses explain 79 percent of the gap: wake losses were about 4.5 percent against 9.8 percent modelled (47 percent of the gap), and availability, electrical losses and the power curve were all better than assumed. A machine learning normal-behaviour model of each turbine, trained on 2017 and tested on 2018, predicts power with 22 percent lower mean absolute error than each turbine's own power curve and nearly halves the spread of daily results, so it separates farm-wide shortfalls (an external power limit on 14 January 2018) from single-turbine underperformance (T06, about 3 percent short through June 2018). A separate solar module, on a different site, reviews a small monitored PV system at NREL in Golden, Colorado, against a physical model: 98.8 percent daylight availability, 0.75 percent outage loss and 0.72 percent suspected snow loss. An independent estimate of that system's energy, made the way a desk study would from its rated size, generic losses and long-term irradiance, came out 20.6 percent below actual production in 2012 and 2013, outside its own uncertainty; the satellite irradiance was within 1.5 percent of NREL's on-site sensor, so the cause was the generic loss and ageing assumptions. Thorough data checks found a corrupt day of revenue meter data, a stuck wind vane, a meter counter gap and a 14-day outage of the whole farm.

---

## 1. Introduction

### 1.1 Background

Before a wind farm is built, an energy yield assessment estimates its long-term annual production. A typical assessment measures the wind on site for a year or more, relates that year to a long-term reference to remove the effect of an unusually windy or calm year, models how the wind varies across the site and how turbines shelter each other (wakes), applies a chain of losses, and states the uncertainty as P50 and P90 values: the annual energy expected to be exceeded with 50 and 90 percent probability. Lenders size debt on the P90, so a biased estimate costs money in either direction.

Assessments can only be improved by comparing them with operation, but operating data are commercially sensitive and few complete records are public. The Penmanshiel wind farm is an exception: its owner, Cubico Sustainable Investments, has published turbine SCADA, meter data and status logs under an open licence. This allows every major assumption of an assessment to be measured after the fact.

Operating data also support a second task that matters to owners and performance engineers: spotting turbines that produce less than they should. Normal-behaviour models, which learn what a healthy turbine produces under given conditions and flag departures from it, are an established way of doing this from SCADA data (Schlechtingen and Santos, 2011; Tautz-Weinert and Watson, 2017).

### 1.2 Objectives

The study set out to answer six questions:

1. **Data quality.** Are the open datasets fit for this analysis, and what issues must be handled?
2. **The estimate.** What would a pre-construction style assessment have predicted, using only information available before construction?
3. **The test.** How did actual production compare with that estimate, and was the difference within the stated uncertainty?
4. **The explanation.** Which assumptions account for the difference, measured from SCADA?
5. **Machine learning.** Can a normal-behaviour model predict each turbine's power better than a power curve, and does it pick out underperformance worth investigating?
6. **Solar (a separate site).** How well would an independent, pre-construction style estimate have predicted the energy of a monitored PV system at NREL in Golden, Colorado, how did the system perform against a physical model, and how much energy was lost to outages, snow and clipping?

---

## 2. Data

| Dataset | Use | Source |
|---|---|---|
| Penmanshiel wind farm: turbine SCADA (10-minute, 2017 and 2018), status logs, revenue meter (2016 to 2019) and wind farm controller meter (2018 to 2021), turbine table | Virtual met mast, actual production, measured losses, turbine health model | Cubico Sustainable Investments, Zenodo record 5946808 (version 0.0.2), CC BY 4.0 |
| ERA5 reanalysis at the site, hourly, 2000 to 2024 | Long-term wind reference, air density, monthly energy index | Copernicus Climate Change Service (Hersbach et al., 2020), through the Open-Meteo archive |
| NREL PVDAQ system 50 in Golden, Colorado (not at Penmanshiel): AC power (15-minute) and NSRDB satellite weather, 2011 to 2013 | Solar performance review | NREL PVDAQ, as packaged in the PVAnalytics example data |
| ERA5 reanalysis at Golden, Colorado, monthly means of hourly irradiance and temperature, 2000 to 2024 | Long-term irradiance reference for the independent solar estimate | Copernicus Climate Change Service, through the Open-Meteo archive |
| NREL PVDAQ system 50 metadata, and the on-site plane-of-array irradiance sensor and output channel 2 on every third day of 2012 and 2013 | Rated size; check on the satellite irradiance | NREL PVDAQ public data lake on OEDI, CC BY 4.0 |
| Senvion MM82 power curve | Reference power curve | renewables.ninja library (Staffell and Pfenninger, 2016) |

The site has 14 turbines (T01, T02, T04 to T15) with 59 m hubs and 82 m rotors on ground between 180 m and 228 m. Commercial operation began on 1 September 2016. Full details, file lists and signal definitions are in [DATASET.md](DATASET.md).

---

## 3. Methods

### 3.1 Data checks

Downloads are verified against the MD5 checksums on the Zenodo record. Each dataset is checked for completeness, duplicate timestamps, clock problems (including daylight saving time), out-of-range values and frozen readings (wind speed or direction unchanged for an hour or more). The two meters are compared where both report, and the revenue meter is compared day by day with the summed turbine meters to find corrupt days. ERA5 is checked for gaps and trend. The checks run as `scripts/check_data.py`.

### 3.2 Pre-construction style energy yield assessment

* **Virtual met mast.** The open data have no met mast, so one is built from the turbines. For each wind direction, the IEC 61400-12-1 Annex A method marks a turbine as disturbed when a neighbour within 20 rotor diameters lies upwind (sector width 1.3 arctan(2.5 D/L + 0.15) + 10 degrees; IEC, 2017). Every 10 minutes the mast speed is the mean nacelle wind speed of the undisturbed turbines that are available and pass the data checks. Calendar 2017 is the 12-month measurement campaign. Hourly averages are centred on the hour to match ERA5, whose values are instantaneous.
* **Long-term correction.** Four measure-correlate-predict methods were compared by fitting on one measured year and predicting the other. The method used scales the measured site distribution, sector by sector, by the ratio of predicted long-term to predicted concurrent mean wind (from a sector-wise linear fit), then reweights sectors to their long-term frequency. The others synthesise an hourly site series from ERA5: sector-wise linear regression, the variance ratio method, and a machine learning method (gradient-boosted trees, scikit-learn) that predicts hourly site wind from ERA5 100 m and 10 m wind, direction, shear, temperature, pressure, time of day, season and the 100 m wind an hour either side. The machine learning settings were fixed in advance; further checks used fewer inputs and a spread correction (quantile mapping onto the measured distribution, built from out-of-fold predictions within the fit year).
* **Energy and wakes.** The MM82 power curve is shifted to the long-term site air density with the IEC cube-root rule, keeping cut-out at 25 m/s. DTU's PyWake runs four engineering wake models on the real layout over a 5 degree by 0.5 m/s frequency table: the Jensen (Park) model with the classic onshore wake decay constant k = 0.075 as the central case (Jensen, 1983; Katić et al., 1986), and the Bastankhah Gaussian (Bastankhah and Porté-Agel, 2014), TurbOPark (Nygaard et al., 2020) and Niayifar (Niayifar and Porté-Agel, 2016) models as sensitivities. The last two use turbulence intensity measured on the undisturbed nacelle anemometers (12.6 percent at 10 m/s). No MM82 thrust curve is public, so PyWake's generic turbine of the same size supplies it.
* **Losses and uncertainty.** A loss chain with typical pre-construction assumptions turns gross energy into net. Uncertainties are combined by root sum of squares, with wind speed terms converted to energy by the modelled sensitivity ratio, and interannual variability added for 1, 10 and 20-year P values.

### 3.3 Predicted against actual

The P50 is turned into a monthly prediction with an energy index: ERA5 wind each hour, moved to the site with a sector-wise linear fit, is run through the same wake-modelled farm power table. Each month's prediction is the P50 scaled by that month's index over the index's own 25-year mean, so any level bias in the fit cancels. Actual energy is the revenue meter, filled from the controller meter from October 2019 (scaled by 1.0020, the ratio of the two meters where both report). Every complete month from September 2016 to June 2021 is compared, with a regression of monthly energy on the index as a cross-check (in the style of OpenOA; Perr-Sauer et al., 2021).

### 3.4 Explaining the gap

For 2017 and 2018, when turbine SCADA is available, each assumption is replaced in turn by its measured value, and the residual is what is left:

* availability from the operator's lost production to downtime, checked with an independent count of full stops;
* electrical losses from the summed turbine meters against the revenue meter;
* the power curve from the fleet's measured nacelle curve (IEC method of bins) against the library curve;
* terrain speed-up from each turbine's free-stream nacelle wind against the other undisturbed turbines;
* wake losses from actual power against terrain-adjusted expected unwaked power at 5 to 11 m/s, by 30-degree sector, scaled onto the model's all-speed loss;
* curtailment from the operator's turbine records.

The end point uses the 2017 and 2018 production ratio, so losses and production cover the same period.

### 3.5 Machine learning normal-behaviour model

* **Model.** One gradient-boosted tree model per turbine (scikit-learn HistGradientBoostingRegressor; Friedman, 2001; Pedregosa et al., 2011) predicts 10-minute power from six inputs: nacelle wind speed, turbulence intensity (standard deviation over mean wind speed), wind direction (as sine and cosine), and ERA5 air temperature and air density. ERA5 is used for temperature because the nacelle temperature signal is missing for weeks of 2018 (all of March, for example) but never in 2017, and a model meets such gaps untrained; the pipeline refuses to score a test set with clearly more missing inputs than its training set. At or above the 25 m/s cut-out the expected power is set to zero, a designed stop that one year of training data holds too few examples of for the trees to learn. Settings were fixed in advance; early stopping on a random 10 percent of the training data sets the number of rounds (the [machine learning technical note](ML_TECHNICAL_NOTE.md), section 5, shows this choice is erratic from turbine to turbine, at a small cost in accuracy).
* **Model choice.** Six candidate models, each with standard settings fixed in advance, were compared: gradient boosting, random forest, extra trees, a multilayer perceptron neural network, k-nearest neighbours and spline ridge regression. The comparison used blocked cross-validation inside 2017 only: each calendar quarter was held out in turn and the models trained on the other three, with mean absolute error as the criterion, set before any scoring. 2018 was scored once, at the end (`scripts/compare_models.py`).
* **Training.** Normal-operation intervals of 2017 (valid data, no downtime). A robust filter drops points more than four scaled median absolute deviations from the median power of their 0.5 m/s wind speed bin, so that partial faults do not become part of what the model treats as normal (0.3 percent of points).
* **Baseline and test.** Each turbine's own power curve by the method of bins, built from the same filtered training data. Both are tested on the same 2018 intervals, a year neither has seen.
* **Monitoring.** For each turbine and day of 2018, actual energy is compared with expected energy over normal-operation intervals below cut-out with expected power above 50 kW. A day needs at least six hours of such intervals and 5 MWh of expected energy, so light-wind days do not raise false alarms. Days more than 5 percent below expected are flagged. A day when at least 10 turbines were assessed and at least half of them are flagged is farm-wide, pointing to an external cause; other flags are single-turbine events.
* **Checks on the flags.** Flags are compared with the turbine status logs (site-wide messages for farm-wide days; each turbine's own messages, those logged by fewer than half the turbines that day, for single-turbine events) and with the operator's lost production to performance. The operator figure compares the same power with each turbine's default power curve, so it moves with the residual by construction: it can support a flag but not confirm it independently.

### 3.6 Solar PV performance

The solar module uses a different site from the wind work: Penmanshiel has no solar, so a public, well-documented PV system was chosen. NREL PVDAQ system 50 is a fixed array at 45 degrees tilt facing 158 degrees, in Golden, Colorado. PVAnalytics finds daylight saving clock shifts by comparing each day's production midpoint with modelled solar noon (change points by ruptures; Truong et al., 2020), and these are corrected. Expected power uses pvlib (Holmgren et al., 2018): Erbs decomposition of satellite global horizontal irradiance (Erbs et al., 1982), Perez transposition (Perez et al., 1990), a physical incidence angle modifier, SAPM cell temperature (King et al., 2004) and the PVWatts DC model (Dobos, 2014). The DC capacity is fitted on clear-sky periods. Outages (output under 5 percent of expected in daylight), suspected snow cover (cold days with a daily index under 0.6) and clipping are separated from the remaining gap.

**Independent energy estimate.** Before the measured output is used, the system's energy is estimated as a desk study would, then tested against 2012 and 2013:

* **Rated size.** NREL lists system 50 at 6.0 kW DC (Siemens M55 modules, installed March 1996). The stream analysed here is one of the system's two output channels; NREL's full records show it carrying 50.0 to 50.7 percent of the system's AC energy on every day checked (eight days across 2011 to 2013), so its rated size is taken as 3.0 kW DC. The stream matches NREL's 1-minute channel 2 data to within 0.01 percent.
* **Losses and ageing.** The generic PVWatts system losses (Dobos, 2014): soiling 2, shading 3, snow 0, mismatch 2, wiring 2, connections 0.5, light-induced degradation 1.5, nameplate rating 1 and availability 3 percent, 14.1 percent in total. Ageing of 0.5 percent a year (Jordan and Kurtz, 2013) since installation leaves 91.6 percent of rated power in 2012 and 2013.
* **Irradiance and energy.** The same pvlib chain on the NSRDB satellite weather gives the energy for the actual weather of each month. A long-term P50 follows from correcting each calendar month by the ratio of its 25-year ERA5 mean (2000 to 2024) to its mean over the record.
* **Uncertainty.** One standard deviation of energy, set before the comparison: long-term irradiance 5, transposition and PV model 3, rated size 2.5, system losses 3 and ageing 4 percent, combined by root sum of squares (8.1 percent); year-to-year irradiance variability from ERA5 adds 1.2 percent for a single year.
* **Check on the irradiance.** NREL's on-site plane-of-array sensor is compared with the satellite-based irradiance on every third day of 2012 and 2013 (days with sensor dropouts, snow or outages left out).

---

## 4. Results

### 4.1 Data quality

All Penmanshiel files match the published checksums. A clean rerun from raw data reproduces every output. 99.6 percent of SCADA intervals pass the checks; the clock is UTC all year. The checks found and handled:

| Issue | Handling |
|---|---|
| Stuck wind vane: T13 reads exactly 172.05 degrees for 17 hours on 16 and 17 August 2017, while stopped with a failed anemometer | Flagged by the frozen-reading check and excluded |
| Corrupt revenue meter day: 20 May 2017, 121.6 MWh against 93.3 MWh from the turbine meters (every other day within about 6 percent of the median ratio) | Replaced by turbine energy less the median electrical loss; left out of the electrical loss measurement |
| Controller meter counter gap: one value repeated from 21 April to 16 May 2018 | Left out of the meter scaling factor |
| Whole-farm outages: 26 September to 9 October 2020 (14 days, about 2,500 MWh not produced while ERA5 shows 7 m/s) and 9 to 10 June 2020 | Real lost production, kept in the actual figures |
| Solar: daylight saving clock shifts, end-of-interval timestamps, air temperature floored at 0 C in the weather file | Shifts corrected; stamps moved back 15 minutes; temperature effect about 0.1 percent, kept |

### 4.2 The pre-construction estimate

Over 2017 and 2018 the virtual mast correlates with ERA5 100 m wind at R squared 0.79 hourly, 0.90 daily and 0.95 monthly. Cross-validation compared the long-term methods:

| Method | Fit 2017, predict 2018 | Fit 2018, predict 2017 | Hourly error (m/s) | Long-term P50 (GWh) |
|---|---|---|---|---|
| Scaled measured distribution, sector-wise (used) | -2.1 percent | -0.6 percent | not an hourly series | 70.3 |
| Machine learning, gradient-boosted trees | -0.6 percent | -1.0 percent | 1.44 to 1.47 | 70.6 |
| Hourly linear regression, 12 sectors | +2.5 percent | +6.2 percent | 1.60 | 73.5 |
| Hourly variance ratio, 12 sectors | +4.5 percent | +8.5 percent | 1.61 to 1.65 | 75.4 |

(Energy error: mean power of one turbine from the reference power curve, without wakes, against the measured year. Hourly error: root mean square error of the synthesised hourly site wind. P50 through the same wake and loss chain.)

Hourly series from linear and variance ratio regression over-predict energy because ERA5 is smoother than the site and puts too few hours at very low wind, and a straight line per sector cannot follow the curved relationship at low wind speeds. The machine learning method learns that curve: it gives the most accurate hourly series (R squared 0.84 to 0.86 against 0.79 to 0.83) and an energy error within about 1 percent in both folds. With wind speed and direction alone it is already within 1.6 percent, so most of the gain comes from the curved fit rather than the extra inputs; across four input sets the error stays between -0.2 and -1.6 percent. A spread correction made it worse (-1.0 and -3.1 percent), so it was not used. On average the machine learning method is slightly closer than the scaled distribution (0.8 against 1.3 percent absolute error), but each wins one of the two folds, two folds cannot separate methods this close, and the long-term P50 differs by only 0.3 percent. The central case therefore stays with the scaled measured distribution, which uses the measured speeds directly, and the machine learning method is offered as an alternative in the dashboard. It would not have closed the gap with actual production. The long-term hub-height wind is **7.25 m/s** (Weibull A 8.19 m/s, k 2.17); 2017 was windy (ERA5 wind index 1.048). ERA5 alone, sheared to hub height, reads 3.7 percent low.

| Wake model | Wake loss | Net of wakes |
|---|---|---|
| Jensen Park, k = 0.075 (central) | 9.8 percent | 76.6 GWh |
| Bastankhah Gaussian, default expansion | 12.0 percent | 74.7 GWh |
| TurbOPark, site turbulence | 12.5 percent | 74.3 GWh |
| Niayifar Gaussian, site turbulence | 8.3 percent | 77.8 GWh |

| Loss | Assumed | GWh per year after |
|---|---|---|
| Gross energy | | 84.92 |
| Wake effect, internal (modelled) | 9.8 percent | 76.62 |
| Wake effect, external (neighbouring wind farms not modelled) | 0.5 percent | 76.24 |
| Turbine availability | 3.0 percent | 73.95 |
| Balance of plant availability | 0.5 percent | 73.58 |
| Grid availability | 0.5 percent | 73.22 |
| Electrical losses to the revenue meter | 2.0 percent | 71.75 |
| Turbine performance | 1.2 percent | 70.89 |
| Environmental (icing, blade degradation) | 0.8 percent | 70.32 |
| **Net energy, P50** | **17.2 percent total** | **70.32** |

Combined uncertainty before future variability is 10.0 percent (sensitivity ratio 2.00 percent energy per percent wind speed; the largest terms are measurement with nacelle anemometers, horizontal extrapolation and the power curve). Interannual variability adds 7.7 percent for a single year.

| Period | Total uncertainty | P50 | P75 | P90 | P99 |
|---|---|---|---|---|---|
| 1 year | 12.7 percent | 70.3 | 64.3 | 58.9 | 49.6 |
| 10 years | 10.3 percent | 70.3 | 65.4 | 61.0 | 53.5 |
| 20 years | 10.2 percent | 70.3 | 65.5 | 61.2 | 53.7 |

(GWh per year.) A desk study on ERA5 alone would have given 68.4 GWh.

### 4.3 Predicted against actual

| Year | Months | Actual MWh | Predicted MWh | Actual against predicted |
|---|---|---|---|---|
| 2016 (September to December) | 4 | 22,967 | 23,221 | -1.1 percent |
| 2017 | 12 | 84,126 | 75,103 | +12.0 percent |
| 2018 | 12 | 76,343 | 67,493 | +13.1 percent |
| 2019 | 12 | 76,658 | 68,349 | +12.2 percent |
| 2020 | 12 | 80,083 | 78,032 | +2.6 percent |
| 2021 (January to June) | 6 | 35,305 | 32,042 | +10.2 percent |
| **All** | **58** | **375,482** | **344,240** | **+9.1 percent** |

The first four months after commercial operation were close to prediction. September and October 2020 were 20 and 13 percent below prediction because of the 14-day whole-farm outage. The ratio is highest in light-wind spring and summer months. Long-term operational energy is **76.7 GWh per year** (regression cross-check 76.9 GWh, R squared 0.95). Against the long-term uncertainty of the estimate, that is about the **P18** level: a conservative estimate, but inside its stated uncertainty.

### 4.4 What explains the gap (2017 and 2018)

| Item | Assumed | Measured |
|---|---|---|
| Availability (turbine, balance of plant, grid) | 96.0 percent | 97.8 percent (lowest turbine T06, 95.0 percent) |
| Electrical losses | 2.0 percent | 0.98 percent |
| Power curve and turbine performance | 1.2 percent loss | 0.4 percent loss |
| Terrain speed-up | uniform flow | factors 0.913 (T07) to 1.042 (T01), +0.2 percent net |
| Wake losses | 9.8 percent | 4.5 percent (6.1 percent without the terrain adjustment) |
| Curtailment | none | none recorded |

| Step | Change, MWh per year |
|---|---|
| Pre-construction P50 | 70,324 |
| Site flow (terrain speed-up) | +113 |
| Power curve and turbine performance | +545 |
| Wakes, internal | +4,124 |
| Availability | +1,367 |
| Electrical losses | +796 |
| Curtailment | 0 |
| Residual (environmental allowance, long-term method, other) | +1,869 |
| **Actual, 2017 and 2018 on a long-term basis** | **79,139** |

Actual production in 2017 and 2018 was 12.5 percent above prediction. The measured items explain **79 percent** of that gap, with wakes the largest single item (47 percent). The residual, 2.7 percent of the P50, could come from the unmeasured environmental allowance, the long-term method (both cross-validation folds read low), the external wake allowance, or effects not captured here. Measured wake efficiency sits above every model in 10 of 12 sectors; the Park model's error (5.2 points) is about 2.7 times its assumed one-standard-deviation uncertainty.

### 4.5 Machine learning turbine health

**Model choice.** Gradient boosting had the lowest cross-validated error within 2017 on all 14 turbines (54 of 56 turbine-quarters), so it was used:

| Model | 2017 cross-validation MAE (kW) | 2018 MAE (kW) | 2018 RMSE (kW) | Daily spread (standard deviation, percent) | Single-turbine flag days, 2018 |
|---|---|---|---|---|---|
| **Gradient boosting (used)** | **24.5** | **28.6** | 50.4 | 2.7 | 7 |
| Neural network (multilayer perceptron) | 26.2 | 30.4 | 46.8 | 2.6 | 21 |
| Random forest | 27.3 | 29.4 | **46.6** | **2.3** | 13 |
| Extra trees | 29.7 | 29.8 | 47.4 | 2.5 | 31 |
| Spline ridge regression | 32.6 | 36.5 | 53.1 | 3.1 | 69 |
| Power curve (baseline) | 36.8 | 36.8 | 59.4 | 5.0 | 181 |
| K-nearest neighbours | 68.3 | 54.5 | 85.6 | 6.0 | 104 |

(MAE and RMSE are means over the 14 turbines; the flag rules are those of the monitoring step.) On 2018, gradient boosting kept the lowest mean absolute error (best on 12 of 14 turbines). Random forest and the neural network had lower root mean square error, meaning fewer large misses, and random forest a slightly tighter daily spread; averaging models could combine these strengths but would need its own cross-validation. K-nearest neighbours did worst, probably because its distances weigh direction, temperature and wind speed alike. The main flags hold across models: every leading model flags 14 January as farm-wide and T06 on 15 to 17 June, and random forest and extra trees flag 6 and all 7 of the gradient boosting single-turbine days. With the power curve as the reference, the same rules would flag 181 single-turbine days and 22 farm-wide days, which shows why the narrower band of normal behaviour matters.

**Accuracy.** On 2018, the models' mean absolute error is **28.6 kW against 36.8 kW** for each turbine's own power curve, **22 percent lower**, and root mean square error is 15 percent lower (50.4 against 59.4 kW); the model is better on both measures for all 14 turbines. R squared is 0.994 against 0.992, as wind speed alone explains most of the variance. By wind speed band, the model gains most between 8 and 16 m/s (49.6 against 75.3 kW at 8 to 12 m/s) and is weaker than the curve between 16 and 25 m/s (42.6 against 21.7 kW), where one year of training holds few storm hours, and below 3 m/s (2.35 against 0.65 kW, a trivial amount of energy). On total energy the model is slightly worse: the turbines produced 0.87 percent more than the models expected, against 0.25 percent more than the power curves expected, mostly in July and August (about +2.5 percent).

| Input | Share of permutation importance |
|---|---|
| Nacelle wind speed | 97.1 percent |
| Turbulence intensity | 1.3 percent |
| Air temperature (ERA5) | 0.9 percent |
| Wind direction (two terms) | 0.5 percent |
| Air density (ERA5) | 0.2 percent |

Wind speed carries almost all the information; the gain over a power curve comes from turbulence, temperature and direction, which a power curve ignores. A model on wind speed alone scores about the same as the power curve (37.4 against 36.8 kW).

**Why it matters for monitoring.** The spread of daily results is what decides which shortfalls can be seen. The standard deviation of daily actual-against-expected energy is **2.7 percent with the model against 5.0 percent with the power curve** (interquartile range 2.6 against 6.1 percent), so the model can flag a 5 percent shortfall that the power curve would bury in noise.

**Monitoring.** 3,620 turbine-days were assessed; the median day is 0.7 percent above expected and the 5th percentile 2.2 percent below. At the 5 percent threshold, 0.8 percent of turbine-days are flagged:

| Date | Scope | Actual against expected | What the status logs show |
|---|---|---|---|
| 14 January 2018 | 14 of 14 turbines | -7.4 percent (40 MWh) | "P output externally reduced" on all 14 turbines: an external power limit |
| 21 January 2018 | 9 of 14 | -5.9 percent (8 MWh) | No site-wide cause for the shortfall; a cold day (about 1 C). Hydraulic oil flushing was logged on 10 turbines, but those stops count as downtime and are already excluded |
| 15 to 17 June 2018 | T06 | -6.1 percent (2.6 MWh) | No messages; T06 ran about 3 percent below its normal for the whole of June, the largest monthly shortfall of any turbine |
| 4 March and 25 August 2018 | T07 | -5.9 and -5.2 percent | No messages |
| 6 September 2018 | T13 | -5.0 percent | No messages |
| 6 October 2018 | T04 | -5.4 percent | No messages |

None of the single-turbine events has a status message of its own, so the logs cannot say what caused them. On those days the operator's performance-loss figure averages 4.5 percent, against 0.3 percent on other days, but the same test (operator loss above 2 percent or a turbine message) is also met on 40 percent of unflagged turbine-days, so it supports the flags without confirming them. Storm Ali (19 September 2018) is not flagged: the shutdowns above 25 m/s are designed behaviour and left out, and the turbines ran normally below cut-out.

### 4.6 Solar PV

| Measure | Result |
|---|---|
| Actual energy, May 2011 to December 2013 (valid intervals) | 13,252 kWh |
| Expected energy (same intervals) | 12,922 kWh |
| Performance index (actual over expected) | **1.026** |
| Performance ratio (relative to the fitted 2.79 kW capacity) | 0.92 (model 0.90) |
| Daylight availability | **98.8 percent** |
| Outage loss | 0.75 percent (largest in December 2013) |
| Suspected snow loss | 0.72 percent |
| Temperature loss (inside the model) | 5.4 percent |
| Clipping | not material (0.2 percent of daylight intervals flagged) |
| Specific yield | 1,847 and 1,794 kWh per kW for 2012 and 2013 (gaps filled with expected energy times the year's index) |

The clear-sky fit improves from R squared 0.810 to 0.825 after the timestamp correction. The monthly index rises in winter (up to 1.19 in February 2012), which fits snow on the ground reflecting extra light onto the steep array; the model's fixed ground reflectance leaves this out.

**Independent estimate against actual.** The long-term P50 is **4,228 kWh a year** (1,409 kWh per kW of rated size) at the system's 2012 to 2013 age, with a 10-year P90 of 3,790 kWh. 2012 and 2013 were close to normal for irradiance (ERA5 index 1.009 and 1.003).

| Year | Estimate for the actual weather (kWh) | Actual (kWh, gaps filled) | Actual against estimate |
|---|---|---|---|
| 2012 | 4,220 | 5,151 | +22.1 percent |
| 2013 | 4,201 | 5,002 | +19.1 percent |
| **Both** | **8,420** | **10,153** | **+20.6 percent** |

The system beat the estimate by more than its uncertainty allows: the average actual sits beyond the P1 level. Replacing each generic assumption with what was measured explains the gap:

| Step (2012 and 2013) | Change (kWh) | Share of the estimate |
|---|---|---|
| Independent estimate for the same weather | 8,420 | 100 |
| Effective DC capacity and DC losses: 2.43 kW assumed (rated size after ageing and generic DC losses) against 2.79 kW measured | +1,227 | +14.6 percent |
| Availability: 3.0 percent assumed against 0.9 percent of measured outages | +208 | +2.5 percent |
| Snow: none assumed against 0.7 percent measured | -73 | -0.9 percent |
| Other (model and irradiance differences, net) | +370 | +4.4 percent |
| **Actual** | **10,153** | **+20.6 percent** |

The satellite irradiance was not the cause: on 171 usable sampled days the on-site sensor read 1.5 percent more than the satellite-based irradiance on the panels, and 0.5 percent less on clear days. Measured against that sensor, the system turned **86 percent** of its rated output into energy (performance ratio 0.86), where the generic losses and ageing assumed 71 percent. The system kept far more of its rated output after 16 years than 0.5 percent a year of ageing and the generic DC losses allow; with under three years of data, slower ageing and smaller DC losses (no shading on an open test field, for example) cannot be separated.

---

## 5. Discussion

**The estimate was conservative but not wrong.** A 9.1 percent outperformance sits inside the estimate's own uncertainty band. What makes the result useful is that each part of it can be traced. Wakes dominate: even the classic onshore Park setting over-predicted wake losses at this moderately turbulent hill site by more than its assumed uncertainty, and the default Gaussian settings were further off. Calibrating wake settings, and the uncertainty given to them, against operating wind farms with similar spacing and terrain would remove the largest error. Availability and electrical assumptions were generic and conservative; fleet data for the turbine model and a loss calculation for the actual cable layout would tighten them.

**Flow matters at turbine level even when it does not at farm level.** Free-stream wind differs by up to 14 percent between positions and tracks elevation. The farm total hardly changed, but turbine-level predictions were out by -16 to +16 percent, narrowing to about -10 to +10 percent after the measured terrain factors. Layout optimisation and turbine-level warranties need a terrain flow model.

**Data checks change results.** Left in, the corrupt meter day would have lowered the measured electrical loss and inflated one month's actual production, and the stuck vane would have fed a false direction into the virtual mast. Checking meters against each other and against the turbines is cheap and should come before any loss is measured.

**The machine learning model earns its place, within limits.** Its real value is the narrower band of normal behaviour: daily results spread about half as much as with a power curve, so a 5 percent shortfall becomes visible. Separating farm-wide from single-turbine shortfalls is the second useful step, because a site-wide limit and a turbine fault need different people. The review of the first version of this model also showed how easily such a monitor misleads: a temperature signal missing only in the test year and storm shutdowns above cut-out each created false shortfalls until they were handled. The model's remaining limits are clear. It learns 2017 as normal, so a fault present all through 2017 would be invisible. Its small summer bias means a fixed threshold is slightly more sensitive in winter than in summer. Its strongest input, nacelle wind speed, is measured behind the rotor of the turbine being judged, so a problem that disturbs both power and the nacelle anemometer could hide. And the flags it raises need checking on site: the status logs say nothing about the single-turbine events.

**Solar.** The independent estimate for the Colorado system missed by 20.6 percent, beyond its own uncertainty, while the satellite irradiance was right to within about 1.5 percent: the generic losses and ageing were the problem. As with the wind farm's availability and electrical assumptions, generic values were cautious; for solar the effect is larger because the losses compound with 16 years of assumed ageing. A real assessment would replace the generic shading and soiling values with a site study, take the snow loss from the climate, and use measured degradation for the module type. The performance review itself found the system slightly above a satellite-irradiance model fitted to its own data, with small, well-separated losses. The methods transfer directly to utility-scale plants at inverter level.

---

## 6. Conclusions

1. **Data quality.** The open data are fit for purpose once four issues are handled: a corrupt meter day, a stuck vane, a meter counter gap and a 14-day outage. All are documented and handled by repeatable code.
2. **The estimate.** A pre-construction style assessment gives a net P50 of **70.3 GWh per year** (28.0 percent capacity factor) and a 10-year P90 of 61.0 GWh. A machine learning long-term correction, tested the same way, is as accurate as the method used and gives almost the same P50 (70.6 GWh); hourly regression methods would have overstated it by 4 to 7 percent.
3. **The test.** The farm produced **9.1 percent more** than predicted over 58 months; its long-term production of 76.7 GWh per year sits at about the **P18** level, inside the estimate's uncertainty.
4. **The explanation.** For 2017 and 2018, measured losses explain **79 percent** of the gap. Over-predicted wake losses are the largest cause (**4.5 percent measured against 9.8 percent modelled**), followed by availability (97.8 against 96.0 percent), electrical losses (1.0 against 2.0 percent) and the power curve.
5. **Machine learning.** A gradient-boosted normal-behaviour model, chosen over five other machine learning models by cross-validation within the training year, predicts turbine power with **22 percent lower mean absolute error** than a power curve on an unseen year (15 percent lower root mean square error) and nearly halves the spread of daily results (2.7 against 5.0 percent). At a 5 percent threshold it flags two farm-wide days, one matched to an external power limit in the status logs, and five single-turbine events, led by T06 in June 2018; the logs do not explain the single-turbine events, so they are candidates for inspection, not diagnoses.
6. **Solar.** For the Colorado PV system, an independent estimate from the rated size, generic losses and long-term irradiance was **20.6 percent below** actual production in 2012 and 2013, beyond its uncertainty. The irradiance was right; the generic losses and ageing were too cautious (71 percent of rated output assumed, 86 percent achieved). Against a model fitted to its own clear-sky data the system produced 2.6 percent more, with 98.8 percent daylight availability and outage and snow losses under 1 percent each.

---

## 7. Limitations and future work

* The virtual mast comes from operating turbines; a real campaign would use a calibrated mast or lidar before construction.
* Nacelle anemometers are behind the rotor and are not calibrated mast instruments; measured wake losses and terrain factors depend on them and on one terrain factor per turbine.
* Turbine SCADA covers 2017 and 2018. For 2019 to 2021 the meters show the 2020 outages but nothing more is explained. A later version of the Zenodo record has data to 2024; loading it would extend both the gap analysis and the turbine health monitoring.
* Only two years overlap between the virtual mast and ERA5, so the long-term methods are compared on two folds; more years of site data would show whether the machine learning correction is reliably better.
* The machine learning model is trained on a single year and tested on the next. Averaging gradient boosting with random forest (fewer large misses on 2018) is worth testing by cross-validation. Training on rolling windows, adding a seasonal term, learning storm behaviour from more years, and comparing with a model built on neighbouring turbines' wind would make it more robust.
* Neighbouring wind farms were not modelled; a 0.5 percent allowance is in the loss chain.
* The solar system is a small research array. Its rated size is inferred as half of NREL's 6.0 kW listing from the measured channel split; if NREL's figure is not at standard test conditions, part of the estimate's gap would sit in the rated size rather than the losses. With under three years of data, ageing cannot be separated from DC losses, and the on-site irradiance check uses a sample of days.

---

## 8. Reproducibility

`python scripts/get_data.py` downloads the data and checks the checksums; `prepare_data.py`, `check_data.py`, `run_wind.py`, `run_health.py`, `run_solar.py`, `make_figures.py` and `build_dashboard.py` reproduce every table, figure and the dashboard in a few minutes. Run again from the raw data in a fresh Python environment, the wind and solar outputs are identical byte for byte, apart from the machine learning long-term check (its energy errors move by up to 0.25 percentage points and its P50 by 0.2 percent between scikit-learn 1.6 and 1.9; the central case does not use it); the turbine health results vary slightly with the scikit-learn version (1.6 against 1.9: the same days and events are flagged, headline accuracy agrees to within 0.1, and event shortfalls to within about 0.3 percentage points). `scripts/compare_models.py` reruns the model comparison (about 25 minutes on two processor cores). The unit tests (`python -m pytest`) run on synthetic data in GitHub Actions on every change. The dashboard was tested by an automated browser script that exercises every control and checks its numbers against the analysis outputs.

---

## References

* Bastankhah, M. and Porté-Agel, F. (2014). A new analytical model for wind-turbine wakes. *Renewable Energy*, 70, 116 to 123.
* Dobos, A. P. (2014). *PVWatts Version 5 Manual*. NREL/TP-6A20-62641. National Renewable Energy Laboratory.
* Erbs, D. G., Klein, S. A. and Duffie, J. A. (1982). Estimation of the diffuse radiation fraction for hourly, daily and monthly-average global radiation. *Solar Energy*, 28(4), 293 to 302.
* Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189 to 1232.
* Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146, 1999 to 2049.
* Holmgren, W. F., Hansen, C. W. and Mikofski, M. A. (2018). pvlib python: a python package for modeling solar energy systems. *Journal of Open Source Software*, 3(29), 884.
* IEC 61400-12-1:2017. Wind energy generation systems, Part 12-1: Power performance measurements of electricity producing wind turbines. International Electrotechnical Commission.
* Jensen, N. O. (1983). *A note on wind generator interaction*. Risø-M-2411. Risø National Laboratory.
* Jordan, D. C. and Kurtz, S. R. (2013). Photovoltaic degradation rates: an analytical review. *Progress in Photovoltaics*, 21(1), 12 to 29.
* Katić, I., Højstrup, J. and Jensen, N. O. (1986). A simple model for cluster efficiency. *European Wind Energy Association Conference and Exhibition*, Rome.
* King, D. L., Boyson, W. E. and Kratochvil, J. A. (2004). *Photovoltaic Array Performance Model*. SAND2004-3535. Sandia National Laboratories.
* Niayifar, A. and Porté-Agel, F. (2016). Analytical modeling of wind farms: a new approach for power prediction. *Energies*, 9(9), 741.
* Nygaard, N. G., Steen, S. T., Poulsen, L. and Pedersen, J. G. (2020). Modelling cluster wakes and wind farm blockage. *Journal of Physics: Conference Series*, 1618 (TORQUE 2020).
* Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825 to 2830.
* Perez, R., Ineichen, P., Seals, R., Michalsky, J. and Stewart, R. (1990). Modeling daylight availability and irradiance components from direct and global irradiance. *Solar Energy*, 44(5), 271 to 289.
* Perr-Sauer, J. et al. (2021). OpenOA: an open-source codebase for operational analysis of wind farms. *Journal of Open Source Software*, 6(58), 2171.
* Plumley, C. (2022). *Penmanshiel Wind Farm Data* (version 0.0.2). Zenodo, record 5946808. Cubico Sustainable Investments.
* Schlechtingen, M. and Santos, I. F. (2011). Comparative analysis of neural network and regression based condition monitoring approaches for wind turbine fault detection. *Mechanical Systems and Signal Processing*, 25(5), 1849 to 1875.
* Staffell, I. and Pfenninger, S. (2016). Using bias-corrected reanalysis to simulate current and future wind power output. *Energy*, 114, 1224 to 1239.
* Tautz-Weinert, J. and Watson, S. J. (2017). Using SCADA data for wind turbine condition monitoring: a review. *IET Renewable Power Generation*, 11(4), 382 to 394.
* Truong, C., Oudre, L. and Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299.
* Software: PyWake (DTU Wind Energy), PVAnalytics (pvlib organisation), pandas, NumPy, SciPy, Plotly.
