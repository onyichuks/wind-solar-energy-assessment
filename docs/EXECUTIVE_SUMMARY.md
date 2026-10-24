# Executive summary

**Penmanshiel wind farm: a pre-construction energy yield assessment tested against operation, with machine learning turbine health monitoring. Plus a solar PV performance review of a separate system in Colorado**

Onyi Chuks, Calgary, Alberta. October 2026. [Interactive dashboard](https://onyichuks.github.io/wind-solar-energy-assessment/) · [Full report](REPORT.md) · [Machine learning note](ML_TECHNICAL_NOTE.md) · [Code](https://github.com/onyichuks/wind-solar-energy-assessment)

## The question

Lenders size wind farm debt on pre-construction energy estimates, yet those estimates are rarely checked against what the farm later produces. Penmanshiel (14 Senvion MM82 turbines, 28.7 MW, Scottish Borders) is one of very few wind farms with a complete public operating record. This project builds the estimate a developer would have made before construction, then measures every major assumption against 58 months of metered production and two years of turbine SCADA.

## What was found

* **The estimate was conservative, not wrong.** The assessment gives a net P50 of **70.3 GWh per year** (10-year P90 61.0 GWh). The farm produced **9.1 percent more** than predicted for the same weather. Its long-term production of 76.7 GWh per year sits at about the P18 level, inside the estimate's own uncertainty. A machine learning long-term correction, tested the same way, gave almost the same P50.
* **Wake losses were the largest error.** Measured from SCADA, wakes cost about **4.5 percent of energy against 9.8 percent modelled**. That one item explains 47 percent of the gap in 2017 and 2018.
* **Generic loss assumptions were cautious.** Availability was 97.8 percent against 96.0 percent assumed, electrical losses 1.0 percent against 2.0 percent, and turbine performance losses 0.4 percent against 1.2 percent assumed. With wakes, these explain **79 percent** of the gap.
* **Flow matters at turbine level.** Turbine positions see free-stream winds up to 14 percent apart, and turbine-level predictions were out by up to 16 percent either way.
* **Data checks changed results.** A corrupt day of revenue meter data, a stuck wind vane, a meter counter gap and a 14-day outage of the whole farm in 2020 were found and handled before any loss was measured.
* **Machine learning sharpens monitoring.** A normal-behaviour model of each turbine, trained on 2017 and tested on 2018, has **22 percent lower error** than each turbine's own power curve and nearly halves the day-to-day noise (2.7 against 5.0 percent), so a 5 percent shortfall stands out. It separates farm-wide shortfalls, such as an external output limit logged on all 14 turbines on 14 January 2018, from single-turbine events, led by turbine T06 in June 2018.
* **Gradient boosting beat five other machine learning models**, including random forest and a neural network, in cross-validation within 2017 on all 14 turbines, and kept the lowest average error on 2018. The main flags hold across models.
* **Solar: generic losses, not irradiance, were the error.** For a small NREL system in Colorado (a separate site), an independent estimate from rated size, generic losses and long-term irradiance was **20.6 percent below** actual in 2012 and 2013, beyond its uncertainty. The irradiance was within 1.5 percent of an on-site sensor.

## What it means

1. **Calibrate wake models on operating wind farms.** Even the classic onshore wake setting over-predicted losses at this turbulent hill site by about 2.7 times its assumed uncertainty. Calibration against similar operating farms would remove the largest single error.
2. **Replace generic loss assumptions with evidence.** Fleet availability data for the turbine model, an electrical loss calculation for the cable layout, and for solar a shading study and measured degradation for the module type would tighten the estimates.
3. **Use a flow model for layout and turbine warranties.** Terrain barely moves the farm total but matters for each turbine.
4. **Check the data before measuring losses.** Comparing meters with each other and with the summed turbine meters is cheap and catches errors that would otherwise bias loss figures.
5. **Monitor with normal-behaviour models, and route alerts by scope.** Farm-wide shortfalls go to grid and controls teams; single-turbine shortfalls go to maintenance. Flags are candidates for inspection, not diagnoses.

## Limits

* The virtual met mast is built from nacelle anemometers on operating turbines; a real campaign would use a calibrated mast or lidar.
* Turbine SCADA covers 2017 and 2018 only, so later years are compared at farm level.
* The machine learning flags are checked against the operator's records, which support but cannot confirm them.

Every number above is produced by the code in the repository and can be reproduced from the open data. Data: Cubico Sustainable Investments (Penmanshiel, CC BY 4.0), Copernicus Climate Change Service (ERA5) and NREL (PVDAQ).
