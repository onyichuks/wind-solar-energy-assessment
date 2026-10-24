"""Solar PV performance assessment of NREL PVDAQ system 50 (fixed tilt, Golden, Colorado).

Usage:  python run_solar.py
Writes tables to outputs/tables, a summary to outputs/solar_results.json and dashboard data
to outputs/solar_dashboard.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from energyassess import solar as so  # noqa: E402
from energyassess.outputs import to_json  # noqa: E402

RAW = ROOT / "data" / "raw" / "solar"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
LAT, LON = 39.7406, -105.1775
TILT, AZIMUTH = 45.0, 158.0  # ground truth published with the PVAnalytics example
# Independent energy estimate: rated size from NREL's PVDAQ metadata. NREL lists system 50 at 6.0 kW DC
# (Siemens M55 modules, installed March 1996). This stream (ac_power_2) is one of its two outputs and
# carries 50.0 to 50.7 percent of the system's AC energy on every day checked (eight days, 2011 to 2013,
# static_data/pvdaq_system50_channel_share.csv), so its rated size is taken as half: 3.0 kW DC.
NAMEPLATE_KW = 3.0
INSTALLED = 1996.2
DEGRADATION_PER_YEAR = 0.005  # typical crystalline silicon (Jordan and Kurtz, 2013)
# Uncertainty, one standard deviation in percent of energy (typical values for a desk study)
EST_UNCERTAINTY = {"Long-term irradiance (satellite data and long-term correction)": 5.0,
                   "Plane-of-array transposition and PV model": 3.0,
                   "Rated size (channel share and nameplate tolerance)": 2.5,
                   "System losses (soiling, shading, mismatch, wiring, availability)": 3.0,
                   "Ageing (0.5 plus or minus 0.25 percent per year over 16 years)": 4.0}


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    p_raw, w = so.load_pvdaq_system50(RAW / "system_50_ac_power_2_full_DST.parquet",
                                      RAW / "system_50_ac_power_2_full_DST_psm3.parquet")

    # 1. clock shifts and data checks
    shift = so.detect_time_shifts(p_raw, LAT, LON)
    p_dst = so.correct_time_shifts(p_raw, shift)
    # The values appear to be stamped at the end of each 15-minute interval: moving them back
    # one step gives the best clear-sky fit and removes the morning-to-afternoon tilt in the
    # residuals. The fit is computed for both alignments and reported.
    p = p_dst.shift(-1)
    changes = shift[shift.diff().fillna(shift) != 0]
    unit = so.expected_power(p.index, w["ghi"], w["temp_air"], LAT, LON, TILT, AZIMUTH, 1.0, 1e9)
    day = unit["poa_wm2"] > 50
    qc = so.qc_power(p, day)
    pd.DataFrame({"check": ["missing", "negative", "stale (daytime)", "valid"],
                  "share_pct": [100 * qc["missing"].mean(), 100 * qc["negative"].mean(),
                                100 * qc["stale"].mean(), 100 * qc["valid"].mean()]}).round(3).to_csv(
        TABLES / "s01_data_quality.csv", index=False)

    # 2. clipping check and DC capacity
    from pvanalytics.features.clipping import geometric
    clipped = geometric(p.fillna(0), freq="15min")
    clipped_share = float(clipped[day].mean())
    pac_max = 1e9 if clipped_share < 0.01 else float(p[clipped].median())  # under 1 percent: no real clipping
    clear = (w["ghi"] >= 0.98 * w["ghi_clear"]) & (w["ghi"] > 300)
    fit_mask = clear & qc["valid"] & (p > 0.1) & ~clipped
    pdc0 = so.fit_dc_capacity(p, unit, fit_mask)

    def clear_r2(series):
        m = clear & qc["valid"] & (series > 0.1)
        cap = so.fit_dc_capacity(series, unit, m)
        a, e = series[m], cap * unit.loc[m, "pac_unclipped_kw"]
        return float(1 - ((a - e) ** 2).sum() / ((a - a.mean()) ** 2).sum())
    r2_as_stamped, r2_aligned = clear_r2(p_dst), clear_r2(p)
    exp = so.expected_power(p.index, w["ghi"], w["temp_air"], LAT, LON, TILT, AZIMUTH, pdc0, pac_max)

    # 3. losses and performance
    cls = so.classify_losses(p, exp, qc["valid"], w["temp_air"])
    ok = cls["state"].isin(["normal", "outage", "suspected snow"])
    energy = (p.where(ok).fillna(0) / 4)
    expected_e = exp["pac_kw"].where(ok).fillna(0) / 4
    unclipped_e = exp["pac_unclipped_kw"].where(ok).fillna(0) / 4
    poa_kwh = exp["poa_wm2"].where(ok).fillna(0) / 4 / 1000
    monthly = pd.DataFrame({
        "actual_kwh": energy.resample("MS").sum(),
        "expected_kwh": expected_e.resample("MS").sum(),
        "expected_unclipped_kwh": unclipped_e.resample("MS").sum(),
        "poa_kwh_m2": poa_kwh.resample("MS").sum(),
        "outage_kwh": cls["lost_kwh"].where(cls["state"] == "outage", 0).resample("MS").sum(),
        "snow_kwh": cls["lost_kwh"].where(cls["state"] == "suspected snow", 0).resample("MS").sum(),
        "expected_in_gaps_kwh": (exp["pac_kw"].where(cls["state"] == "no data").fillna(0) / 4).resample("MS").sum(),
        "data_coverage": ok.groupby(ok.index.to_period("M").to_timestamp()).mean().reindex(
            energy.resample("MS").sum().index.tz_localize(None)).to_numpy() / day.resample("MS").mean().to_numpy(),
    })
    monthly["daylight_n"] = cls["state"].isin(["normal", "outage", "suspected snow"]).resample("MS").sum()
    monthly["outage_n"] = (cls["state"] == "outage").resample("MS").sum()
    monthly["clipping_kwh"] = monthly["expected_unclipped_kwh"] - monthly["expected_kwh"]
    monthly["performance_index"] = monthly["actual_kwh"] / monthly["expected_kwh"]
    monthly["performance_ratio"] = so.performance_ratio(monthly["actual_kwh"], monthly["poa_kwh_m2"], pdc0)
    monthly["expected_pr"] = so.performance_ratio(monthly["expected_kwh"], monthly["poa_kwh_m2"], pdc0)
    monthly.index = monthly.index.tz_localize(None)
    monthly = monthly.iloc[1:]  # the first month starts mid-April
    monthly.round(4).to_csv(TABLES / "s02_monthly_performance.csv")

    tot = monthly.sum(numeric_only=True)
    unclip = tot["expected_unclipped_kwh"]
    waterfall = pd.DataFrame([
        {"step": "Expected, before inverter limit", "kwh": unclip},
        {"step": "Inverter clipping (modelled)", "kwh": -tot["clipping_kwh"]},
        {"step": "Outages (no output in daylight)", "kwh": -tot["outage_kwh"]},
        {"step": "Suspected snow cover", "kwh": -tot["snow_kwh"]},
        {"step": "Other (model error and unmodelled effects, net)", "kwh": tot["actual_kwh"] - (unclip - tot["clipping_kwh"] - tot["outage_kwh"] - tot["snow_kwh"])},
        {"step": "Actual", "kwh": tot["actual_kwh"]},
    ])
    waterfall["pct_of_expected"] = 100 * waterfall["kwh"] / unclip
    waterfall.round(2).to_csv(TABLES / "s03_loss_waterfall.csv", index=False)

    # year on year: performance index for complete years
    yearly = monthly.groupby(monthly.index.year)[["actual_kwh", "expected_kwh", "poa_kwh_m2", "outage_kwh", "snow_kwh",
                                                  "expected_in_gaps_kwh"]].sum()
    yearly["months"] = monthly.groupby(monthly.index.year).size()
    yearly["performance_index"] = yearly["actual_kwh"] / yearly["expected_kwh"]
    yearly["performance_ratio"] = so.performance_ratio(yearly["actual_kwh"], yearly["poa_kwh_m2"], pdc0)
    # specific yield needs the whole year: daylight intervals with no data are filled with
    # expected energy times that year's performance index
    yearly["energy_gap_filled_kwh"] = yearly["actual_kwh"] + yearly["expected_in_gaps_kwh"] * yearly["performance_index"]
    yearly["specific_yield_kwh_per_kwp"] = yearly["energy_gap_filled_kwh"] / pdc0
    yearly.round(4).to_csv(TABLES / "s04_yearly_performance.csv")

    # availability (time-based over daylight hours with data)
    daylight = cls["state"].isin(["normal", "outage", "suspected snow"])
    availability = 1 - (cls["state"] == "outage").sum() / daylight.sum()

    # temperature effect: hot-weather loss in the expected model
    t_loss = float(1 - (exp["pdc_kw"].where(ok).sum() / (pdc0 * exp["poa_eff_wm2"].where(ok).sum() / 1000)))

    # 4. independent energy estimate (pre-construction style): rated size, generic losses and
    # long-term irradiance only, no measured output; then tested against 2012 and 2013
    era5m = pd.read_csv(ROOT / "static_data" / "era5_golden_monthly.csv")
    era5m.index = pd.PeriodIndex(era5m["month"], freq="M")
    full_years = era5m["ghi_mean_wm2"].groupby(era5m.index.year).agg(["mean", "size"])
    full_years = full_years[full_years["size"] == 12]["mean"]
    iav_pct = float(100 * full_years.std() / full_years.mean())
    unit_e = (unit["pac_unclipped_kw"].clip(lower=0).fillna(0) / 4)  # kWh per kW DC, every interval
    unit_m = unit_e.resample("MS").sum()
    unit_m.index = unit_m.index.tz_localize(None)
    unit_m = unit_m.loc[monthly.index]  # complete months of the record (May 2011 to December 2013)
    loss_all = so.combined_loss(so.PVWATTS_LOSSES)
    age = {y: so.ageing_factor(y + 0.5, INSTALLED, DEGRADATION_PER_YEAR) for y in sorted(set(monthly.index.year))}
    est_m = pd.Series([unit_m[d] * NAMEPLATE_KW * age[d.year] * (1 - loss_all) for d in unit_m.index], index=unit_m.index)
    factors = so.long_term_month_factors(era5m, monthly.index.to_period("M"))
    new_m = pd.Series([unit_m[d] * NAMEPLATE_KW * (1 - loss_all) for d in unit_m.index], index=unit_m.index)
    lt_new = float((new_m.groupby(new_m.index.month).mean() * factors).sum())  # long-term, as new
    test_years = [y for y in (2012, 2013) if y in yearly.index and yearly.loc[y, "months"] == 12]
    age_test = float(np.mean([age[y] for y in test_years]))
    p50 = lt_new * age_test
    hist_pct = float(np.sqrt(sum(v ** 2 for v in EST_UNCERTAINTY.values())))
    pvals = so.p_values_kwh(p50, hist_pct, iav_pct)
    est_rows = []
    for y in test_years:
        pred_same_weather = float(est_m[est_m.index.year == y].sum())
        actual_y = float(yearly.loc[y, "energy_gap_filled_kwh"])
        est_rows.append({"year": y, "predicted_same_weather_kwh": pred_same_weather, "actual_kwh": actual_y,
                         "actual_vs_predicted_pct": 100 * (actual_y / pred_same_weather - 1),
                         "era5_irradiance_index": float(era5m["ghi_mean_wm2"][era5m.index.year == y].mean() / full_years.mean()),
                         "ageing_factor": age[y]})
    est_years = pd.DataFrame(est_rows).set_index("year")
    est_years.round(4).to_csv(TABLES / "s06_estimate_vs_actual.csv")
    # gap waterfall over the test years: each generic assumption replaced by what was measured
    unit_test = float(sum(unit_m[unit_m.index.year == y].sum() for y in test_years))
    sel = monthly[monthly.index.year.isin(test_years)]
    outage_share = float(sel["outage_kwh"].sum() / sel["expected_unclipped_kwh"].sum())
    snow_share = float(sel["snow_kwh"].sum() / sel["expected_unclipped_kwh"].sum())
    a_gen = so.PVWATTS_LOSSES["Availability"]
    dc_generic = {k: v for k, v in so.PVWATTS_LOSSES.items() if k not in ("Availability", "Snow")}
    eff_cap_assumed = NAMEPLATE_KW * age_test * (1 - so.combined_loss(dc_generic))
    e0 = float(est_years["predicted_same_weather_kwh"].sum())
    e1 = unit_test * pdc0 * (1 - a_gen)
    e2 = unit_test * pdc0 * (1 - outage_share)
    e3 = e2 * (1 - snow_share)
    act = float(est_years["actual_kwh"].sum())
    gap_wf = pd.DataFrame([
        {"step": "Independent estimate for the same weather", "kwh": e0},
        {"step": "Effective DC capacity and DC losses (measured)", "kwh": e1 - e0},
        {"step": "Availability (measured outages)", "kwh": e2 - e1},
        {"step": "Snow (measured)", "kwh": e3 - e2},
        {"step": "Other (model and irradiance differences, net)", "kwh": act - e3},
        {"step": "Actual", "kwh": act},
    ])
    gap_wf["pct_of_estimate"] = 100 * gap_wf["kwh"] / e0
    gap_wf.round(2).to_csv(TABLES / "s07_estimate_gap_waterfall.csv", index=False)
    chain = pd.DataFrame([{"item": "Rated size (kW DC)", "value": NAMEPLATE_KW},
                          {"item": "Ageing factor at the test years", "value": age_test},
                          *[{"item": f"{k} loss (fraction)", "value": v} for k, v in so.PVWATTS_LOSSES.items()],
                          {"item": "Total system loss (fraction)", "value": loss_all},
                          {"item": "Long-term P50, as new (kWh per year)", "value": lt_new},
                          {"item": "Long-term P50 at the test years' age (kWh per year)", "value": p50}])
    chain.round(4).to_csv(TABLES / "s05_independent_estimate.csv", index=False)
    pvals.round(1).to_csv(TABLES / "s08_estimate_p_values.csv", index=False)
    # check on the satellite irradiance: NREL's on-site plane-of-array sensor, every third day of 2012
    # and 2013 (daily sums of 1-minute data; days with sensor dropouts, snow or outages left out)
    smp = pd.read_csv(ROOT / "static_data" / "pvdaq_system50_daily_sample.csv", parse_dates=["date"]).set_index("date")
    mod_poa = (unit["poa_wm2"] / 4).resample("D").sum()
    mod_poa.index = mod_poa.index.tz_localize(None)
    smp["model_poa_wh_m2"] = mod_poa.reindex(smp.index)
    ratio = smp["poa_wh_m2"] / smp["model_poa_wh_m2"]
    pr_np = (smp["ac2_wh"] / 1000) / (smp["poa_wh_m2"] / 1000 * NAMEPLATE_KW)
    good = (smp["n"] == 1440) & (smp["poa_wh_m2"] > 1000) & (smp["ac2_wh"] > 500) & ratio.between(0.6, 1.6) & (pr_np > 0.5)
    g = smp[good]
    clear_days = g["model_poa_wh_m2"] > 6000
    sensor = {"days_sampled": int(len(smp)), "days_used": int(good.sum()),
              "measured_over_satellite_poa": r(float(g["poa_wh_m2"].sum() / g["model_poa_wh_m2"].sum()), 4),
              "measured_over_satellite_poa_clear_days": r(float(g.loc[clear_days, "poa_wh_m2"].sum() / g.loc[clear_days, "model_poa_wh_m2"].sum()), 4),
              "performance_ratio_vs_rated_size": r(float(g["ac2_wh"].sum() / 1000 / (g["poa_wh_m2"].sum() / 1000 * NAMEPLATE_KW)), 4)}
    poa_test = float(sum(mod_poa[mod_poa.index.year == y].sum() for y in test_years)) / 1000  # kWh per m2, satellite
    sensor["estimate_performance_ratio_vs_rated_size"] = r(e0 / (poa_test * NAMEPLATE_KW), 4)
    # the analysed stream is NREL's channel 2: daily energy against the 1-minute files
    raw_daily = (p_raw.clip(lower=0) / 4).resample("D").sum()
    raw_daily.index = raw_daily.index.tz_localize(None)
    full = smp[(smp["n"] == 1440) & (smp["ac2_wh"] > 1000)]
    sensor["stream_over_channel2_energy"] = r(float(raw_daily.reindex(full.index).sum() / (full["ac2_wh"].sum() / 1000)), 4)
    share = pd.read_csv(ROOT / "static_data" / "pvdaq_system50_channel_share.csv")
    sensor["channel2_share_of_ac_energy_range"] = [r(share["channel2_share_of_ac_energy"].min(), 3), r(share["channel2_share_of_ac_energy"].max(), 3)]
    sensor["channel_share_days"] = int(len(share))
    mean_actual = float(est_years["actual_kwh"].mean())
    z = (mean_actual / p50 - 1) * 100 / float(pvals.loc[pvals["period_years"] == 1, "total_uncertainty_pct"].iloc[0])
    from scipy import stats as _st
    estimate = {
        "method": "Rated size from NREL metadata, PVWatts default losses, 0.5 percent per year ageing, pvlib chain on NSRDB satellite weather, long-term corrected by month to 25 years of ERA5 (2000 to 2024)",
        "rated_size_kw_dc": NAMEPLATE_KW, "installed": "March 1996", "degradation_per_year_pct": 100 * DEGRADATION_PER_YEAR,
        "pvwatts_losses_pct": {k: 100 * v for k, v in so.PVWATTS_LOSSES.items()}, "total_loss_pct": r(100 * loss_all, 2),
        "ageing_factor_test_years": r(age_test, 4),
        "long_term_factors": {int(k): r(v, 4) for k, v in factors.items()},
        "p50_new_kwh": r(lt_new, 0), "p50_kwh": r(p50, 0), "p50_specific_yield_kwh_per_kw": r(p50 / NAMEPLATE_KW, 0),
        "interannual_variability_pct": r(iav_pct, 2), "uncertainty_pct": {k: v for k, v in EST_UNCERTAINTY.items()},
        "historical_uncertainty_pct": r(hist_pct, 2),
        "p_values": pvals.round(1).to_dict(orient="records"),
        "years": est_years.reset_index().round(4).to_dict(orient="records"),
        "actual_vs_predicted_same_weather_pct": r(100 * (act / e0 - 1), 2),
        "actual_mean_vs_p50_pct": r(100 * (mean_actual / p50 - 1), 2),
        "actual_mean_exceedance_level": r(100 * (1 - _st.norm.cdf(z)), 1),
        "effective_capacity_assumed_kw": r(eff_cap_assumed, 3), "effective_capacity_measured_kw": r(pdc0, 3),
        "outage_loss_test_years_pct": r(100 * outage_share, 2), "snow_loss_test_years_pct": r(100 * snow_share, 2),
        "gap_waterfall": gap_wf.round(2).to_dict(orient="records"),
        "on_site_sensor_check": sensor,
    }

    res = {
        "system": {"source": "NREL PVDAQ system 50 (via PVAnalytics example data)", "location": "Golden, Colorado",
                   "latitude": LAT, "longitude": LON, "tilt_deg": TILT, "azimuth_deg": AZIMUTH,
                   "period": f"{monthly.index.min():%B %Y} to {monthly.index.max():%B %Y}",
                   "fitted_dc_capacity_kw": r(pdc0, 3), "peak_measured_ac_kw": r(float(p.max()), 3),
                   "clipped_daytime_intervals_pct": r(100 * clipped_share, 3)},
        "data_quality": {"valid_pct": r(100 * qc["valid"].mean(), 2),
                         "clear_sky_r2_as_stamped": r(r2_as_stamped, 3), "clear_sky_r2_aligned": r(r2_aligned, 3),
                         "clock_shift_changes": [{"date": f"{d:%Y-%m-%d}", "shift_min": int(v)} for d, v in changes.items()]},
        "totals": {"actual_kwh": r(tot["actual_kwh"], 0), "expected_kwh": r(tot["expected_kwh"], 0),
                   "performance_index": r(tot["actual_kwh"] / tot["expected_kwh"], 4),
                   "performance_ratio": r(so.performance_ratio(tot["actual_kwh"], tot["poa_kwh_m2"], pdc0), 4),
                   "expected_performance_ratio": r(so.performance_ratio(tot["expected_kwh"], tot["poa_kwh_m2"], pdc0), 4),
                   "time_availability_pct": r(100 * availability, 2),
                   "clipping_loss_pct": r(100 * tot["clipping_kwh"] / unclip, 2),
                   "outage_loss_pct": r(100 * tot["outage_kwh"] / unclip, 2),
                   "snow_loss_pct": r(100 * tot["snow_kwh"] / unclip, 2),
                   "temperature_loss_pct": r(100 * t_loss, 2)},
        "yearly": yearly.reset_index().rename(columns={"index": "year"}).round(4).to_dict(orient="records"),
        "waterfall": waterfall.round(2).to_dict(orient="records"),
        "independent_estimate": estimate,
    }
    (OUT / "solar_results.json").write_text(to_json(res, indent=2))

    # dashboard: monthly series, a clear and a snowy example day, scatter of actual against expected
    hourly = pd.DataFrame({"actual": p, "expected": exp["pac_kw"]}).resample("1h").mean()
    sample = hourly[(hourly["expected"] > 0.05)].dropna().sample(2500, random_state=2)
    snow_days = cls.index[cls["state"] == "suspected snow"].normalize().unique()
    clear_day = (clear & qc["valid"]).resample("D").sum().idxmax()
    daily = pd.DataFrame({"a": p.where(ok).resample("D").sum(), "e": exp["pac_kw"].where(ok).resample("D").sum(),
                          "clear": (clear & qc["valid"]).resample("D").sum(), "out": (cls["state"] == "outage").resample("D").sum(),
                          "sd": (p - exp["pac_kw"]).where(ok).abs().resample("D").mean()})
    daily["pi"] = daily["a"] / daily["e"]
    cloudy_day = daily[(daily["clear"] == 0) & (daily["e"] > daily["e"].median()) & daily["pi"].between(0.9, 1.1)]["sd"].idxmax()
    outage_day = daily["out"].idxmax()

    def day_profile(d):
        sl = slice(d, d + pd.Timedelta(hours=23, minutes=45))
        return {"time": [f"{t:%H:%M}" for t in p.loc[sl].index], "actual": p.loc[sl].round(3).fillna(0).tolist(),
                "expected": exp["pac_kw"].loc[sl].round(3).tolist(), "date": f"{d:%d %B %Y}"}

    dash = {"results": res,
            "monthly": {"month": [f"{d:%Y-%m}" for d in monthly.index], **{k: monthly[k].round(4).tolist() for k in
                        ["actual_kwh", "expected_kwh", "expected_unclipped_kwh", "poa_kwh_m2", "performance_index", "performance_ratio",
                         "outage_kwh", "snow_kwh", "clipping_kwh", "daylight_n", "outage_n"]}},
            "scatter": {"expected": sample["expected"].round(3).tolist(), "actual": sample["actual"].round(3).tolist()},
            "days": {"clear": day_profile(clear_day), "cloudy": day_profile(cloudy_day),
                     "snow": day_profile(snow_days[len(snow_days) // 2]) if len(snow_days) else None,
                     "outage": day_profile(outage_day)},
            "waterfall": waterfall.round(2).to_dict(orient="list")}
    (OUT / "solar_dashboard.json").write_text(to_json(dash, compact=True))
    print(json.dumps(res["system"], indent=1)); print(json.dumps(res["totals"], indent=1))
    print(waterfall.round(1).to_string()); print(yearly.round(3).to_string())
    print(json.dumps({k: v for k, v in estimate.items() if k not in ("gap_waterfall", "years", "p_values")}, indent=1))
    print(est_years.round(3).to_string()); print(gap_wf.round(1).to_string()); print(pvals.round(1).to_string())
    print(monthly[["actual_kwh", "expected_kwh", "performance_index", "performance_ratio", "outage_kwh", "snow_kwh", "data_coverage"]].round(3).to_string())


if __name__ == "__main__":
    main()
