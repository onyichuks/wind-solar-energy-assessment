"""Data quality checks on every dataset the analysis uses.

Writes outputs/tables/q01 to q04 and a summary in outputs/data_quality.json, and prints the
main findings. docs/DATASET.md explains what was found and how the analysis handles it.

Usage:  python scripts/check_data.py      (after scripts/prepare_data.py)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energyassess import operational as op, resource as rs, scada as sc, solar as so  # noqa: E402
from energyassess.outputs import to_json  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
RATED_KW = 2050.0
CAPACITY_KW = 14 * RATED_KW


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def circ_mean(a: pd.DataFrame) -> pd.Series:
    rad = np.radians(a)
    return np.degrees(np.arctan2(np.sin(rad).mean(axis=1), np.cos(rad).mean(axis=1))) % 360


def scada_checks(s: pd.DataFrame, era5: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    q = op.apply_qc(s)
    g = q.groupby("turbine")
    steps = q.groupby("turbine")["time"].diff().dropna()
    tbl = pd.DataFrame({
        "intervals": g.size(),
        "duplicate_timestamps": g["time"].apply(lambda t: int(t.duplicated().sum())),
        "missing_power_or_wind_pct": 100 * g["qc_missing"].mean(),
        "out_of_range": g["qc_range"].sum(),
        "frozen_wind_speed": g["wind_ms"].apply(lambda v: int(sc.flat_runs(v, 6).sum())),
        "frozen_wind_direction": g["wind_dir_deg"].apply(lambda v: int(sc.flat_runs(v, 6).sum())),
        "anemometer_error_codes": g.apply(lambda d: int(((d["wind1_ms"] < 0) | (d["wind2_ms"] < 0)).sum()), include_groups=False),
        "valid_pct": 100 * g["qc_valid"].mean(),
    })
    # energy counter against average power: totals agree, single intervals can spill over
    ok = q["energy_kwh"].notna() & q["power_kw"].notna()
    tbl["energy_over_power_ratio"] = (q[ok].groupby("turbine")["energy_kwh"].sum()
                                      / (q[ok].groupby("turbine")["power_kw"].sum() / 6))
    tbl["intervals_above_rated_energy"] = q.assign(a=q["energy_kwh"] > RATED_KW / 6 * 1.01).groupby("turbine")["a"].sum()
    # direction calibration: offset of each turbine from the farm mean while all are producing
    d = q.pivot(index="time", columns="turbine", values="wind_dir_deg")
    p = q.pivot(index="time", columns="turbine", values="power_kw")
    d = d.where(p > 200).dropna()
    off = (d.sub(circ_mean(d), axis=0) + 180) % 360 - 180
    monthly_off = off.groupby(off.index.tz_convert(None).to_period("M")).median()
    tbl["direction_offset_median_deg"] = off.median()
    tbl["direction_offset_monthly_range_deg"] = monthly_off.max() - monthly_off.min()
    # anemometer drift: ratio of the two nacelle anemometers by quarter
    a = q[(q["wind1_ms"] > 4) & (q["wind2_ms"] > 4)]
    sensor = (a["wind1_ms"] / a["wind2_ms"]).groupby([a["turbine"], a["time"].dt.tz_convert(None).dt.to_period("Q")]).median().unstack()
    tbl["sensor_ratio_2017"] = sensor.loc[:, sensor.columns.year == 2017].mean(axis=1)
    tbl["sensor_ratio_2018"] = sensor.loc[:, sensor.columns.year == 2018].mean(axis=1)

    # timestamps: regular grid, no daylight saving duplicates, alignment with ERA5
    farm_ws = q.pivot(index="time", columns="turbine", values="wind_ms").mean(axis=1)
    ref = era5["ws100"].resample("10min").interpolate().reindex(farm_ws.index)
    bst = farm_ws.index.tz_convert("Europe/London").strftime("%Z") == "BST"

    def best_lag(mask):
        corr = {}
        for lag in range(-12, 13):
            x, y = farm_ws[mask], ref.shift(lag)[mask]
            m = x.notna() & y.notna()
            corr[-10 * lag] = np.corrcoef(x[m], y[m])[0, 1]
        return int(pd.Series(corr).idxmax())

    timing = {"step_counts": {str(k): int(v) for k, v in steps.value_counts().items()},
              "rows_per_day": sorted(set(q.groupby(q["time"].dt.date).size().tolist())),
              "era5_best_lag_minutes_summer_time": best_lag(bst),
              "era5_best_lag_minutes_winter_time": best_lag(~bst),
              "note": "ERA5 matches SCADA best when ERA5 is taken this many minutes after the SCADA stamp; the same "
                      "lag in summer and winter means the SCADA clock is UTC all year"}
    return tbl, timing


def meter_checks(s: pd.DataFrame, plant: pd.DataFrame, era5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    month = plant.index.tz_convert(None).to_period("M")
    both = plant["grid_kwh"].notna() & plant["pmu_kwh"].notna()
    spread = sc.flat_runs(plant["pmu_kwh"], 6) & (plant["pmu_kwh"] > 1.0)
    e = s.pivot(index="time", columns="turbine", values="energy_kwh")
    turb = e[e.notna().all(axis=1)].sum(axis=1)
    j = pd.DataFrame({"turb": turb, "grid": plant["grid_kwh"].reindex(turb.index)}).dropna()
    jm = j.groupby(j.index.tz_convert(None).to_period("M")).sum()
    tbl = pd.DataFrame({
        "intervals": plant.groupby(month).size(),
        "grid_meter_intervals": plant["grid_kwh"].notna().groupby(month).sum(),
        "pmu_intervals": plant["pmu_kwh"].notna().groupby(month).sum(),
        "export_coverage_pct": 100 * plant["export_kwh"].notna().groupby(month).mean(),
        "export_mwh": plant["export_kwh"].groupby(month).sum() / 1000,
        "grid_over_pmu": plant.loc[both, "grid_kwh"].groupby(month[both]).sum() / plant.loc[both, "pmu_kwh"].groupby(month[both]).sum(),
        "pmu_spread_intervals": spread.groupby(month).sum(),
        "turbine_to_meter_loss_pct": 100 * (1 - jm["grid"] / jm["turb"]),
    })
    tbl.index.name = "month"

    events = []
    days = op.meter_day_check(s, plant)
    for d, row in days[days["flag"]].iterrows():
        events.append({"start": f"{d:%Y-%m-%d}", "end": f"{d:%Y-%m-%d}", "meter": "grid meter",
                       "event": "daily energy does not match the turbine meters",
                       "detail": f"meter {row['meter_mwh']:.1f} MWh against turbines {row['turbine_mwh']:.1f} MWh",
                       "handling": "day replaced by turbine energy less the median electrical loss"})
    run_id = (spread != spread.shift()).cumsum()
    for _, block in plant[spread].groupby(run_id[spread]):
        if len(block) >= 144:
            events.append({"start": f"{block.index[0]:%Y-%m-%d}", "end": f"{block.index[-1]:%Y-%m-%d}", "meter": "PMU",
                           "event": "one energy value repeated (counter gap spread evenly)",
                           "detail": f"{len(block)} intervals at {block['pmu_kwh'].iloc[0]:.1f} kWh",
                           "handling": "left out of the PMU scaling factor; grid meter used for these dates"})
    # whole-farm outages: no export for a day or more while ERA5 shows wind
    daily = pd.DataFrame({"mwh": plant["export_kwh"].resample("D").sum() / 1000,
                          "ws100": era5["ws100"].resample("D").mean()}).dropna()
    daily = daily.loc["2016-09-01":]
    zero = daily["mwh"] < 0.5
    zgrp = (zero != zero.shift()).cumsum()
    windy = daily["ws100"].groupby(zgrp).transform("max") > 5.0  # calm spells also give zero export
    out_day = zero & windy
    normal = daily[~zero]
    bins = np.arange(0, 31, 1.0)
    typical = normal.groupby(pd.cut(normal["ws100"], bins), observed=True)["mwh"].median()
    expected = typical.reindex(pd.cut(daily["ws100"], bins)).to_numpy()
    daily["expected_mwh"] = expected
    grp = (out_day != out_day.shift()).cumsum()
    for _, block in daily[out_day].groupby(grp[out_day]):
        span = plant.loc[block.index[0]:block.index[-1] + pd.Timedelta("1D") - pd.Timedelta("10min")]
        pw = span["plant_power_kw"].abs().max()
        power_note = f", PMU plant power never above {pw:.0f} kW" if np.isfinite(pw) else ""
        events.append({"start": f"{block.index[0]:%Y-%m-%d}", "end": f"{block.index[-1]:%Y-%m-%d}",
                       "meter": "grid meter" if block.index[0] < pd.Timestamp("2019-10-01", tz="UTC") else "PMU",
                       "event": "whole wind farm not exporting",
                       "detail": f"{len(block)} days, ERA5 mean wind {block['ws100'].mean():.1f} m/s{power_note}, "
                                 f"roughly {block['expected_mwh'].sum():.0f} MWh not produced",
                       "handling": "real outage: kept in the actual production"})
    ev = pd.DataFrame(events)
    half_hourly = plant["grid_kwh"].groupby(plant.index.floor("30min")).transform(lambda v: len(v) == 3 and v.nunique() == 1)
    summary = {"grid_meter_period": [f"{plant['grid_kwh'].first_valid_index():%Y-%m-%d}", f"{plant['grid_kwh'].last_valid_index():%Y-%m-%d}"],
               "pmu_period": [f"{plant['pmu_kwh'].first_valid_index():%Y-%m-%d}", f"{plant['pmu_kwh'].last_valid_index():%Y-%m-%d}"],
               "grid_meter_half_hourly_share_pct": r(100 * half_hourly[plant["grid_kwh"].notna() & (plant["grid_kwh"] != 0)].mean(), 1),
               "grid_meter_negative_intervals": int((plant["grid_kwh"] < 0).sum()),
               "export_missing_intervals": int(plant["export_kwh"].isna().sum()),
               "pmu_to_grid_ratio": r(plant.attrs.get("pmu_to_grid_ratio", np.nan), 4),
               "daily_meter_to_turbine_median_ratio": r(days.attrs["median_ratio"], 4),
               "outage_days_after_sept_2016": int(out_day.sum())}
    return tbl, ev, summary


def era5_checks(era5: pd.DataFrame) -> dict:
    full = pd.date_range(era5.index.min(), era5.index.max(), freq="h")
    yearly = era5["ws100"].groupby(era5.index.year).mean()
    from scipy import stats
    trend = stats.linregress(yearly.index, yearly.to_numpy())
    return {"hours": len(era5), "expected_hours": len(full), "duplicates": int(era5.index.duplicated().sum()),
            "missing_values": int(era5.isna().sum().sum()),
            "ranges": {c: [r(era5[c].min(), 1), r(era5[c].max(), 1)] for c in era5.columns},
            "annual_mean_ws100_range": [r(yearly.min(), 2), r(yearly.max(), 2)],
            "trend_ms_per_decade": r(10 * trend.slope, 3), "trend_p_value": r(trend.pvalue, 2)}


def solar_checks() -> dict:
    p, w = so.load_pvdaq_system50(RAW / "solar" / "system_50_ac_power_2_full_DST.parquet",
                                  RAW / "solar" / "system_50_ac_power_2_full_DST_psm3.parquet")
    raw_t = pd.read_parquet(RAW / "solar" / "system_50_ac_power_2_full_DST_psm3.parquet")["temp_air"]
    lat, lon, tilt, az = 39.7406, -105.1775, 45.0, 158.0
    base = so.expected_power(p.index, w["ghi"], w["temp_air"], lat, lon, tilt, az, 1.0, 1e9)
    colder = so.expected_power(p.index, w["ghi"], w["temp_air"].where(w["temp_air"] > 0, -5.0), lat, lon, tilt, az, 1.0, 1e9)
    day = base["poa_wm2"] > 50
    return {"power_steps": {str(k): int(v) for k, v in p.index.to_series().diff().value_counts().items()},
            "power_duplicates": int(p.index.duplicated().sum()),
            "power_missing_pct": r(100 * p.isna().mean(), 2),
            "power_negative": int((p < 0).sum()),
            "power_max_kw": r(p.max(), 3),
            "weather_missing_pct": r(100 * w["ghi"].isna().mean(), 2),
            "air_temperature_min_c": r(raw_t.min(), 1),
            "air_temperature_exact_zero_pct": r(100 * (raw_t == 0).mean(), 1),
            "daylight_hours_at_zero_c_pct": r(100 * (w["temp_air"][day] == 0).mean(), 1),
            "expected_energy_change_if_zero_readings_were_minus5c_pct": r(100 * (colder["pac_kw"][day].sum() / base["pac_kw"][day].sum() - 1), 2)}


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    TABLES.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(PROC / "turbine_scada_10min.parquet")
    plant = sc.load_plant_meter(RAW / "Penmanshiel_Grid_3153.zip", RAW / "Penmanshiel_PMU_3152.zip")
    era5 = rs.load_era5_openmeteo(RAW / "era5_penmanshiel_openmeteo.csv")

    t1, timing = scada_checks(s, era5)
    t1.round(4).to_csv(TABLES / "q01_scada_checks_by_turbine.csv")
    t2, events, meters = meter_checks(s, plant, era5)
    t2.round(4).to_csv(TABLES / "q02_meter_checks_by_month.csv")
    events.to_csv(TABLES / "q03_meter_events.csv", index=False)
    summary = {"scada": {"turbines": int(t1.shape[0]), "intervals": int(t1["intervals"].sum()),
                         "valid_pct": r(t1["valid_pct"].mean(), 2), "frozen_wind_direction": int(t1["frozen_wind_direction"].sum()),
                         "frozen_wind_speed": int(t1["frozen_wind_speed"].sum()),
                         "anemometer_error_codes": int(t1["anemometer_error_codes"].sum()),
                         "energy_over_power_ratio_range": [r(t1["energy_over_power_ratio"].min(), 4), r(t1["energy_over_power_ratio"].max(), 4)],
                         "direction_offset_range_deg": [r(t1["direction_offset_median_deg"].min(), 1), r(t1["direction_offset_median_deg"].max(), 1)],
                         "timing": timing},
               "meters": meters, "meter_events": events.to_dict(orient="records"),
               "era5": era5_checks(era5), "solar": solar_checks()}
    (OUT / "data_quality.json").write_text(to_json(summary, indent=2))
    pd.set_option("display.width", 200)
    print(t1.round(3).to_string())
    print(events.to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "meter_events"}, indent=1, default=str))


if __name__ == "__main__":
    main()
