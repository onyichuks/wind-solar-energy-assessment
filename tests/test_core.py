"""Unit tests on small synthetic inputs. They need no downloaded data, so they run in CI."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energyassess import eya, health as hm, operational as op, resource as rs, scada as sc, solar as so, turbine as tb, wakes as wk  # noqa: E402


# ---------------------------------------------------------------- resource

def test_air_density_is_standard_at_sea_level_and_15c():
    assert rs.air_density(15.0, 1013.25, 0.0, 2.0) == pytest.approx(1.225, abs=0.002)
    # higher ground means thinner air
    assert rs.air_density(15.0, 1013.25, 0.0, 300.0) < rs.air_density(15.0, 1013.25, 0.0, 2.0)


def test_shear_extrapolation_follows_power_law():
    t = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
    era5 = pd.DataFrame({"ws10": 5.0, "ws100": 5.0 * 10 ** 0.2, "wd100": 200.0, "t2m": 10.0, "sp": 1000.0}, index=t)
    ws59 = rs.extrapolate_to_height(era5, 59.0)
    assert np.allclose(ws59, era5["ws100"] * 0.59 ** 0.2)


def test_mcp_linear_recovers_relationship_and_variance_ratio_keeps_spread():
    rng = np.random.default_rng(0)
    ref = rng.uniform(2, 15, 5000)
    wd = rng.uniform(0, 360, 5000)
    site = 0.5 + 0.9 * ref + rng.normal(0, 0.8, 5000)
    lin = rs.mcp_fit(ref, wd, site, 12, "linear")
    assert lin["slope"].mean() == pytest.approx(0.9, abs=0.03)
    vr = rs.mcp_fit(ref, wd, site, 12, "variance_ratio")
    pred = rs.mcp_predict(vr, ref, wd)
    assert pred.std() == pytest.approx(site.std(), rel=0.02)


def test_scaled_distribution_moves_the_mean_by_the_long_term_ratio():
    rng = np.random.default_rng(1)
    ref_c = rng.weibull(2.0, 8000) * 8
    wd_c = rng.uniform(0, 360, 8000)
    site = 0.95 * ref_c
    ref_lt = np.concatenate([ref_c, ref_c]) * 1.05  # long term 5 percent windier
    wd_lt = np.concatenate([wd_c, wd_c])
    out = rs.mcp_scaled_distribution(site, wd_c, ref_c, wd_c, ref_lt, wd_lt)
    assert out["mean_long_term"] / out["mean_concurrent"] == pytest.approx(1.05, rel=0.01)
    assert np.allclose(out["table"]["freq_long_term"].sum(), 1.0)


def test_ml_long_term_correction_learns_a_curved_relationship_and_quantile_map_restores_spread():
    rng = np.random.default_rng(3)
    t = pd.date_range("2017-01-01", periods=6000, freq="h", tz="UTC")
    ws100 = rng.weibull(2.0, len(t)) * 8.0
    era5 = pd.DataFrame({"ws10": 0.75 * ws100, "ws100": ws100, "wd100": rng.uniform(0, 360, len(t)),
                         "t2m": 8.0, "sp": 1000.0}, index=t)
    site = np.clip(0.05 * ws100 ** 1.6 + 0.3, 0, None) + rng.normal(0, 0.3, len(t))  # curved, not linear
    x = rs.ml_features(era5)
    assert list(x.columns) == rs.ML_FEATURES
    assert x["ws100_prev"].isna().sum() == 1 and x["ws100_next"].isna().sum() == 1
    tr, te = slice(0, 4000), slice(4000, None)
    ml = rs.mcp_ml_fit(x.iloc[tr], site[tr])
    p_ml = rs.mcp_ml_predict(ml, x.iloc[te])
    lin = rs.mcp_fit(ws100[tr], era5["wd100"].to_numpy()[tr], site[tr], 1, "linear")
    p_lin = rs.mcp_predict(lin, ws100[te], era5["wd100"].to_numpy()[te])
    rmse = lambda p: np.sqrt(np.mean((p - site[te]) ** 2))
    assert rmse(p_ml) < rmse(p_lin) and rmse(p_ml) < 1.1 * 0.3  # close to the noise added (0.3 m/s)
    # a regression squeezes the spread; mapping onto the observed distribution restores it
    squeezed = 0.6 * (site - site.mean()) + site.mean()
    mapped = rs.quantile_mapper(squeezed, site)(squeezed)
    assert np.std(squeezed) < 0.7 * np.std(site)
    assert np.std(mapped) == pytest.approx(np.std(site), rel=0.02)


def test_weighted_joint_frequency_sums_to_one_and_bins_correctly():
    wdc, wsc, f = rs.weighted_joint_frequency([5.2, 5.4, 10.1], [0.0, 358.0, 90.0], [1, 1, 2], np.arange(0, 30.5, 0.5), 5.0)
    assert f.sum() == pytest.approx(1.0)
    assert f[0, list(wsc).index(5.25)] == pytest.approx(0.5)  # 358 degrees wraps into the north bin
    assert f[18, list(wsc).index(10.25)] == pytest.approx(0.5)


def test_weibull_moments_match_known_distribution():
    rng = np.random.default_rng(2)
    v = rng.weibull(2.2, 200_000) * 8.0
    a, k = rs.weibull_moments(v)
    assert a == pytest.approx(8.0, rel=0.01) and k == pytest.approx(2.2, rel=0.03)


# ---------------------------------------------------------------- turbine and wakes

def test_density_adjustment_shifts_curve_along_speed_axis():
    pc = tb.load_power_curve()
    dense = tb.density_adjusted(pc, 1.30)
    thin = tb.density_adjusted(pc, 1.10)
    assert dense.loc[8.0] > pc.loc[8.0] > thin.loc[8.0]
    assert dense.max() == pytest.approx(pc.max())
    assert dense.loc[25.0] == pytest.approx(pc.loc[25.0])  # still at rated power at cut-out in dense air
    assert dense.loc[25.5] == 0.0 and thin.loc[25.5] == 0.0  # cut-out unchanged


def test_pywake_turbine_stops_at_cut_out():
    wt = wk.make_turbine(tb.load_power_curve(), 82.0, 59.0, 2050.0)
    assert wt.power(np.array([25.0]))[0] == pytest.approx(2050e3)
    assert wt.power(np.array([25.25]))[0] == 0.0


def test_downstream_turbine_produces_less_in_line_with_the_wind():
    pc = tb.load_power_curve()
    wt = wk.make_turbine(pc, 82.0, 59.0, 2050.0)
    model = wk.wind_farm_model("bastankhah", wt)
    p = wk.power_matrix(model, np.array([0.0, 0.0]), np.array([0.0, -500.0]), np.array([0.0, 90.0]), np.array([8.0]))
    # wind from north: turbine 1 (south) is in the wake of turbine 0
    assert p[1, 0, 0] < 0.9 * p[0, 0, 0]
    # wind from east: side by side, no wake
    assert p[1, 1, 0] == pytest.approx(p[0, 1, 0], rel=1e-3)


# ---------------------------------------------------------------- energy yield

def test_loss_chain_is_multiplicative():
    chain = eya.loss_chain(100_000.0, 0.10, {"A": 0.05, "B": 0.02})
    assert chain.attrs["net_mwh"] == pytest.approx(100_000 * 0.90 * 0.95 * 0.98)


def test_p_values_order_and_averaging():
    table = eya.combine(eya.Uncertainty(wind_speed={"ws": 2.0}, energy={"e": 3.0}, interannual_speed_pct=4.0), sens=1.8)
    pv = eya.p_values(50_000.0, table).set_index("period_years")
    assert (pv["P90_mwh"] < pv["P50_mwh"]).all()
    assert pv.loc[1, "P90_mwh"] < pv.loc[10, "P90_mwh"] < pv.loc[20, "P90_mwh"]
    assert table.attrs["historical_pct"] == pytest.approx(np.hypot(3.6, 3.0))


# ---------------------------------------------------------------- operational

def test_iec_disturbed_sector_widths():
    assert op.disturbed_sector_width(2 * 82.0) == pytest.approx(np.degrees(1.3 * np.arctan(2.5 / 2 + 0.15)) + 10)
    assert op.disturbed_sector_width(4 * 82.0) > op.disturbed_sector_width(10 * 82.0)


def test_undisturbed_table_marks_the_downwind_turbine():
    t = pd.DataFrame({"turbine": ["A", "B"], "lat": [55.9000, 55.9036], "lon": [-2.3, -2.3]})  # B about 400 m north of A
    free = op.undisturbed_table(t)
    assert not free.loc[0.0, "A"]  # wind from north: A sits behind B
    assert free.loc[0.0, "B"]
    assert free.loc[90.0, "A"] and free.loc[90.0, "B"]


def test_flat_runs_marks_whole_runs_and_gaps_break_them():
    s = pd.Series([1.0, 2, 2, 2, 2, 2, 2, 3, np.nan, 3, 3])
    assert sc.flat_runs(s, 6).tolist() == [False] + [True] * 6 + [False] * 4


def _scada_frame(n=12, **override):
    t = pd.date_range("2017-01-01", periods=n, freq="10min", tz="UTC")
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"turbine": "T01", "time": t, "wind_ms": rng.uniform(5, 9, n), "wind_dir_deg": rng.uniform(0, 360, n),
                       "power_kw": rng.uniform(300, 900, n)})
    for k, v in override.items():
        df[k] = v
    return df


def test_qc_flags_a_stuck_wind_vane():
    df = _scada_frame()
    df.loc[3:9, "wind_dir_deg"] = 172.05  # seven identical readings
    q = op.apply_qc(df)
    assert q["qc_frozen"].tolist() == [False] * 3 + [True] * 7 + [False] * 2
    assert not q.loc[3:9, "qc_valid"].any() and q.loc[:2, "qc_valid"].all()


def test_hourly_means_are_centred_on_the_hour():
    t = pd.date_range("2017-01-01 00:00", periods=18, freq="10min", tz="UTC")
    mast = pd.DataFrame({"mast_ws": np.arange(18, dtype=float), "mast_dir": 90.0}, index=t)
    h = op.to_hourly(mast, min_count=1)
    # the hour labelled 01:00 averages the intervals stamped 00:30 to 01:20 (values 3 to 8)
    assert h.loc["2017-01-01 01:00", "mast_ws"] == pytest.approx(5.5)


def test_meter_day_check_flags_only_the_corrupt_day():
    t = pd.date_range("2017-01-01", periods=144 * 5, freq="10min", tz="UTC")
    turb = pd.concat([pd.DataFrame({"turbine": name, "time": t, "energy_kwh": 200.0}) for name in ["T01", "T02"]])
    grid = pd.Series(0.99 * 400.0, index=t)
    grid[(t >= "2017-01-03") & (t < "2017-01-04")] *= 1.3
    days = op.meter_day_check(turb, pd.DataFrame({"grid_kwh": grid}), min_mwh=1.0)
    assert days.attrs["median_ratio"] == pytest.approx(0.99)
    assert days.index[days["flag"]].strftime("%Y-%m-%d").tolist() == ["2017-01-03"]


def test_circular_mean_wraps_through_north():
    assert op.circular_mean_deg(np.array([350.0, 10.0])) == pytest.approx(0.0, abs=1e-6) or \
        op.circular_mean_deg(np.array([350.0, 10.0])) == pytest.approx(360.0, abs=1e-6)


# ---------------------------------------------------------------- solar

def test_performance_ratio_definition():
    # 10 kWh from a 2 kW array under 6 kWh/m2 of plane-of-array irradiation: (10/2)/6
    assert so.performance_ratio(10.0, 6.0, 2.0) == pytest.approx(5 / 6)


def test_dc_capacity_fit_is_least_squares():
    idx = pd.date_range("2012-06-01 10:00", periods=5, freq="15min", tz="Etc/GMT+7")
    unit = pd.DataFrame({"pac_unclipped_kw": [0.5, 0.6, 0.7, 0.8, 0.9]}, index=idx)
    actual = 2.5 * unit["pac_unclipped_kw"]
    assert so.fit_dc_capacity(actual, unit, pd.Series(True, index=idx)) == pytest.approx(2.5)


def test_independent_solar_estimate_helpers():
    # PVWatts default losses combine multiplicatively to 14.08 percent
    assert so.combined_loss(so.PVWATTS_LOSSES) == pytest.approx(0.1408, abs=0.0005)
    assert so.ageing_factor(2012.5, 1996.2, 0.005) == pytest.approx(1 - 0.005 * 16.3)
    assert so.ageing_factor(1990.0, 1996.2, 0.005) == 1.0
    # long-term factors: a record that was 10 percent sunnier than the long term gives factors of 1/1.1
    months = pd.period_range("2000-01", "2009-12", freq="M")
    era5 = pd.DataFrame({"ghi_mean_wm2": 200.0}, index=months)
    record = pd.period_range("2008-01", "2009-12", freq="M")
    era5.loc[record, "ghi_mean_wm2"] = 220.0
    f = so.long_term_month_factors(era5, record)
    lt = (200.0 * 8 + 220.0 * 2) / 10
    assert len(f) == 12 and np.allclose(f, lt / 220.0)
    pv = so.p_values_kwh(1000.0, 8.0, 2.0)
    one = pv[pv["period_years"] == 1].iloc[0]
    assert one["P50_kwh"] == pytest.approx(1000.0) and one["P90_kwh"] < one["P75_kwh"] < one["P50_kwh"]
    assert one["total_uncertainty_pct"] == pytest.approx(np.sqrt(8 ** 2 + 2 ** 2))


def test_time_shift_correction_moves_values_back_one_hour():
    idx = pd.date_range("2012-06-01", periods=96, freq="15min", tz="Etc/GMT+7")
    p = pd.Series(np.arange(96, dtype=float), index=idx)
    shift = pd.Series([-60.0], index=[idx[0].normalize()])
    out = so.correct_time_shifts(p, shift)
    assert out.iloc[0] == 4.0  # the value stamped 01:00 belongs at 00:00


# ---------------------------------------------------------------- turbine health (machine learning)

def _synthetic_turbine(n=6000, seed=4):
    """Power follows a cubic curve that turbulence lowers near rated, which a plain power curve cannot see."""
    rng = np.random.default_rng(seed)
    ws = rng.uniform(3, 20, n)
    ti = rng.uniform(0.05, 0.25, n)
    base = np.clip(2050 * ((ws - 3) / 9) ** 3, 0, 2050)
    power = base * (1 - 0.6 * (ti - 0.15) * np.exp(-((ws - 11) / 2.5) ** 2)) + rng.normal(0, 20, n)
    return pd.DataFrame({"wind_ms": ws, "turbulence": ti, "dir_sin": 0.0, "dir_cos": 1.0,
                         "air_temp_c": 10.0, "air_density": 1.22, "power_kw": power})


def test_healthy_mask_drops_a_partial_fault_and_keeps_high_wind():
    df = _synthetic_turbine()
    fault = df.index[df["wind_ms"] > 8][:60]
    df.loc[fault, "power_kw"] *= 0.5  # 60 intervals at half power in good wind
    storm = pd.DataFrame({"wind_ms": [26.0, 27.5, 29.0] * 4, "power_kw": 0.0})  # shutdowns above cut-out
    keep = hm.healthy_mask(pd.concat([df, storm], ignore_index=True))
    assert not keep[fault].any() and keep[:len(df)].drop(fault).mean() > 0.99
    assert keep[len(df):].all()  # high-wind points are judged, not dropped


def test_model_beats_the_power_curve_and_flags_a_derate():
    df = _synthetic_turbine()
    train, test = df.iloc[:5000], df.iloc[5000:].copy()
    model, curve = hm.fit_model(train), hm.binned_curve(train)
    ml = hm.accuracy(test["power_kw"], hm.predict(model, test))
    pc = hm.accuracy(test["power_kw"], hm.curve_predict(curve, test["wind_ms"]))
    assert ml["mae_kw"] < pc["mae_kw"] and abs(ml["energy_bias_pct"]) < 1.5
    test["power_kw"] *= 0.9  # a 10 percent derate shows up as a negative residual
    assert hm.accuracy(test["power_kw"], hm.predict(model, test))["energy_bias_pct"] < -8


def test_predict_stops_at_cut_out_and_missing_inputs_are_refused():
    df = _synthetic_turbine()
    model = hm.fit_model(df)
    storm = df.iloc[:3].assign(wind_ms=[24.9, 25.0, 30.0])
    assert list(hm.predict(model, storm)[1:]) == [0.0, 0.0]
    gappy = df.copy()
    gappy.loc[gappy.index[:600], "air_temp_c"] = np.nan  # 10 percent missing, none in training
    with pytest.raises(ValueError):
        hm.check_missing(df, gappy)


def test_flag_days_separates_farm_wide_from_single_turbine_shortfalls():
    days = pd.to_datetime(["2018-01-01", "2018-01-02"])
    names = [f"T{i:02d}" for i in range(1, 13)]
    idx = pd.MultiIndex.from_product([names, days], names=["turbine", "day"])
    res = np.zeros(len(idx))
    res[0::2] = -8.0  # day 1: every turbine short
    res[5] = -6.0     # day 2: only T03 short
    daily = pd.DataFrame({"residual_pct": res, "actual_kwh": 9000.0, "expected_kwh": 10000.0, "operator_perf_loss_kwh": 0.0,
                          "mean_wind_ms": 8.0, "max_wind_ms": 12.0, "share_above_20ms": 0.0, "mean_temp_c": 5.0}, index=idx)
    f = hm.flag_days(daily, -5.0)
    assert set(f.loc[f["scope"] == "farm-wide"].index.get_level_values("day")) == {days[0]}
    assert f.loc[("T03", days[1]), "scope"] == "turbine"
    ev = hm.events(f)
    assert list(ev["scope"]) == ["farm-wide", "turbine"] and ev.loc[0, "turbines"] == "12 of 12" and ev.loc[1, "turbines"] == "T03"
    few = hm.flag_days(daily.xs(days[0], level="day", drop_level=False).iloc[:4], -5.0)  # 4 of 4 short, too few assessed
    assert (few["scope"] == "turbine").all()
