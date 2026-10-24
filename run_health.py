"""Penmanshiel turbine health: a machine learning normal-behaviour model for each turbine.

Trains on 2017 SCADA, tests on 2018 (data the model has never seen), compares the model
with each turbine's own power curve, and uses the daily gap between actual and expected
energy to flag underperformance. Flags are checked against the operator's performance
loss records and the turbine status logs.

Usage:  python run_health.py      (after scripts/prepare_data.py)
Writes outputs/tables/h01 to h05, outputs/health_results.json and outputs/health_dashboard.json.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from energyassess import health as hm, operational as op, resource as rs  # noqa: E402
from energyassess.outputs import to_json  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
TRAIN_YEAR, TEST_YEAR = 2017, 2018
HUB = 59.0
THRESHOLD_PCT = -5.0


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    TABLES.mkdir(parents=True, exist_ok=True)
    turbines = pd.read_csv(ROOT / "static_data" / "penmanshiel_turbines.csv")
    era5 = rs.load_era5_openmeteo(DATA / "raw" / "era5_penmanshiel_openmeteo.csv")
    rho = pd.Series(rs.air_density(era5["t2m"], era5["sp"], era5.attrs["elevation_m"], turbines["elevation_m"].mean() + HUB),
                    index=era5.index)
    weather = pd.DataFrame({"air_temp_c": era5["t2m"], "air_density": rho})
    scada = hm.add_features(op.apply_qc(pd.read_parquet(DATA / "processed" / "turbine_scada_10min.parquet")), weather)
    status = pd.read_parquet(DATA / "processed" / "turbine_status.parquet")
    data = scada[hm.normal_operation(scada)]

    acc_rows, imp_rows, preds, scatter = [], [], [], {}
    from sklearn.inspection import permutation_importance
    for name, g in data.groupby("turbine"):
        train_all = g[g["time"].dt.year == TRAIN_YEAR]
        test = g[g["time"].dt.year == TEST_YEAR].copy()
        hm.check_missing(train_all, test)
        healthy = hm.healthy_mask(train_all)
        train = train_all[healthy]
        model = hm.fit_model(train)
        curve = hm.binned_curve(train)
        test["expected_kw"] = hm.predict(model, test)
        test["curve_kw"] = hm.curve_predict(curve, test["wind_ms"])
        ml, pc = hm.accuracy(test["power_kw"], test["expected_kw"]), hm.accuracy(test["power_kw"], test["curve_kw"])
        acc_rows.append({"turbine": name, "train_intervals": len(train), "train_dropped_as_unhealthy": int((~healthy).sum()),
                         "test_intervals": len(test), "boosting_iterations": int(model.n_iter_),
                         **{f"{k}_model": v for k, v in ml.items()}, **{f"{k}_power_curve": v for k, v in pc.items()}})
        sample = test.sample(min(4000, len(test)), random_state=1)
        pi = permutation_importance(model, sample[hm.FEATURES], sample["power_kw"], scoring="neg_mean_absolute_error",
                                    n_repeats=3, random_state=0)
        imp_rows.append(pd.Series(pi.importances_mean, index=hm.FEATURES, name=name))
        sc = test.sample(min(300, len(test)), random_state=2)
        scatter[name] = {"wind": sc["wind_ms"].round(2).tolist(), "actual": sc["power_kw"].round(0).tolist(),
                         "expected": sc["expected_kw"].round(0).tolist()}
        preds.append(test[["turbine", "time", "power_kw", "expected_kw", "curve_kw", "lost_performance_kwh", "wind_ms", "air_temp_c"]])

    acc = pd.DataFrame(acc_rows).set_index("turbine")
    acc.round(4).to_csv(TABLES / "h01_model_accuracy_2018.csv")
    imp = pd.DataFrame(imp_rows)
    imp_share = (imp.clip(lower=0).div(imp.clip(lower=0).sum(axis=1), axis=0)).mean().sort_values(ascending=False)
    pd.DataFrame({"feature": [hm.FEATURE_LABELS[f] for f in imp_share.index], "share_of_importance": imp_share.values,
                  "mae_increase_kw": imp.mean()[imp_share.index].values}).round(4).to_csv(TABLES / "h02_feature_importance.csv", index=False)

    pred = pd.concat(preds, ignore_index=True)
    # error by wind speed band, model against power curve
    bands = pd.cut(pred["wind_ms"], [0, 3, 8, 12, 16, 25, 40], right=False)
    by_band = pred.groupby(bands, observed=True).apply(lambda d: pd.Series({
        "intervals": len(d), "mae_kw_model": (d["power_kw"] - d["expected_kw"]).abs().mean(),
        "mae_kw_power_curve": (d["power_kw"] - d["curve_kw"]).abs().mean()}), include_groups=False)
    by_band.index = [f"{iv.left:g} to {iv.right:g} m/s" for iv in by_band.index]
    by_band.round(2).to_csv(TABLES / "h06_error_by_wind_speed_2018.csv")
    daily = hm.daily_residuals(pred)
    daily_curve = hm.daily_residuals(pred, expected_col="curve_kw")
    common = daily.index.intersection(daily_curve.index)
    spread = {"daily_residual_std_pct_model": r(daily.loc[common, "residual_pct"].std(), 2),
              "daily_residual_std_pct_power_curve": r(daily_curve.loc[common, "residual_pct"].std(), 2),
              "daily_residual_iqr_pct_model": r(float(np.subtract(*daily.loc[common, "residual_pct"].quantile([0.75, 0.25]))), 2),
              "daily_residual_iqr_pct_power_curve": r(float(np.subtract(*daily_curve.loc[common, "residual_pct"].quantile([0.75, 0.25]))), 2)}
    log = hm.status_log_by_day(status)
    # each turbine's own messages (a message logged by half the turbines or more that day is site-wide)
    daily["status_messages"] = hm.messages_per_turbine_day(log, n_turbines=len(turbines)).reindex(daily.index).to_numpy()
    flagged = hm.flag_days(daily, THRESHOLD_PCT)
    flagged.round(4).to_csv(TABLES / "h03_daily_residuals_2018.csv")
    ev = hm.events(flagged)
    num = ["residual_pct", "shortfall_mwh", "operator_perf_loss_pct", "mean_wind_ms", "max_wind_ms", "share_above_20ms", "mean_temp_c"]
    ev[num] = ev[num].astype(float)
    # what the turbine status logs say on the flagged days
    ev["status_messages"] = [
        "; ".join(sorted({m for m in flagged.loc[(t, slice(s, e)), "status_messages"].dropna()})) if sc == "turbine"
        else hm.messages_for_day(log, s) for sc, t, s, e in zip(ev["scope"], ev["turbines"], ev["start"], ev["end"])]
    ev.round(3).to_csv(TABLES / "h04_underperformance_events_2018.csv", index=False)
    monthly = flagged.assign(month=flagged.index.get_level_values("day").to_period("M")).groupby(["turbine", "month"])[["actual_kwh", "expected_kwh"]].sum()
    mres = (100 * (monthly["actual_kwh"] / monthly["expected_kwh"] - 1)).unstack()
    mres.round(2).to_csv(TABLES / "h05_monthly_residual_pct_2018.csv")

    # how the flags compare with the operator's own performance-loss figure (from the default power curve)
    t_days = flagged[flagged["scope"] == "turbine"]
    others = flagged[~flagged["flag"]]
    op_pct = lambda d: float(100 * d["operator_perf_loss_kwh"].sum() / d["actual_kwh"].sum()) if len(d) else float("nan")
    farm = ev[ev["scope"] == "farm-wide"]
    tev = ev[ev["scope"] == "turbine"]
    results = {
        "setup": {"train_year": TRAIN_YEAR, "test_year": TEST_YEAR, "model": "Gradient-boosted trees (scikit-learn HistGradientBoostingRegressor), one per turbine",
                  "features": [hm.FEATURE_LABELS[f] for f in hm.FEATURES], "baseline": "Each turbine's own power curve, method of bins, same training data",
                  "cut_out_rule": "expected power is zero at or above 25 m/s, and those intervals are left out of the daily residuals",
                  "threshold_pct": THRESHOLD_PCT, "farm_wide_share": 0.5, "farm_wide_min_turbines": 10,
                  "daily_rules": "normal-operation intervals below cut-out with expected power above 50 kW; days with at least 6 hours of them and 5 MWh expected"},
        "accuracy": {
            "mae_kw_model": r(acc["mae_kw_model"].mean(), 2), "mae_kw_power_curve": r(acc["mae_kw_power_curve"].mean(), 2),
            "rmse_kw_model": r(acc["rmse_kw_model"].mean(), 2), "rmse_kw_power_curve": r(acc["rmse_kw_power_curve"].mean(), 2),
            "r2_model": r(acc["r2_model"].mean(), 4), "r2_power_curve": r(acc["r2_power_curve"].mean(), 4),
            "mae_reduction_pct": r(100 * (1 - acc["mae_kw_model"].mean() / acc["mae_kw_power_curve"].mean()), 1),
            "rmse_reduction_pct": r(100 * (1 - acc["rmse_kw_model"].mean() / acc["rmse_kw_power_curve"].mean()), 1),
            "turbines_where_model_beats_curve_on_mae": int((acc["mae_kw_model"] < acc["mae_kw_power_curve"]).sum()),
            "turbines_where_model_beats_curve_on_rmse": int((acc["rmse_kw_model"] < acc["rmse_kw_power_curve"]).sum()),
            **spread,
            "mae_by_wind_speed": by_band.round(2).reset_index().rename(columns={"index": "band"}).to_dict(orient="records"),
            "energy_bias_pct_model": r(acc["energy_bias_pct_model"].mean(), 2),
            "energy_bias_pct_power_curve": r(acc["energy_bias_pct_power_curve"].mean(), 2),
            "train_points_dropped_pct": r(100 * acc["train_dropped_as_unhealthy"].sum() / (acc["train_intervals"].sum() + acc["train_dropped_as_unhealthy"].sum()), 2),
        },
        "feature_importance_share": {hm.FEATURE_LABELS[f]: r(v, 4) for f, v in imp_share.items()},
        "monitoring": {
            "turbine_days_assessed": int(len(flagged)), "daily_residual_median_pct": r(flagged["residual_pct"].median(), 2),
            "daily_residual_p05_pct": r(flagged["residual_pct"].quantile(0.05), 2),
            "farm_wide_days": [f"{d:%Y-%m-%d}" for d in farm["start"]],
            "farm_wide_shortfall_mwh": r(farm["shortfall_mwh"].sum(), 1),
            "turbine_events": int(len(tev)), "turbine_event_days": int(tev["days"].sum()),
            "turbine_event_shortfall_mwh": r(tev["shortfall_mwh"].sum(), 1),
            "turbine_events_with_operator_loss_over_2pct_or_own_status_message": int(((tev["operator_perf_loss_pct"] > 2) | (tev["status_messages"] != "")).sum()),
            "unflagged_turbine_days_meeting_the_same_test_pct": r(100 * float(((100 * others["operator_perf_loss_kwh"] / others["actual_kwh"] > 2) | others["status_messages"].notna()).mean()), 1),
            "farm_wide_days_with_a_message_from_most_turbines": int(sum(
                log[log["day"] == d].groupby("message")["turbine"].nunique().max() >= 0.5 * len(turbines)
                if (log["day"] == d).any() else False for d in farm["start"])),
            "operator_perf_loss_pct_on_turbine_flag_days": r(op_pct(t_days), 2),
            "operator_perf_loss_pct_on_unflagged_days": r(op_pct(others), 2),
        },
        "events": ev.assign(start=ev["start"].dt.strftime("%Y-%m-%d"), end=ev["end"].dt.strftime("%Y-%m-%d")).round(3).to_dict(orient="records"),
    }
    (OUT / "health_results.json").write_text(to_json(results, indent=2))

    # dashboard data: daily series per turbine (flags are recomputed in the page for any threshold)
    days = list(pd.date_range(f"{TEST_YEAR}-01-01", f"{TEST_YEAR}-12-31", freq="D"))  # calm days stay empty
    fleet = flagged["residual_pct"].groupby(level="day").median()
    dash = {"results": {k: v for k, v in results.items() if k != "events"},
            "days": [f"{d:%Y-%m-%d}" for d in days],
            "fleet_median": fleet.reindex(days).round(2).tolist(),
            "day_status": [hm.messages_for_day(log, d) for d in days],
            "turbines": {}, "accuracy": acc.reset_index().round(3).to_dict(orient="list"),
            "importance": {"feature": [hm.FEATURE_LABELS[f] for f in imp_share.index], "share": imp_share.round(4).tolist()},
            "scatter": scatter, "monthly": {"month": [str(m) for m in mres.columns], "turbine": mres.index.tolist(),
                                            "residual": mres.round(2).values.tolist()}}
    for name, g in flagged.groupby(level="turbine"):
        g = g.droplevel("turbine").reindex(days)
        dash["turbines"][name] = {"residual": g["residual_pct"].round(4).tolist(), "expected": (g["expected_kwh"] / 1000).round(3).tolist(),
                                  "actual": (g["actual_kwh"] / 1000).round(3).tolist(),
                                  "op_loss_pct": (100 * g["operator_perf_loss_kwh"] / g["actual_kwh"]).round(2).tolist(),
                                  "wind": g["mean_wind_ms"].round(1).tolist(),
                                  "status": g["status_messages"].where(g["status_messages"].notna(), None).tolist()}
    (OUT / "health_dashboard.json").write_text(to_json(dash, compact=True))

    pd.set_option("display.width", 220)
    print(acc[["mae_kw_model", "mae_kw_power_curve", "rmse_kw_model", "rmse_kw_power_curve", "energy_bias_pct_model"]].round(2).to_string())
    print(to_json({k: results[k] for k in ("accuracy", "feature_importance_share", "monitoring")}, indent=1))
    print(ev.round(2).to_string())


if __name__ == "__main__":
    main()
