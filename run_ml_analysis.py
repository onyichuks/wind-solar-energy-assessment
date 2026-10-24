"""Machine learning checks behind the turbine health model, for docs/ML_TECHNICAL_NOTE.md.

* training curves on a held-out quarter of 2017 (time-blocked, not a random split), and how the
  production setting, early stopping on a random 10 percent of the training data, compares;
* how much training data the model needs (learning curve by months of 2017);
* where the errors are in 2018: by month, hour of day, wind direction sector and turbulence, for the
  boosted model, the linear spline model and the power curve;
* residual autocorrelation of the 10-minute errors;
* a fault injection test: known shortfalls added to the 2018 data to measure how often each reference
  detects them, by day and by week, and how often it flags untouched days.

All three references are compared on the same turbine-days and turbine-weeks.

Usage:  python run_ml_analysis.py            (after scripts/prepare_data.py; about 30 minutes on two cores)
        python run_ml_analysis.py --quick    (two turbines; tables go to a temporary folder, no JSON written)
Writes outputs/ml_results.json and outputs/tables/m*.csv.
"""
from __future__ import annotations

import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from energyassess import health as hm, operational as op, resource as rs  # noqa: E402
from energyassess.outputs import to_json  # noqa: E402
from compare_models import models as candidate_models  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
HUB = 59.0
THRESHOLD_DAY, THRESHOLD_WEEK = -5.0, -3.0
FAULTS = (0.02, 0.03, 0.05, 0.08, 0.10)
LOCAL = "Europe/London"
T0 = time.time()
QUICK = "--quick" in sys.argv  # two turbines; tables to a temporary folder, no JSON
REFS = {"expected_kw": "Gradient boosting", "spline_kw": "Spline regression (linear)", "curve_kw": "Power curve"}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def guard(model, X: pd.DataFrame, raw: np.ndarray, rated_kw: float = 2050.0) -> np.ndarray:
    """The same clipping and cut-out rule as health.predict, for staged predictions."""
    p = np.clip(raw, -20.0, rated_kw * 1.01)
    return np.where(X["wind_ms"].to_numpy() >= hm.CUT_OUT_MS, 0.0, p)


def training_curve(tr: pd.DataFrame, va: pd.DataFrame, iters: int = 1500, every: int = 5) -> dict:
    m = hm.make_model().set_params(max_iter=iters, early_stopping=False).fit(tr[hm.FEATURES], tr["power_kw"])
    smp = tr.sample(min(10000, len(tr)), random_state=0)
    out = {}
    for name, d in (("train", smp), ("validation", va)):
        y = d["power_kw"].to_numpy()
        vals = [float(np.mean(np.abs(guard(m, d, p) - y)))
                for i, p in enumerate(m.staged_predict(d[hm.FEATURES]), start=1) if i % every == 0]
        out[name] = vals
    # the production setting on the same data: early stopping on a random 10 percent, scored as it stands
    es = hm.make_model().fit(tr[hm.FEATURES], tr["power_kw"])
    v = np.array(out["validation"])
    yv = va["power_kw"].to_numpy()
    out["best_iteration"] = (int(np.argmin(v)) + 1) * every
    out["early_stopping_iteration"] = int(es.n_iter_)
    out["mae_at_best"] = float(v.min())
    out["mae_early_stopped_model"] = float(np.mean(np.abs(guard(es, va, es.predict(va[hm.FEATURES])) - yv)))
    out["mae_at_1500"] = float(v[-1])
    return out


def common_days(dailies) -> pd.MultiIndex:
    """Turbine-days assessed under every reference, so spreads and detection rates compare like with like."""
    dailies = list(dailies)
    idx = dailies[0].index
    for d in dailies[1:]:
        idx = idx.intersection(d.index)
    return idx.sort_values()


def lag_autocorr(e: pd.Series, lag: int) -> float:
    return float(e.autocorr(lag))


def seg(pred: pd.DataFrame, key: pd.Series, name: str) -> pd.DataFrame:
    rows = []
    for k, g in pred.groupby(key, observed=True):
        r = {name: k, "intervals": len(g), "actual_mean_kw": g["power_kw"].mean()}
        for col, label in REFS.items():
            r[f"mae_kw {label}"] = (g["power_kw"] - g[col]).abs().mean()
        rows.append(r)
    return pd.DataFrame(rows)


def weekly(daily: pd.DataFrame, min_days: int = 3) -> pd.DataFrame:
    d = daily.reset_index()
    d["week"] = d["day"].dt.to_period("W-SUN")
    g = d.groupby(["turbine", "week"])
    w = pd.DataFrame({"actual_kwh": g["actual_kwh"].sum(), "expected_kwh": g["expected_kwh"].sum(), "days": g.size()})
    w = w[w["days"] >= min_days].copy()
    w["residual_pct"] = 100 * (w["actual_kwh"] / w["expected_kwh"] - 1)
    return w


def fault_test(daily: pd.DataFrame, weeks: pd.DataFrame, seed: int = 0, per_turbine: int = 8) -> dict:
    """Scale actual energy by (1 - fault) on chosen turbine-weeks and count how often the flags catch it.

    A uniform shortfall is a simplification (real faults such as pitch or yaw errors depend on wind
    speed), but it measures the one thing that matters for alarms: how far normal days scatter."""
    rng = np.random.default_rng(seed)
    r = daily["residual_pct"] / 100
    rw = weeks["residual_pct"] / 100
    wk_of_day = daily.index.get_level_values("day").to_period("W-SUN")
    turb = daily.index.get_level_values("turbine")
    chosen = set()
    for t in sorted(set(turb)):
        avail = sorted(weeks.loc[t].index) if t in weeks.index.get_level_values(0) else []
        for w in rng.choice(len(avail), size=min(per_turbine, len(avail)), replace=False):
            chosen.add((t, avail[w]))
    in_day = np.array([(t, w) in chosen for t, w in zip(turb, wk_of_day)])
    in_week = np.array([k in chosen for k in weeks.index])
    # Thresholds that flag exactly 1 percent of untouched days (or weeks): a fair comparison, because a
    # noisier reference only "detects" more at the fixed rule by also flagging many healthy days.
    thr_day = float(np.percentile(r[~in_day] * 100, 1))
    thr_week = float(np.percentile(rw[~in_week] * 100, 1))
    out = {"windows": len(chosen),
           "untouched_day_flag_rate_pct": float(100 * (r[~in_day] * 100 < THRESHOLD_DAY).mean()),
           "untouched_week_flag_rate_pct": float(100 * (rw[~in_week] * 100 < THRESHOLD_WEEK).mean()),
           "threshold_day_at_1pct_pct": thr_day, "threshold_week_at_1pct_pct": thr_week,
           "faults": []}
    for f in FAULTS:
        rd = ((1 + r[in_day]) * (1 - f) - 1) * 100
        dflag = rd < THRESHOLD_DAY
        keys = list(zip(turb[in_day], wk_of_day[in_day]))
        caught_window = pd.Series(dflag.to_numpy(), index=pd.MultiIndex.from_tuples(keys)).groupby(level=[0, 1]).any()
        rwk = ((1 + rw[in_week]) * (1 - f) - 1) * 100
        caught_window_1pct = pd.Series((rd < thr_day).to_numpy(),
                                       index=pd.MultiIndex.from_tuples(keys)).groupby(level=[0, 1]).any()
        out["faults"].append({"fault_pct": 100 * f, "day_detection_pct": float(100 * dflag.mean()),
                              "window_caught_by_daily_flag_pct": float(100 * caught_window.mean()),
                              "window_caught_by_daily_flag_at_1pct_pct": float(100 * caught_window_1pct.mean()),
                              "week_detection_pct": float(100 * (rwk < THRESHOLD_WEEK).mean()),
                              "day_detection_at_1pct_pct": float(100 * (rd < thr_day).mean()),
                              "week_detection_at_1pct_pct": float(100 * (rwk < thr_week).mean())})
    return out


def fault_test_repeated(daily: pd.DataFrame, weeks: pd.DataFrame, seeds=range(20)) -> dict:
    """fault_test averaged over 20 random draws of the turbine-weeks, to steady the rates."""
    runs = [fault_test(daily, weeks, seed=s) for s in seeds]
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if k != "faults"}
    out["windows"] = int(runs[0]["windows"])
    out["draws"] = len(runs)
    out["faults"] = [{k: float(np.mean([r["faults"][i][k] for r in runs])) for k in runs[0]["faults"][i]}
                     for i in range(len(FAULTS))]
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    TABLES.mkdir(parents=True, exist_ok=True)
    turbines = pd.read_csv(ROOT / "static_data" / "penmanshiel_turbines.csv")
    era5 = rs.load_era5_openmeteo(DATA / "raw" / "era5_penmanshiel_openmeteo.csv")
    rho = pd.Series(rs.air_density(era5["t2m"], era5["sp"], era5.attrs["elevation_m"], turbines["elevation_m"].mean() + HUB),
                    index=era5.index)
    weather = pd.DataFrame({"air_temp_c": era5["t2m"], "air_density": rho})
    scada = hm.add_features(op.apply_qc(pd.read_parquet(DATA / "processed" / "turbine_scada_10min.parquet")), weather)
    data = scada[hm.normal_operation(scada)]
    spline_make = candidate_models()["Spline regression"]
    lc_sets = {"last 1 month": [12], "last 2 months": [11, 12], "last 3 months": [10, 11, 12],
               "3 months spread through the year (Feb, Jun, Oct)": [2, 6, 10], "last 6 months": list(range(7, 13)),
               "last 9 months": list(range(4, 13)), "all 12 months": list(range(1, 13))}

    preds, curves, iters, lc_rows = [], {}, [], []
    if QUICK:
        data = data[data["turbine"].isin(sorted(data["turbine"].unique())[:2])]
        globals()["TABLES"] = Path(tempfile.mkdtemp(prefix="ml_quick_"))
        log(f"quick mode: tables go to {TABLES}")
    for name, g in data.groupby("turbine"):
        y17 = g[g["time"].dt.year == 2017]
        test = g[g["time"].dt.year == 2018].copy()
        hm.check_missing(y17, test)  # same guard as run_health.py
        tr = y17[hm.healthy_mask(y17)]
        gb = hm.fit_model(tr)
        test["expected_kw"] = hm.predict(gb, test)
        test["spline_kw"] = hm.predict(spline_make().fit(tr[hm.FEATURES], tr["power_kw"]), test)
        test["curve_kw"] = hm.curve_predict(hm.binned_curve(tr), test["wind_ms"])
        iters.append({"turbine": name, "production_iterations": int(gb.n_iter_)})
        # training curve: train January to September 2017, validate October to December 2017
        q4 = y17["time"].dt.quarter == 4
        trc = y17[~q4]
        curves[name] = training_curve(trc[hm.healthy_mask(trc)], y17[q4])
        # learning curve on 2018
        for label, months in lc_sets.items():
            sub = y17[y17["time"].dt.month.isin(months)]
            sub = sub[hm.healthy_mask(sub)]
            col = f"lc::{label}"
            test[col] = test["expected_kw"] if label == "all 12 months" else hm.predict(hm.fit_model(sub), test)
            lc_rows.append({"turbine": name, "training": label, "train_intervals": len(sub),
                            "mae_kw": float((test["power_kw"] - test[col]).abs().mean())})
        preds.append(test)
        log(f"{name} done")

    pred = pd.concat(preds, ignore_index=True)

    # accuracy on 2018
    # means over the 14 turbines, as in the research report (health_results.json)
    acc = {}
    for col, label in REFS.items():
        per = [hm.accuracy(g["power_kw"], g[col]) for _, g in pred.groupby("turbine")]
        acc[label] = {"mae_kw": float(np.mean([a["mae_kw"] for a in per])),
                      "rmse_kw": float(np.mean([a["rmse_kw"] for a in per])),
                      "energy_bias_pct": float(np.mean([a["energy_bias_pct"] for a in per]))}

    # training curves, averaged over turbines
    it = list(range(5, 1501, 5))
    tc = {"iterations": it,
          "train_mean": np.mean([curves[t]["train"] for t in curves], axis=0).tolist(),
          "validation_mean": np.mean([curves[t]["validation"] for t in curves], axis=0).tolist(),
          "per_turbine": [{"turbine": t, "best_iteration": c["best_iteration"],
                           "early_stopping_iteration": c["early_stopping_iteration"],
                           "mae_at_best": c["mae_at_best"], "mae_early_stopped_model": c["mae_early_stopped_model"],
                           "mae_at_1500": c["mae_at_1500"]}
                          for t, c in curves.items()]}
    pd.DataFrame(tc["per_turbine"]).merge(pd.DataFrame(iters), on="turbine").to_csv(TABLES / "m01_training_curves.csv", index=False)

    # learning curve with daily spread for each training set
    lc = pd.DataFrame(lc_rows).groupby("training", sort=False).agg(train_intervals=("train_intervals", "mean"),
                                                                    mae_kw=("mae_kw", "mean")).reset_index()
    lc_daily = {label: hm.daily_residuals(pred, expected_col=f"lc::{label}") for label in lc_sets}
    common = common_days(lc_daily.values())
    lc["daily_spread_pct"] = lc["training"].map(
        {label: float(d.loc[common, "residual_pct"].std()) for label, d in lc_daily.items()})
    lc["turbine_days"] = len(common)
    lc.to_csv(TABLES / "m02_learning_curve.csv", index=False)
    log("learning curve done")

    # where the errors are
    loc = pred["time"].dt.tz_convert(LOCAL)
    sector = (((pred["wind_dir_deg"] + 15) % 360) // 30 * 30).astype("Int64")  # missing direction stays missing
    segs = {"month": loc.dt.month, "hour": loc.dt.hour, "direction_sector_deg": sector,
            "turbulence_band": pd.cut(pred["turbulence"], [0, 0.06, 0.09, 0.12, 0.15, 0.2, 5],
                                      labels=["below 6%", "6 to 9%", "9 to 12%", "12 to 15%", "15 to 20%", "above 20%"]),
            "wind_band": pd.cut(pred["wind_ms"], [0, 3, 6, 9, 12, 15, 25, 40], right=False,
                                labels=["0 to 3", "3 to 6", "6 to 9", "9 to 12", "12 to 15", "15 to 25", "25 and above"])}
    seg_out = {}
    for name, key in segs.items():
        t = seg(pred, key, name)
        t.to_csv(TABLES / f"m03_errors_by_{name}.csv", index=False)
        seg_out[name] = t.to_dict("records")

    # residual autocorrelation (contiguous 10-minute series per turbine)
    ac = {1: [], 6: [], 144: []}
    for _, g in pred.sort_values("time").groupby("turbine"):
        e = (g.set_index("time")["power_kw"] - g.set_index("time")["expected_kw"]).asfreq("10min")
        for k in ac:
            ac[k].append(lag_autocorr(e, k))
    autocorr = {"lag_10_min": float(np.nanmean(ac[1])), "lag_1_hour": float(np.nanmean(ac[6])),
                "lag_1_day": float(np.nanmean(ac[144]))}

    # daily and weekly spread, and the fault injection test, for each reference
    spread, faults = {}, {}
    ref_daily = {label: hm.daily_residuals(pred, expected_col=col) for col, label in REFS.items()}
    common = common_days(ref_daily.values())
    for label, d in ref_daily.items():
        d = d.loc[common]
        w = weekly(d)
        spread[label] = {"turbine_days": int(len(d)), "daily_std_pct": float(d["residual_pct"].std()),
                         "turbine_weeks": int(len(w)), "weekly_std_pct": float(w["residual_pct"].std())}
        faults[label] = fault_test_repeated(d, w)
    rows = [{"reference": lab, **f} for lab, v in faults.items() for f in v["faults"]]
    pd.DataFrame(rows).to_csv(TABLES / "m04_fault_injection.csv", index=False)
    log("fault test done")

    res = {"setup": {"train": "2017 normal operation, robust filter", "test": "2018",
                     "training_curve_split": "train January to September 2017, validate October to December 2017",
                     "fault_injection": "uniform shortfall on 8 random turbine-weeks per turbine, 20 random draws averaged; fixed rule -5 percent per day and -3 percent per week, and thresholds that flag 1 percent of untouched days or weeks",
                     "references": list(REFS.values())},
           "accuracy_2018": acc, "training_curve": tc, "learning_curve": lc.to_dict("records"),
           "segments": seg_out, "residual_autocorrelation": autocorr, "spread": spread, "fault_test": faults}
    if not QUICK:
        (OUT / "ml_results.json").write_text(to_json(res, indent=1))
    print(to_json({k: res[k] for k in ("accuracy_2018", "learning_curve", "residual_autocorrelation", "spread")}, indent=1))
    print(to_json(faults, indent=1))
    print(pd.DataFrame(tc["per_turbine"]).merge(pd.DataFrame(iters), on="turbine").to_string())
    log("done")


if __name__ == "__main__":
    main()
