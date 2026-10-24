"""Which machine learning model? Candidate normal-behaviour models compared for every turbine.

The model is chosen by blocked cross-validation inside 2017 only: each calendar quarter is held
out in turn and the model trained on the other three, scored by mean absolute error (the
criterion set before any scoring). 2018 is scored once, at the end, as the final test, with the
same daily flag rules as run_health.py. Every model uses standard settings fixed in advance.

Usage:  python scripts/compare_models.py      (after scripts/prepare_data.py; about 25 minutes on two cores)
Writes outputs/tables/h07_model_comparison.csv and h08_model_comparison_by_turbine.csv.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from energyassess import health as hm, operational as op, resource as rs  # noqa: E402

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv  # one turbine, for a fast check
TABLES = ROOT / "outputs" / "tables"
CHOSEN = "Gradient boosting (used)"


def models():
    """Standard settings fixed before any scoring; none were tuned on 2017 folds or 2018."""
    return {
        "Gradient boosting (used)": lambda: hm.make_model(seed=0),  # the model in run_health.py
        "Random forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=150, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=0)),
        "Extra trees": lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(
            n_estimators=150, min_samples_leaf=10, max_features=0.7, n_jobs=-1, random_state=0)),
        "K-nearest neighbours": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                      KNeighborsRegressor(n_neighbors=30, weights="distance")),
        "Neural network (MLP)": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 64), alpha=1e-3, learning_rate_init=1e-3, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, max_iter=300, random_state=0)),
        "Spline regression": lambda: make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=12, degree=3),
                                                   StandardScaler(), Ridge(alpha=1.0)),
    }


def fit_predict(make, train, test):
    m = make().fit(train[hm.FEATURES], train["power_kw"])
    return hm.predict(m, test)


def scores(actual, pred):
    e = np.asarray(actual, float) - np.asarray(pred, float)
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def main():
    turbines = pd.read_csv(ROOT / "static_data" / "penmanshiel_turbines.csv")
    era5 = rs.load_era5_openmeteo(ROOT / "data" / "raw" / "era5_penmanshiel_openmeteo.csv")
    rho = pd.Series(rs.air_density(era5["t2m"], era5["sp"], era5.attrs["elevation_m"], turbines["elevation_m"].mean() + 59.0),
                    index=era5.index)
    weather = pd.DataFrame({"air_temp_c": era5["t2m"], "air_density": rho})
    scada = hm.add_features(op.apply_qc(pd.read_parquet(ROOT / "data" / "processed" / "turbine_scada_10min.parquet")), weather)
    data = scada[hm.normal_operation(scada)]
    names = sorted(data["turbine"].unique())
    if QUICK:
        names = names[:1]
    M = models()
    cv_rows, test_rows, preds, timing = [], [], [], {k: 0.0 for k in M}
    for name in names:
        g = data[data["turbine"] == name]
        y17 = g[g["time"].dt.year == 2017]
        test = g[g["time"].dt.year == 2018].copy()
        q = y17["time"].dt.quarter
        # 1) blocked cross-validation inside 2017
        for fold in (1, 2, 3, 4):
            tr_all, va = y17[q != fold], y17[q == fold]
            tr = tr_all[hm.healthy_mask(tr_all)]
            curve = hm.binned_curve(tr)
            mae, rmse = scores(va["power_kw"], hm.curve_predict(curve, va["wind_ms"]))
            cv_rows.append({"turbine": name, "fold": fold, "model": "Power curve (baseline)", "mae_kw": mae, "rmse_kw": rmse})
            for label, make in M.items():
                t0 = time.time()
                p = fit_predict(make, tr, va)
                timing[label] += time.time() - t0
                mae, rmse = scores(va["power_kw"], p)
                cv_rows.append({"turbine": name, "fold": fold, "model": label, "mae_kw": mae, "rmse_kw": rmse})
        # 2) final fit on all of 2017, scored once on 2018
        tr = y17[hm.healthy_mask(y17)]
        curve = hm.binned_curve(tr)
        out = test[["turbine", "time", "power_kw", "wind_ms", "lost_performance_kwh", "air_temp_c"]].copy()
        out["Power curve (baseline)"] = hm.curve_predict(curve, test["wind_ms"])
        for label, make in M.items():
            out[label] = fit_predict(make, tr, test)
        for label in ["Power curve (baseline)", *M]:
            mae, rmse = scores(out["power_kw"], out[label])
            e = out["power_kw"].sum() / out[label].sum() - 1
            test_rows.append({"turbine": name, "model": label, "mae_kw": mae, "rmse_kw": rmse, "energy_bias_pct": 100 * e})
        preds.append(out)
        print(name, "done", flush=True)

    cv = pd.DataFrame(cv_rows)
    ts = pd.DataFrame(test_rows)
    pred = pd.concat(preds, ignore_index=True)
    labels = ["Power curve (baseline)", *M]
    # daily residual spread and flags for each model, same rules as run_health.py
    flags, daily_stats = {}, {}
    for label in labels:
        d = hm.daily_residuals(pred, expected_col=label)
        f = hm.flag_days(d, -5.0)
        flags[label] = f
        daily_stats[label] = {"daily_std_pct": d["residual_pct"].std(),
                              "daily_iqr_pct": float(np.subtract(*d["residual_pct"].quantile([0.75, 0.25]))),
                              "farm_wide_days": int(f.loc[f["scope"] == "farm-wide"].index.get_level_values("day").nunique()),
                              "single_turbine_flag_days": int((f["scope"] == "turbine").sum())}
    chosen_days = flags[CHOSEN].index[flags[CHOSEN]["scope"] == "turbine"]
    for label in labels:
        flagged = flags[label].index[flags[label]["flag"]]
        daily_stats[label]["chosen_single_turbine_days_also_flagged"] = int(sum(i in flagged for i in chosen_days))
    cv_turbine = cv.groupby(["turbine", "model"])["mae_kw"].mean().unstack()
    summary = pd.DataFrame({
        "cv_2017_mae_kw": cv.groupby("model")["mae_kw"].mean(),
        "cv_2017_rmse_kw": cv.groupby("model")["rmse_kw"].mean(),
        "turbines_best_on_cv_mae": cv_turbine.idxmin(axis=1).value_counts(),
        "test_2018_mae_kw": ts.groupby("model")["mae_kw"].mean(),
        "test_2018_rmse_kw": ts.groupby("model")["rmse_kw"].mean(),
        "test_2018_energy_bias_pct": ts.groupby("model")["energy_bias_pct"].mean(),
        "turbines_best_on_2018_mae": ts.loc[ts.groupby("turbine")["mae_kw"].idxmin(), "model"].value_counts(),
    }).join(pd.DataFrame(daily_stats).T)
    for c in ("turbines_best_on_cv_mae", "turbines_best_on_2018_mae"):
        summary[c] = summary[c].fillna(0).astype(int)
    summary["fit_seconds_in_cv"] = pd.Series(timing)
    summary = summary.sort_values("cv_2017_mae_kw")
    summary.index.name = "model"
    pd.set_option("display.width", 250)
    print(summary.round(2).to_string())
    if not QUICK:
        TABLES.mkdir(parents=True, exist_ok=True)
        summary.round(3).to_csv(TABLES / "h07_model_comparison.csv")
        ts.round(3).to_csv(TABLES / "h08_model_comparison_by_turbine.csv", index=False)


if __name__ == "__main__":
    main()
