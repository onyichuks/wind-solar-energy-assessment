"""Turbine health: a machine learning normal-behaviour model of each turbine's power.

The model learns how much power a healthy turbine produces for the conditions it sees
(nacelle wind speed, turbulence, wind direction, air temperature and air density). Trained on
one year and applied to the next, the gap between actual and expected energy shows when a
turbine underperforms. Days when most turbines fall short together point to an external
cause (storm control, weather, grid); days when one turbine falls short point to the turbine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["wind_ms", "turbulence", "dir_sin", "dir_cos", "air_temp_c", "air_density"]
FEATURE_LABELS = {"wind_ms": "Nacelle wind speed", "turbulence": "Turbulence intensity", "dir_sin": "Wind direction (east-west)",
                  "dir_cos": "Wind direction (north-south)", "air_temp_c": "Air temperature (ERA5)", "air_density": "Air density (ERA5)"}
BINS = np.arange(0.0, 40.5, 0.5)  # covers every wind speed in the data, so high-wind points are judged, not dropped
CUT_OUT_MS = 25.0


def add_features(df: pd.DataFrame, weather_hourly: pd.DataFrame) -> pd.DataFrame:
    """Model inputs: wind speed, turbulence and direction from the turbine; air temperature and density from ERA5.

    ERA5 is used for temperature because the nacelle temperature signal is missing for weeks at a
    time in 2018 (all of March, for example) but never in 2017, and a model meets those gaps
    untrained. `weather_hourly` has columns air_temp_c and air_density on an hourly UTC index.
    """
    out = df.copy()
    out["turbulence"] = out["wind_std_ms"] / out["wind_ms"].clip(lower=1.0)
    rad = np.radians(out["wind_dir_deg"])
    out["dir_sin"], out["dir_cos"] = np.sin(rad), np.cos(rad)
    hours = out["time"].dt.floor("h")
    for col in ("air_temp_c", "air_density"):
        out[col] = weather_hourly[col].reindex(hours).to_numpy()
    return out


def missing_shares(df: pd.DataFrame) -> pd.Series:
    return df[FEATURES].isna().mean()


def check_missing(train: pd.DataFrame, test: pd.DataFrame, tolerance: float = 0.01) -> None:
    """Refuse to score a test set with clearly more missing inputs than the training set."""
    gap = missing_shares(test) - missing_shares(train)
    bad = gap[gap > tolerance]
    if len(bad):
        raise ValueError(f"Inputs missing far more often in the test data than in training: {bad.round(3).to_dict()}")


def normal_operation(df: pd.DataFrame) -> pd.Series:
    """Valid data and no downtime recorded: the turbine was available and meant to produce."""
    return df["qc_valid"] & (df["lost_downtime_kwh"].fillna(0) <= 0)


def healthy_mask(df: pd.DataFrame, k: float = 4.0, floor_kw: float = 10.0) -> pd.Series:
    """Drop training points far from the typical power for their wind speed.

    A robust filter: in each 0.5 m/s bin, points more than k scaled median absolute
    deviations from the bin median are left out, so partial faults in the training year
    do not become part of what the model treats as normal.
    """
    b = pd.cut(df["wind_ms"], BINS)
    med = df.groupby(b, observed=True)["power_kw"].transform("median")
    mad = (df["power_kw"] - med).abs().groupby(b, observed=True).transform("median")
    return (df["power_kw"] - med).abs() <= k * 1.4826 * mad.clip(lower=floor_kw)


def make_model(seed: int = 0):
    """Gradient-boosted trees (scikit-learn), settings fixed in advance, 10 percent held out for early stopping.

    Chosen over random forest, extra trees, a neural network, k-nearest neighbours and spline
    regression by cross-validation within the training year (scripts/compare_models.py)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, random_state=seed)


def fit_model(train: pd.DataFrame, seed: int = 0):
    return make_model(seed).fit(train[FEATURES], train["power_kw"])


def predict(model, df: pd.DataFrame, rated_kw: float = 2050.0) -> np.ndarray:
    """Expected power. At or above the 25 m/s cut-out the turbine stops by design, a rule the
    trees cannot learn from the handful of storm intervals in one year, so it is applied directly."""
    p = np.clip(model.predict(df[FEATURES]), -20.0, rated_kw * 1.01)
    return np.where(df["wind_ms"].to_numpy() >= CUT_OUT_MS, 0.0, p)


def binned_curve(train: pd.DataFrame) -> pd.Series:
    """Baseline: the turbine's own power curve by the method of bins (mean power per 0.5 m/s bin)."""
    g = train.groupby(pd.cut(train["wind_ms"], BINS), observed=True)["power_kw"].mean()
    g.index = [iv.mid for iv in g.index]
    return g


def curve_predict(curve: pd.Series, wind_ms) -> np.ndarray:
    return np.interp(np.asarray(wind_ms, dtype=float), curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float))


def accuracy(actual, expected) -> dict:
    a, p = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    e = a - p
    return {"rmse_kw": float(np.sqrt(np.mean(e ** 2))), "mae_kw": float(np.mean(np.abs(e))),
            "r2": float(1 - np.sum(e ** 2) / np.sum((a - a.mean()) ** 2)), "energy_bias_pct": float(100 * (a.sum() / p.sum() - 1))}


def daily_residuals(df: pd.DataFrame, min_expected_kwh: float = 5000.0, min_intervals: int = 36,
                    expected_col: str = "expected_kw") -> pd.DataFrame:
    """Actual against expected energy per turbine and day, over normal-operation intervals.

    Intervals with expected power under 50 kW are left out (idling noise), and so are intervals
    at or above the 25 m/s cut-out, where the turbine stops by design. A day needs at least six
    hours of the remaining intervals and 5 MWh of expected energy, so that light-wind days do
    not raise false alarms.
    """
    d = df[(df[expected_col] > 50) & (df["wind_ms"] < CUT_OUT_MS)].copy()
    d["expected_kw"] = d[expected_col]
    d["day"] = d["time"].dt.tz_convert(None).dt.floor("D")
    g = d.groupby(["turbine", "day"])
    out = pd.DataFrame({"actual_kwh": g["power_kw"].sum() / 6, "expected_kwh": g["expected_kw"].sum() / 6,
                        "intervals": g.size(), "operator_perf_loss_kwh": g["lost_performance_kwh"].sum(),
                        "mean_wind_ms": g["wind_ms"].mean(), "max_wind_ms": g["wind_ms"].max(),
                        "share_above_20ms": g["wind_ms"].apply(lambda v: float((v >= 20).mean())),
                        "mean_temp_c": g["air_temp_c"].mean()})
    out = out[(out["expected_kwh"] >= min_expected_kwh) & (out["intervals"] >= min_intervals)].copy()
    out["residual_pct"] = 100 * (out["actual_kwh"] / out["expected_kwh"] - 1)
    return out


def flag_days(daily: pd.DataFrame, threshold_pct: float = -5.0, farm_share: float = 0.5, min_turbines: int = 10) -> pd.DataFrame:
    """Flag turbine-days below the threshold.

    A day is farm-wide when at least `min_turbines` turbines were assessed and at least
    `farm_share` of them are flagged; other flags are single-turbine events.
    """
    out = daily.copy()
    out["flag"] = out["residual_pct"] < threshold_pct
    days = out.index.get_level_values("day")
    share = out["flag"].groupby(level="day").mean()
    assessed = out["flag"].groupby(level="day").size()
    out["farm_share_flagged"] = share.reindex(days).to_numpy()
    out["turbines_assessed"] = assessed.reindex(days).to_numpy()
    farm = (out["farm_share_flagged"] >= farm_share) & (out["turbines_assessed"] >= min_turbines)
    out["scope"] = np.where(~out["flag"], "", np.where(farm, "farm-wide", "turbine"))
    return out


def status_log_by_day(status: pd.DataFrame) -> pd.DataFrame:
    """Stop, warning and partial-performance messages from the turbine status logs, one row per turbine, day and message."""
    s = status[status["status"].isin(["Stop", "Warning"]) | (status["iec_category"] == "Partial Performance")].copy()
    s["end"] = s["end"].fillna(s["start"] + pd.Timedelta("10min"))
    rows = []
    for e in s.itertuples():
        for day in pd.date_range(e.start.tz_convert(None).floor("D"), e.end.tz_convert(None).floor("D"), freq="D"):
            rows.append((e.turbine, day, e.message))
    return pd.DataFrame(rows, columns=["turbine", "day", "message"]).drop_duplicates()


def messages_per_turbine_day(log: pd.DataFrame, n_turbines: int | None = None, max_share: float = 0.5) -> pd.Series:
    """Messages per turbine and day. With `n_turbines`, only turbine-specific ones: a message
    logged that day by at least `max_share` of the turbines is a site-wide event, not the turbine's own."""
    if n_turbines:
        spread = log.groupby(["day", "message"])["turbine"].transform("nunique")
        log = log[spread < max_share * n_turbines]
    return log.groupby(["turbine", "day"])["message"].agg(lambda x: "; ".join(sorted(set(x))))


def messages_for_day(log: pd.DataFrame, day, top: int = 3) -> str:
    """The most widespread status messages on one day, with the number of turbines that logged each."""
    counts = log[log["day"] == day].groupby("message")["turbine"].nunique().sort_values(ascending=False).head(top)
    return "; ".join(f"{m} ({n} turbine{'s' if n > 1 else ''})" for m, n in counts.items())


def events(flagged: pd.DataFrame) -> pd.DataFrame:
    """Group flagged days: farm-wide days as one event per day, turbine days as runs of consecutive days per turbine."""
    f = flagged[flagged["flag"]].reset_index()
    rows = []
    for day, g in f[f["scope"] == "farm-wide"].groupby("day"):
        rows.append({"scope": "farm-wide", "turbines": f"{len(g)} of {int(g['turbines_assessed'].iloc[0])}",
                     "start": day, "end": day, "days": 1, "residual_pct": 100 * (g["actual_kwh"].sum() / g["expected_kwh"].sum() - 1),
                     "shortfall_mwh": (g["expected_kwh"] - g["actual_kwh"]).sum() / 1000,
                     "operator_perf_loss_pct": 100 * g["operator_perf_loss_kwh"].sum() / g["actual_kwh"].sum(),
                     "mean_wind_ms": g["mean_wind_ms"].mean(), "max_wind_ms": g["max_wind_ms"].max(),
                     "share_above_20ms": g["share_above_20ms"].mean(), "mean_temp_c": g["mean_temp_c"].mean()})
    t = f[f["scope"] == "turbine"].sort_values(["turbine", "day"])
    for turbine, g in t.groupby("turbine"):
        run = (g["day"].diff() != pd.Timedelta("1D")).cumsum()
        for _, r in g.groupby(run):
            rows.append({"scope": "turbine", "turbines": turbine, "start": r["day"].min(), "end": r["day"].max(), "days": len(r),
                         "residual_pct": 100 * (r["actual_kwh"].sum() / r["expected_kwh"].sum() - 1),
                         "shortfall_mwh": (r["expected_kwh"] - r["actual_kwh"]).sum() / 1000,
                         "operator_perf_loss_pct": 100 * r["operator_perf_loss_kwh"].sum() / r["actual_kwh"].sum(),
                         "mean_wind_ms": r["mean_wind_ms"].mean(), "max_wind_ms": r["max_wind_ms"].max(),
                         "share_above_20ms": r["share_above_20ms"].mean(), "mean_temp_c": r["mean_temp_c"].mean()})
    cols = ["scope", "turbines", "start", "end", "days", "residual_pct", "shortfall_mwh", "operator_perf_loss_pct",
            "mean_wind_ms", "max_wind_ms", "share_above_20ms", "mean_temp_c"]
    return pd.DataFrame(rows, columns=cols).sort_values(["scope", "start"]).reset_index(drop=True)
