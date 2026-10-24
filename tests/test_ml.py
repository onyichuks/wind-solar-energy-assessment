"""Unit tests for the checks in run_ml_analysis.py (docs/ML_TECHNICAL_NOTE.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_ml_analysis as M  # noqa: E402


def _daily(noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Two turbines, every day of 2018, actual equal to expected (plus optional noise)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2018-01-01", "2018-12-31", freq="D")
    idx = pd.MultiIndex.from_product([["T01", "T02"], days], names=["turbine", "day"])
    exp = np.full(len(idx), 20000.0)
    act = exp * (1 + rng.normal(0, noise, len(idx)))
    d = pd.DataFrame({"actual_kwh": act, "expected_kwh": exp}, index=idx)
    d["residual_pct"] = 100 * (d["actual_kwh"] / d["expected_kwh"] - 1)
    return d


def test_weekly_residual_is_energy_weighted():
    d = _daily()
    # one lost day in the first week of T01, on a day with twice the usual expected energy
    d.iloc[0, d.columns.get_loc("expected_kwh")] = 40000.0
    d.iloc[0, d.columns.get_loc("actual_kwh")] = 0.0
    w = M.weekly(d)
    first = w.loc["T01"].iloc[0]
    assert first["days"] == 7
    # energy weighted: 120 MWh produced against 160 MWh expected is -25 percent
    # (a plain mean of the daily percentages would give -100/7)
    assert first["residual_pct"] == pytest.approx(-25.0)


def test_common_days_keeps_only_shared_turbine_days():
    a, b = _daily(), _daily().iloc[5:]
    idx = M.common_days([a, b])
    assert len(idx) == len(a) - 5 and idx.is_monotonic_increasing


def test_fault_test_detects_by_size():
    d = _daily()
    w = M.weekly(d)
    out = M.fault_test(d, w, per_turbine=4)
    assert out["windows"] == 8
    assert out["untouched_day_flag_rate_pct"] == 0.0
    by = {f["fault_pct"]: f for f in out["faults"]}
    # with no noise, a shortfall is caught exactly when it passes the fixed thresholds
    assert by[2.0]["day_detection_pct"] == 0.0 and by[2.0]["week_detection_pct"] == 0.0
    assert by[5.0]["week_detection_pct"] == 100.0
    assert by[8.0]["day_detection_pct"] == 100.0 and by[8.0]["window_caught_by_daily_flag_pct"] == 100.0
    # with no scatter, the 1 percent thresholds sit at zero, so every shortfall is caught on every day
    assert out["threshold_day_at_1pct_pct"] == 0.0
    assert by[2.0]["day_detection_at_1pct_pct"] == 100.0 and by[2.0]["window_caught_by_daily_flag_at_1pct_pct"] == 100.0


def test_training_curve_scores_every_fifth_iteration():
    rng = np.random.default_rng(3)
    n = 3000
    wind = rng.uniform(3, 15, n)
    df = pd.DataFrame({"wind_ms": wind, "turbulence": rng.uniform(0.05, 0.2, n), "dir_sin": 0.0, "dir_cos": 1.0,
                       "air_temp_c": 8.0, "air_density": 1.22})
    df["power_kw"] = np.clip((wind - 3) ** 3, 0, 2000) + rng.normal(0, 20, n)
    out = M.training_curve(df.iloc[:2400], df.iloc[2400:], iters=60, every=5)
    assert len(out["validation"]) == len(out["train"]) == 12
    assert out["best_iteration"] % 5 == 0 and 5 <= out["best_iteration"] <= 60
    assert out["mae_at_best"] == pytest.approx(min(out["validation"]))
    assert out["mae_at_1500"] == pytest.approx(out["validation"][-1])
    assert np.isfinite(out["mae_early_stopped_model"]) and out["early_stopping_iteration"] >= 1


def test_noisier_reference_needs_a_lower_threshold():
    quiet, noisy = _daily(0.02), _daily(0.05)
    tq = M.fault_test(quiet, M.weekly(quiet))["threshold_day_at_1pct_pct"]
    tn = M.fault_test(noisy, M.weekly(noisy))["threshold_day_at_1pct_pct"]
    assert tn < tq < 0


def test_published_ml_results_match_the_report():
    p = ROOT / "outputs" / "ml_results.json"
    if not p.exists():
        pytest.skip("run run_ml_analysis.py first")
    R = json.loads(p.read_text())
    H = json.loads((ROOT / "outputs" / "health_results.json").read_text())
    # the boosted model in the note is the same model as in the report
    assert R["accuracy_2018"]["Gradient boosting"]["mae_kw"] == pytest.approx(H["accuracy"]["mae_kw_model"], abs=0.01)
    assert R["accuracy_2018"]["Power curve"]["mae_kw"] == pytest.approx(H["accuracy"]["mae_kw_power_curve"], abs=0.01)
    lc = {r["training"]: r for r in R["learning_curve"]}
    assert lc["all 12 months"]["mae_kw"] == pytest.approx(H["accuracy"]["mae_kw_model"], abs=0.01)
