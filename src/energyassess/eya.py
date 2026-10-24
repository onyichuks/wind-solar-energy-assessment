"""Energy yield assessment: loss chain, uncertainty and exceedance (P) values.

Loss categories and uncertainty categories follow the structure of IEC 61400-15 style
pre-construction reports. Values are the assumptions a consultant would make before
construction, stated openly so the gap analysis can test each one against operation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# Pre-construction loss assumptions, as fractions of energy (wakes come from the model)
DEFAULT_LOSSES = {
    "Wake effect, external (neighbouring wind farms)": 0.005,
    "Turbine availability": 0.030,
    "Balance of plant availability": 0.005,
    "Grid availability": 0.005,
    "Electrical losses to the revenue meter": 0.020,
    "Turbine performance (sub-optimal operation, high wind hysteresis)": 0.012,
    "Environmental (icing, blade degradation)": 0.008,
    "Curtailment (none known before construction)": 0.000,
}

LOSS_GROUP = {
    "Wake effect, internal (modelled)": "Wakes",
    "Wake effect, external (neighbouring wind farms)": "Wakes",
    "Turbine availability": "Availability",
    "Balance of plant availability": "Availability",
    "Grid availability": "Availability",
    "Electrical losses to the revenue meter": "Electrical",
    "Turbine performance (sub-optimal operation, high wind hysteresis)": "Turbine performance",
    "Environmental (icing, blade degradation)": "Environmental",
    "Curtailment (none known before construction)": "Curtailment",
}


def loss_chain(gross_mwh: float, internal_wake: float, losses: dict | None = None) -> pd.DataFrame:
    """Apply losses multiplicatively, as efficiencies, from gross to net energy."""
    losses = dict(DEFAULT_LOSSES if losses is None else losses)
    items = {"Wake effect, internal (modelled)": internal_wake, **losses}
    rows, energy = [], gross_mwh
    for name, loss in items.items():
        lost = energy * loss
        energy -= lost
        rows.append({"loss": name, "group": LOSS_GROUP.get(name, "Other"), "loss_pct": 100 * loss,
                     "efficiency_pct": 100 * (1 - loss), "lost_mwh": lost, "energy_after_mwh": energy})
    out = pd.DataFrame(rows)
    out.attrs.update({"gross_mwh": gross_mwh, "net_mwh": energy,
                      "total_efficiency": energy / gross_mwh})
    return out


@dataclass
class Uncertainty:
    """Uncertainty components as one standard deviation, in percent.

    Wind speed components are converted to energy with the sensitivity ratio
    (percent change in energy per percent change in wind speed).
    """
    wind_speed: dict = field(default_factory=dict)
    energy: dict = field(default_factory=dict)
    interannual_speed_pct: float = 3.9
    climate_energy_pct: float = 1.0


def sensitivity_ratio(energy_fn, ws_scale_step: float = 0.05) -> float:
    """Percent energy change per percent wind speed change, by central difference."""
    up, down = energy_fn(1 + ws_scale_step), energy_fn(1 - ws_scale_step)
    mid = energy_fn(1.0)
    return ((up - down) / mid) / (2 * ws_scale_step)


def combine(unc: Uncertainty, sens: float) -> pd.DataFrame:
    """Uncertainty table in energy terms, with the root-sum-square totals."""
    rows = []
    for name, v in unc.wind_speed.items():
        rows.append({"component": name, "basis": "wind speed", "speed_pct": v, "energy_pct": v * sens})
    for name, v in unc.energy.items():
        rows.append({"component": name, "basis": "energy", "speed_pct": np.nan, "energy_pct": v})
    out = pd.DataFrame(rows)
    hist = float(np.sqrt((out["energy_pct"] ** 2).sum()))
    out.attrs.update({"historical_pct": hist, "iav_energy_pct": unc.interannual_speed_pct * sens,
                      "climate_pct": unc.climate_energy_pct, "sensitivity": sens})
    return out


def p_values(p50: float, table: pd.DataFrame, years=(1, 10, 20)) -> pd.DataFrame:
    """P50, P75, P90 and P99 for several averaging periods, normal distribution."""
    hist = table.attrs["historical_pct"]
    iav = table.attrs["iav_energy_pct"]
    clim = table.attrs["climate_pct"]
    rows = []
    for n in years:
        total = np.sqrt(hist ** 2 + clim ** 2 + iav ** 2 / n)
        row = {"period_years": n, "total_uncertainty_pct": total}
        for p in (50, 75, 90, 99):
            z = stats.norm.ppf(p / 100.0)
            row[f"P{p}_mwh"] = p50 * (1 - z * total / 100.0)
        rows.append(row)
    return pd.DataFrame(rows)
