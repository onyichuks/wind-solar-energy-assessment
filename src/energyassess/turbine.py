"""Turbine models: reference power curve, air density adjustment and thrust coefficient."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STATIC = Path(__file__).resolve().parents[2] / "static_data"
RHO_STD = 1.225


def load_power_curve(path=STATIC / "mm82_power_curve.csv") -> pd.Series:
    """Reference power curve (kW) at standard air density, indexed by wind speed (m/s)."""
    pc = pd.read_csv(path)
    return pc.set_index("wind_speed_ms")["power_kw"]


def density_adjusted(curve: pd.Series, rho: float, rho_ref: float = RHO_STD) -> pd.Series:
    """Power curve at another air density, IEC 61400-12-1 method for pitch-regulated turbines.

    Power at site wind speed V equals reference power at V * (rho / rho_ref)^(1/3), so the
    curve is shifted along the speed axis rather than scaled.
    """
    v = curve.index.to_numpy(dtype=float)
    p = curve.to_numpy(dtype=float)
    v_std = v * (rho / rho_ref) ** (1.0 / 3.0)
    # cut-out stays at the same site wind speed: interpolate on the operating curve held at
    # its last value above cut-out, so denser air does not move cut-out down a bin
    cut_out = v[p > 0].max()
    p_op = np.where(v > cut_out, p[v <= cut_out][-1], p)
    adj = np.interp(v_std, v, p_op, left=0.0, right=p_op[-1])
    adj[v > cut_out] = 0.0
    return pd.Series(adj, index=curve.index, name=f"power_kw_rho{rho:.3f}")


def power_from_curve(curve: pd.Series, ws) -> np.ndarray:
    """Interpolated power (kW) for each wind speed; zero below cut-in and above cut-out."""
    v = curve.index.to_numpy(dtype=float)
    p = curve.to_numpy(dtype=float)
    out = np.interp(np.asarray(ws, dtype=float), v, p, left=0.0, right=0.0)
    out[np.asarray(ws) > v[p > 0].max()] = 0.0
    return out


def thrust_curve(ws, rated_kw: float, diameter: float, hub_height: float) -> np.ndarray:
    """Thrust coefficient from PyWake's generic turbine of the same size.

    No manufacturer thrust curve is public for the MM82, so the generic model (constant
    thrust coefficient of 0.8 below rated, falling above rated as the blades pitch) is used.
    This is the usual fallback in early-stage assessments and is covered in the wake
    model uncertainty.
    """
    from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine

    gwt = GenericWindTurbine("generic", diameter, hub_height, rated_kw)
    return gwt.ct(np.asarray(ws, dtype=float))


def capacity_factor(energy_mwh: float, capacity_mw: float, hours: float = 8766.0) -> float:
    return energy_mwh / (capacity_mw * hours)
