"""Wake modelling with DTU's PyWake on the real turbine layout."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import turbine as tb


def layout_xy(turbines: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Local east and north coordinates (m) around the layout centroid.

    An equirectangular projection is accurate to well under a metre across a site a few
    kilometres wide, which is far below the resolution that matters for wakes.
    """
    lat0 = np.radians(turbines["lat"].mean())
    lon0 = turbines["lon"].mean()
    x = (turbines["lon"] - lon0).to_numpy() * 111_320.0 * np.cos(lat0)
    y = (turbines["lat"] - turbines["lat"].mean()).to_numpy() * 110_574.0
    return x, y


def make_turbine(curve_kw: pd.Series, diameter: float, hub_height: float, rated_kw: float):
    """PyWake turbine from a power curve (kW) and the generic thrust curve."""
    from py_wake.wind_turbines import WindTurbine
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

    ws = curve_kw.index.to_numpy(dtype=float)
    p = curve_kw.to_numpy(dtype=float)
    ct = tb.thrust_curve(ws, rated_kw, diameter, hub_height)
    ct[p <= 0] = 0.0
    cut_out = float(ws[p > 0].max())  # without it PyWake would interpolate down to zero past cut-out
    return WindTurbine(name="MM82", diameter=diameter, hub_height=hub_height,
                       powerCtFunction=PowerCtTabular(ws, p, "kW", ct, ws_cutout=cut_out))


MODELS = {
    "Jensen Park (1983), k = 0.075, onshore setting": "jensen",
    "Bastankhah Gaussian (2014), default expansion": "bastankhah",
    "TurbOPark (Nygaard 2022), site turbulence": "turbopark",
    "Niayifar Gaussian (2016), site turbulence": "niayifar",
}


def wind_farm_model(name: str, wt, ti: float = 0.10):
    """Engineering wake models compared in the assessment (literature settings; ti feeds the turbulence-aware ones)."""
    from py_wake.site import UniformSite
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014, Niayifar_PorteAgel_2016
    from py_wake.literature.turbopark import Nygaard_2022
    from py_wake.literature.noj import Jensen_1983

    site = UniformSite(p_wd=[1.0], ti=ti)
    if name == "bastankhah":
        return Bastankhah_PorteAgel_2014(site, wt, k=0.0324555)
    if name == "turbopark":
        return Nygaard_2022(site, wt)
    if name == "jensen":
        return Jensen_1983(site, wt, k=0.075)
    if name == "niayifar":
        return Niayifar_PorteAgel_2016(site, wt)  # wake expansion grows with turbulence intensity
    raise ValueError(name)


def power_matrix(model, x, y, wd, ws) -> np.ndarray:
    """Power (kW) for every turbine, direction and speed: array [turbine, wd, ws]."""
    sim = model(x, y, wd=wd, ws=ws)
    return sim.Power.values / 1000.0


def free_power_matrix(curve_kw: pd.Series, n_turbines: int, wd, ws) -> np.ndarray:
    """Power with no wakes, same shape as power_matrix."""
    p = tb.power_from_curve(curve_kw, ws)
    return np.broadcast_to(p, (n_turbines, len(wd), len(ws))).copy()


def energy_mwh_per_year(power_kw: np.ndarray, freq: np.ndarray, hours: float = 8766.0) -> np.ndarray:
    """Annual energy per turbine (MWh) from power [turbine, wd, ws] and frequency [wd, ws]."""
    return (power_kw * freq[None, :, :]).sum(axis=(1, 2)) * hours / 1000.0


def sector_efficiency(power_kw: np.ndarray, free_kw: np.ndarray, freq: np.ndarray, wd, sector_width: float = 30.0):
    """Wind farm efficiency (waked over free energy) by direction sector."""
    wd = np.asarray(wd)
    sec = (np.floor(((wd + sector_width / 2.0) % 360.0) / sector_width)).astype(int)
    rows = []
    for s in range(int(360 / sector_width)):
        m = sec == s
        waked = (power_kw[:, m, :] * freq[None, m, :]).sum()
        free = (free_kw[:, m, :] * freq[None, m, :]).sum()
        rows.append({"centre_deg": s * sector_width, "efficiency": waked / free if free > 0 else np.nan,
                     "energy_share": free})
    out = pd.DataFrame(rows)
    out["energy_share"] /= out["energy_share"].sum()
    return out
