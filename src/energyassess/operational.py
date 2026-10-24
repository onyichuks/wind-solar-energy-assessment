"""Operational analysis: data quality, undisturbed sectors, virtual met mast, losses and measured wakes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import turbine as tb
from . import wakes as wk
from .scada import flat_runs

RATED_KW = 2050.0
DIAMETER = 82.0


# ---------------------------------------------------------------- data quality

def apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Flag missing, out-of-range and frozen intervals.

    Frozen means an hour or more (six 10-minute values) of identical wind speed or wind
    direction readings, for example a stuck wind vane. Expects one row per turbine and
    10-minute step, sorted by turbine and time, as written by scripts/prepare_data.py.
    """
    out = df.sort_values(["turbine", "time"]).reset_index(drop=True)
    out["qc_missing"] = out[["power_kw", "wind_ms", "wind_dir_deg"]].isna().any(axis=1)
    out["qc_range"] = (out["power_kw"] < -50) | (out["power_kw"] > 1.1 * RATED_KW) | (out["wind_ms"] < 0) | (out["wind_ms"] > 40)
    frozen = np.zeros(len(out), dtype=bool)
    for col in ("wind_ms", "wind_dir_deg"):
        frozen |= out.groupby("turbine")[col].transform(lambda s: flat_runs(s, 6)).to_numpy(dtype=bool)
    out["qc_frozen"] = frozen
    out["qc_valid"] = ~(out["qc_missing"] | out["qc_range"] | out["qc_frozen"])
    return out


def qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("turbine")
    return pd.DataFrame({"intervals": g.size(), "missing_pct": 100 * g["qc_missing"].mean(),
                         "out_of_range_pct": 100 * g["qc_range"].mean(), "frozen_pct": 100 * g["qc_frozen"].mean(),
                         "valid_pct": 100 * g["qc_valid"].mean()}).round(2)


# ---------------------------------------------------------------- undisturbed sectors (IEC 61400-12-1 Annex A)

def disturbed_sector_width(distance_m: float, diameter: float = DIAMETER) -> float:
    """Width (degrees) of the sector disturbed by a neighbouring turbine at the given distance."""
    ld = distance_m / diameter
    return float(np.degrees(1.3 * np.arctan(2.5 / ld + 0.15)) + 10.0)


def undisturbed_table(turbines: pd.DataFrame, step: float = 1.0, max_distance_d: float = 20.0) -> pd.DataFrame:
    """Boolean table [direction, turbine]: True where no neighbour is upwind within the IEC sector."""
    x, y = wk.layout_xy(turbines)
    names = turbines["turbine"].tolist()
    dirs = np.arange(0.0, 360.0, step)
    free = pd.DataFrame(True, index=dirs, columns=names)
    for i, ti in enumerate(names):
        for j, tj in enumerate(names):
            if i == j:
                continue
            dx, dy = x[j] - x[i], y[j] - y[i]
            dist = np.hypot(dx, dy)
            if dist > max_distance_d * DIAMETER:
                continue
            bearing = np.degrees(np.arctan2(dx, dy)) % 360.0  # wind from here reaches j first
            half = disturbed_sector_width(dist) / 2.0
            diff = np.abs((dirs - bearing + 180.0) % 360.0 - 180.0)
            free.loc[diff <= half, ti] = False
    return free


def circular_mean_deg(a, axis=None):
    r = np.radians(a)
    return np.degrees(np.arctan2(np.nanmean(np.sin(r), axis=axis), np.nanmean(np.cos(r), axis=axis))) % 360.0


# ---------------------------------------------------------------- virtual met mast

def virtual_mast(df: pd.DataFrame, free: pd.DataFrame) -> pd.DataFrame:
    """Free-stream hub-height wind from the nacelle anemometers of undisturbed turbines.

    For each 10-minute interval the wind farm direction is the circular mean of the
    turbines' absolute wind direction. The mast speed is the mean nacelle wind speed of
    the turbines that are undisturbed for that direction, are not stopped for a fault,
    and pass the data checks. Stands in for an on-site met mast, which the open data lack.
    """
    ok = df["qc_valid"] & (df["lost_downtime_kwh"].fillna(0) <= 0)
    ws = df.pivot(index="time", columns="turbine", values="wind_ms").where(
        ok.groupby([df["time"], df["turbine"]]).first().unstack())
    wd = df.pivot(index="time", columns="turbine", values="wind_dir_deg").where(ws.notna())
    farm_dir = pd.Series(circular_mean_deg(wd.to_numpy(), axis=1), index=wd.index)
    d_idx = np.round(farm_dir.fillna(0).to_numpy()).astype(int) % 360
    mask = free.reindex(columns=ws.columns).to_numpy()[d_idx]
    sel = ws.to_numpy().copy()
    sel[~mask] = np.nan
    out = pd.DataFrame({"mast_ws": np.nanmean(sel, axis=1), "mast_dir": farm_dir.to_numpy(),
                        "n_turbines": np.isfinite(sel).sum(axis=1)}, index=ws.index)
    out.loc[out["n_turbines"] == 0, "mast_ws"] = np.nan
    out.loc[farm_dir.isna(), ["mast_ws", "mast_dir"]] = np.nan
    return out


def free_stream_turbulence(df: pd.DataFrame, mast: pd.DataFrame, free: pd.DataFrame, ws_range=(9.5, 10.5)) -> float:
    """Mean turbulence intensity seen by undisturbed turbines near 10 m/s (nacelle anemometry, IEC reference speed)."""
    d_idx = np.round(mast["mast_dir"].reindex(df["time"]).fillna(0).to_numpy()).astype(int) % 360
    col = free.columns.get_indexer(df["turbine"])
    is_free = free.to_numpy()[d_idx, col]
    sel = df["qc_valid"].to_numpy() & is_free & df["wind_ms"].between(*ws_range).to_numpy()
    return float((df["wind_std_ms"] / df["wind_ms"])[sel].mean())


def to_hourly(mast: pd.DataFrame, min_count: int = 4) -> pd.DataFrame:
    """Hourly means centred on the hour, to match ERA5 values, which are instantaneous at the hour.

    SCADA rows are stamped at the start of each 10-minute interval, so the hour labelled
    00:00 averages the six intervals from 23:30 to 00:30.
    """
    centred = mast.shift(freq="30min")
    g = centred.resample("1h")
    out = pd.DataFrame({"mast_ws": g["mast_ws"].mean(), "count": g["mast_ws"].count()})
    out["mast_dir"] = centred["mast_dir"].resample("1h").apply(lambda a: circular_mean_deg(a.to_numpy()) if a.notna().any() else np.nan)
    out.loc[out["count"] < min_count, ["mast_ws", "mast_dir"]] = np.nan
    return out


# ---------------------------------------------------------------- losses

def turbine_losses(df: pd.DataFrame) -> pd.DataFrame:
    """Energy, operator-reported downtime and performance losses, and energy-based availability."""
    g = df.groupby("turbine")
    out = pd.DataFrame({
        "energy_mwh": g["energy_kwh"].sum() / 1000,
        "lost_downtime_mwh": g["lost_downtime_kwh"].sum() / 1000,
        "lost_performance_mwh": g["lost_performance_kwh"].sum() / 1000,
        "mean_wind_ms": g["wind_ms"].mean(),
    })
    out["availability_pct"] = 100 * out["energy_mwh"] / (out["energy_mwh"] + out["lost_downtime_mwh"])
    return out


def own_downtime_loss(df: pd.DataFrame, curves: pd.DataFrame) -> pd.Series:
    """Independent downtime loss (kWh): turbine not producing while its own power curve says it should.

    Used to check the operator's Greenbyte figures, which come from the turbine's default curve.
    """
    exp = np.zeros(len(df))
    for t, idx in df.groupby("turbine").groups.items():
        c = curves[t].dropna()
        exp[df.index.get_indexer(idx)] = np.interp(df.loc[idx, "wind_ms"], c.index, c.to_numpy(), left=0, right=0)
    down = df["qc_valid"] & (df["power_kw"] <= 10) & (df["wind_ms"].between(4.0, 25.0))
    return pd.Series(np.where(down, exp / 6.0, 0.0), index=df.index)


def measured_power_curves(df: pd.DataFrame, bins=np.arange(0.25, 25.5, 0.5), min_count: int = 3) -> pd.DataFrame:
    """Power curves (kW) by the method of bins: mean power of all normal-operation intervals
    in each 0.5 m/s bin of nacelle wind speed (IEC 61400-12-1 style, with nacelle anemometry).

    Normal operation means valid data and no downtime recorded; intervals with zero output at
    low wind stay in, as they belong to the curve. Bins with fewer than `min_count` intervals
    are left empty.
    """
    normal = df["qc_valid"] & (df["lost_downtime_kwh"].fillna(0) <= 0)
    d = df.loc[normal, ["turbine", "wind_ms", "power_kw"]].copy()
    d["bin"] = pd.cut(d["wind_ms"], bins, labels=0.5 * (bins[:-1] + bins[1:]))
    g = d.groupby(["bin", "turbine"], observed=True)["power_kw"]
    pc = g.mean().where(g.size() >= min_count).unstack()
    pc.index = pc.index.astype(float)
    return pc


def electrical_loss(df: pd.DataFrame, plant: pd.DataFrame) -> dict:
    """Loss between the summed turbine meters and the revenue meter, over intervals where all report."""
    e = df.pivot(index="time", columns="turbine", values="energy_kwh")
    full = e.notna().all(axis=1)
    turb = e[full].sum(axis=1)
    grid = plant["grid_kwh"].reindex(turb.index)
    both = grid.notna()
    t_sum, g_sum = turb[both].sum(), grid[both].sum()
    return {"intervals": int(both.sum()), "turbine_mwh": t_sum / 1000, "meter_mwh": g_sum / 1000,
            "loss_pct": 100 * (1 - g_sum / t_sum)}


def meter_day_check(df: pd.DataFrame, plant: pd.DataFrame, threshold: float = 0.10, min_mwh: float = 20.0) -> pd.DataFrame:
    """Daily revenue meter energy against the summed turbine meters, to find corrupt meter days.

    Uses intervals where every turbine and the grid meter report. Days with at least
    `min_mwh` of turbine energy whose meter-to-turbine ratio is more than `threshold` away
    from the median ratio are flagged. The 10-minute values of both meters are counter
    differences that can spill into neighbouring intervals, so only daily totals are compared.
    """
    e = df.pivot(index="time", columns="turbine", values="energy_kwh")
    turb = e[e.notna().all(axis=1)].sum(axis=1)
    grid = plant["grid_kwh"].reindex(turb.index)
    both = grid.notna()
    day = pd.DataFrame({"turbine_mwh": turb[both], "meter_mwh": grid[both]}).resample("D").sum() / 1000
    day["intervals"] = both[both].resample("D").size()
    day = day[day["turbine_mwh"] >= min_mwh].copy()
    day["ratio"] = day["meter_mwh"] / day["turbine_mwh"]
    median = float(day["ratio"].median())
    day["flag"] = (day["ratio"] / median - 1).abs() > threshold
    day.attrs["median_ratio"] = median
    return day


# ---------------------------------------------------------------- terrain (site flow) factors

def _pivot_ok(df: pd.DataFrame, value: str) -> pd.DataFrame:
    ok = df["qc_valid"] & (df["lost_downtime_kwh"].fillna(0) <= 0)
    return df.pivot(index="time", columns="turbine", values=value).where(
        ok.groupby([df["time"], df["turbine"]]).first().unstack())


def terrain_factors(df: pd.DataFrame, mast: pd.DataFrame, free: pd.DataFrame, ws_range=(4.0, 15.0)) -> pd.DataFrame:
    """Relative free-stream wind speed at each turbine position, measured from SCADA.

    For every interval in which a turbine is undisturbed, its nacelle wind speed is divided by
    the mean of the other undisturbed turbines. The median ratio, normalised to a farm mean of
    one, is the turbine's speed-up factor. A pre-construction flow model (for example WAsP or
    CFD) would predict these; the uniform-flow assumption used in the estimate does not.
    """
    ws = _pivot_ok(df, "wind_ms")
    m = mast.reindex(ws.index)
    d_idx = np.round(m["mast_dir"].fillna(0).to_numpy()).astype(int) % 360
    fmask = free.reindex(columns=ws.columns).to_numpy()[d_idx]
    w = ws.to_numpy()
    in_range = m["mast_ws"].between(*ws_range).to_numpy()
    rows = []
    for i, name in enumerate(ws.columns):
        others = np.where(fmask & np.isfinite(w), w, np.nan)
        others[:, i] = np.nan
        with np.errstate(all="ignore"):
            om = np.nanmean(others, axis=1)
        sel = fmask[:, i] & np.isfinite(w[:, i]) & np.isfinite(om) & in_range
        rows.append({"turbine": name, "intervals": int(sel.sum()), "ratio": float(np.median(w[sel, i] / om[sel]))})
    out = pd.DataFrame(rows).set_index("turbine")
    out["speed_factor"] = out["ratio"] / out["ratio"].mean()
    return out


def terrain_adjusted_efficiency(df: pd.DataFrame, mast: pd.DataFrame, free: pd.DataFrame, factors: pd.Series,
                                fleet_curve: pd.Series, ws_range=(5.0, 11.0), sector_width: float = 30.0) -> pd.DataFrame:
    """Wake efficiency by sector with the terrain effect removed.

    Expected unwaked power at each turbine comes from the virtual mast speed, moved to the
    turbine position with the speed-up factors, through the fleet's measured nacelle power
    curve. Efficiency is actual over expected, for intervals with every turbine in normal
    operation. A waked turbine with no output is kept: deep wakes are part of the loss.
    """
    p = _pivot_ok(df, "power_kw").clip(lower=0)  # standby consumption is not a wake effect
    full = p.notna().all(axis=1)
    m = mast.reindex(p.index)
    sel = full & m["mast_ws"].between(*ws_range)
    p, m = p[sel], m[sel]
    s = factors.reindex(p.columns).to_numpy()
    d_idx = np.round(m["mast_dir"].to_numpy()).astype(int) % 360
    fmask = free.reindex(columns=p.columns).to_numpy()[d_idx]
    s_free = np.array([s[row].mean() for row in fmask])
    u = m["mast_ws"].to_numpy()[:, None] * s[None, :] / s_free[:, None]
    c = fleet_curve.dropna()
    expected = np.interp(u, c.index.to_numpy(dtype=float), c.to_numpy(dtype=float))
    sec = (np.floor(((m["mast_dir"].to_numpy() + sector_width / 2) % 360) / sector_width)).astype(int)
    res = pd.DataFrame({"sector": sec, "actual": p.to_numpy().sum(axis=1), "expected": expected.sum(axis=1)})
    out = res.groupby("sector").agg(actual=("actual", "sum"), expected=("expected", "sum"), n=("actual", "size"))
    out["efficiency"] = out["actual"] / out["expected"]
    out["centre_deg"] = out.index * sector_width
    return out.reset_index()


def terrain_energy_ratio(curve: pd.Series, freq: np.ndarray, wd, ws, factors: pd.Series, free: pd.DataFrame, names) -> pd.Series:
    """Per-turbine gross energy with measured speed-up factors over gross energy with uniform flow."""
    s = factors.reindex(names).to_numpy()
    fmask = free.reindex(columns=names).to_numpy()[np.round(np.asarray(wd)).astype(int) % 360]
    s_free = np.array([s[row].mean() for row in fmask])  # the virtual mast sees the undisturbed turbines
    base = tb.power_from_curve(curve, ws)
    uniform = (freq * base[None, :]).sum()
    out = []
    for i in range(len(names)):
        u = np.asarray(ws)[None, :] * (s[i] / s_free)[:, None]
        out.append((freq * tb.power_from_curve(curve, u.ravel()).reshape(u.shape)).sum() / uniform)
    return pd.Series(out, index=names)


# ---------------------------------------------------------------- measured wake efficiency

def measured_sector_efficiency(df: pd.DataFrame, mast: pd.DataFrame, free: pd.DataFrame,
                               ws_range=(5.0, 11.0), sector_width: float = 30.0) -> pd.DataFrame:
    """Wind farm efficiency by sector from SCADA: mean turbine power over mean undisturbed-turbine power.

    Only intervals with all turbines in normal operation, and a free-stream speed in the
    range where wakes matter most (below rated), are used.
    """
    p = _pivot_ok(df, "power_kw").clip(lower=0)
    full = p.notna().all(axis=1)
    m = mast.reindex(p.index)
    sel = full & m["mast_ws"].between(*ws_range)
    p, m = p[sel], m[sel]
    d_idx = np.round(m["mast_dir"].to_numpy()).astype(int) % 360
    fmask = free.reindex(columns=p.columns).to_numpy()[d_idx]
    pv = p.to_numpy()
    free_mean = np.nanmean(np.where(fmask, pv, np.nan), axis=1)
    farm_mean = pv.mean(axis=1)
    sec = (np.floor(((m["mast_dir"].to_numpy() + sector_width / 2) % 360) / sector_width)).astype(int)
    res = pd.DataFrame({"sector": sec, "farm": farm_mean, "free": free_mean}).dropna()
    out = res.groupby("sector").agg(farm=("farm", "sum"), free=("free", "sum"), n=("farm", "size"))
    out["efficiency"] = out["farm"] / out["free"]
    out["centre_deg"] = out.index * sector_width
    per_turbine = pd.DataFrame(pv / free_mean[:, None], columns=p.columns)
    per_turbine["sector"] = sec
    return out.reset_index(), per_turbine.groupby("sector").median()


def model_wake_efficiency(power_kw: np.ndarray, free_kw: np.ndarray, wd, ws, freq: np.ndarray,
                          ws_range=(5.0, 11.0), sector_width: float = 30.0) -> pd.DataFrame:
    """Modelled waked over unwaked energy by sector, in the same speed range as the measurement."""
    wd, ws = np.asarray(wd), np.asarray(ws)
    in_ws = (ws >= ws_range[0]) & (ws <= ws_range[1])
    sec = (np.floor(((wd + sector_width / 2) % 360) / sector_width)).astype(int)
    rows = []
    for s in range(int(360 / sector_width)):
        m = sec == s
        f = freq[np.ix_(m, in_ws)]
        waked = (power_kw[:, m][:, :, in_ws].sum(axis=0) * f).sum()
        unwaked = (free_kw[:, m][:, :, in_ws].sum(axis=0) * f).sum()
        rows.append({"sector": s, "centre_deg": s * sector_width, "efficiency": waked / unwaked})
    return pd.DataFrame(rows)


def modelled_sector_efficiency(power_kw: np.ndarray, wd, ws, freq: np.ndarray, free: pd.DataFrame, names,
                               ws_range=(5.0, 11.0), sector_width: float = 30.0) -> pd.DataFrame:
    """Same efficiency definition as measured_sector_efficiency, applied to PyWake output."""
    wd = np.asarray(wd)
    ws = np.asarray(ws)
    in_ws = (ws >= ws_range[0]) & (ws <= ws_range[1])
    fmask = free.reindex(columns=names).to_numpy()[np.round(wd).astype(int) % 360]  # [wd, turbine]
    sec = (np.floor(((wd + sector_width / 2) % 360) / sector_width)).astype(int)
    rows = []
    for s in range(int(360 / sector_width)):
        num = den = 0.0
        for k in np.where(sec == s)[0]:
            f = freq[k, in_ws]
            pw = power_kw[:, k, in_ws]  # [turbine, ws]
            fm = fmask[k]
            if not fm.any():
                continue
            num += (pw.mean(axis=0) * f).sum()
            den += (pw[fm].mean(axis=0) * f).sum()
        rows.append({"sector": s, "centre_deg": s * sector_width, "efficiency": num / den if den else np.nan})
    return pd.DataFrame(rows)
