"""Wind resource: reanalysis loading, air density, shear, measure-correlate-predict and wind statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

R_DRY = 287.05  # J/(kg K)
G = 9.80665  # m/s2
LAPSE = 0.0065  # K/m, standard atmosphere


def load_era5_openmeteo(path) -> pd.DataFrame:
    """Read an Open-Meteo ERA5 archive CSV (hourly, UTC) into a tidy frame.

    Columns returned: ws10, ws100 (m/s), wd100 (degrees), t2m (C), sp (hPa).
    The first two lines of the file hold the grid point; they are stored in attrs.
    """
    meta = pd.read_csv(path, nrows=1)
    df = pd.read_csv(path, skiprows=3)
    df.columns = ["time", "ws10", "ws100", "wd100", "t2m", "sp"]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").astype(float)
    df.attrs.update({"lat": float(meta.loc[0, "latitude"]), "lon": float(meta.loc[0, "longitude"]),
                     "elevation_m": float(meta.loc[0, "elevation"])})
    return df


def air_density(t2m_c, sp_hpa, z_ground_m: float, z_target_m: float):
    """Air density at a target altitude from 2 m temperature and surface pressure.

    Temperature follows the standard lapse rate and pressure the hypsometric equation,
    the usual way of moving reanalysis density to hub height at the site elevation.
    """
    t_ground = np.asarray(t2m_c) + 273.15
    dz = z_target_m - (z_ground_m + 2.0)
    t_target = t_ground - LAPSE * dz
    t_mean = 0.5 * (t_ground + t_target)
    p_target = np.asarray(sp_hpa) * 100.0 * np.exp(-G * (z_target_m - z_ground_m) / (R_DRY * t_mean))
    return p_target / (R_DRY * t_target)


def shear_exponent(ws_low, ws_high, z_low: float, z_high: float, min_ws: float = 2.0):
    """Power-law shear exponent per record; NaN where the lower speed is too low to be meaningful."""
    ws_low = np.asarray(ws_low, dtype=float)
    ws_high = np.asarray(ws_high, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.log(ws_high / ws_low) / np.log(z_high / z_low)
    alpha[(ws_low < min_ws) | ~np.isfinite(alpha)] = np.nan
    return np.clip(alpha, -0.2, 0.7)


def extrapolate_to_height(era5: pd.DataFrame, z_target: float) -> pd.Series:
    """ERA5 100 m wind moved to a target height with hourly shear from the 10 m and 100 m levels.

    Hours with too little wind for a stable exponent take the median exponent for that
    calendar month and hour of day.
    """
    alpha = pd.Series(shear_exponent(era5["ws10"], era5["ws100"], 10.0, 100.0), index=era5.index)
    clim = alpha.groupby([alpha.index.month, alpha.index.hour]).transform("median")
    alpha = alpha.fillna(clim)
    return era5["ws100"] * (z_target / 100.0) ** alpha


def sector_index(direction_deg, n_sectors: int = 12):
    """Sector number (0 = north-centred) for each direction."""
    width = 360.0 / n_sectors
    return (np.floor(((np.asarray(direction_deg) + width / 2.0) % 360.0) / width)).astype(int)


def mcp_fit(ref_ws, ref_dir, target_ws, n_sectors: int = 12, method: str = "variance_ratio",
            min_points: int = 30) -> pd.DataFrame:
    """Fit sector-wise measure-correlate-predict relationships: target = intercept + slope * reference.

    method "variance_ratio" keeps the variance of the target; method "linear" is ordinary
    least squares, which narrows the spread of predicted speeds. Both are used here for the
    cross-validation comparison and for the monthly energy index (where the level cancels).
    """
    ref_ws = np.asarray(ref_ws, dtype=float)
    target_ws = np.asarray(target_ws, dtype=float)
    sec = sector_index(ref_dir, n_sectors)
    ok = np.isfinite(ref_ws) & np.isfinite(target_ws)
    rows = []
    for s in range(n_sectors):
        m = ok & (sec == s)
        x, y = ref_ws[m], target_ws[m]
        if m.sum() < min_points:
            rows.append({"sector": s, "n": int(m.sum()), "slope": np.nan, "intercept": np.nan, "r2": np.nan})
            continue
        r = np.corrcoef(x, y)[0, 1]
        if method == "variance_ratio":
            slope = y.std(ddof=1) / x.std(ddof=1)
        elif method == "linear":
            slope = r * y.std(ddof=1) / x.std(ddof=1)
        else:
            raise ValueError(method)
        intercept = y.mean() - slope * x.mean()
        rows.append({"sector": s, "n": int(m.sum()), "slope": slope, "intercept": intercept, "r2": r * r,
                     "ref_mean": x.mean(), "target_mean": y.mean()})
    fit = pd.DataFrame(rows).set_index("sector")
    # Sectors with too few points borrow the all-direction fit
    if fit["slope"].isna().any():
        x, y = ref_ws[ok], target_ws[ok]
        r = np.corrcoef(x, y)[0, 1]
        slope = y.std(ddof=1) / x.std(ddof=1) if method == "variance_ratio" else r * y.std(ddof=1) / x.std(ddof=1)
        fit.loc[fit["slope"].isna(), ["slope", "intercept"]] = [slope, y.mean() - slope * x.mean()]
    fit.attrs["method"] = method
    return fit


def mcp_predict(fit: pd.DataFrame, ref_ws, ref_dir) -> np.ndarray:
    """Apply a sector-wise fit to a long-term reference series; speeds are kept non-negative."""
    sec = sector_index(ref_dir, len(fit))
    slope = fit["slope"].to_numpy()[sec]
    intercept = fit["intercept"].to_numpy()[sec]
    return np.maximum(intercept + slope * np.asarray(ref_ws, dtype=float), 0.0)


def mcp_scaled_distribution(site_ws, site_dir, ref_conc_ws, ref_conc_dir, ref_lt_ws, ref_lt_dir,
                            n_sectors: int = 12, min_points: int = 20) -> dict:
    """Long-term correction that keeps the shape of the measured site distribution.

    1. Fit sector-wise linear relationships between the site and the reference over the
       concurrent period.
    2. In each reference sector, scale the measured site speeds by the ratio of the
       predicted long-term mean to the predicted concurrent mean.
    3. Reweight each sector so its frequency matches the long-term reference.

    Returns the long-term-corrected site records (speed, direction, weight) and the
    sector table. Cross-validation between the two measured years (fit on one, predict the
    other) gave energy errors of -2.1 and -0.6 percent for this method, against +2.5 to +8.5
    percent for hourly linear and variance ratio series synthesised from ERA5 (see docs/REPORT.md).
    """
    site_ws = np.asarray(site_ws, dtype=float)
    fit = mcp_fit(ref_conc_ws, ref_conc_dir, site_ws, n_sectors, "linear")
    pred_conc = mcp_predict(fit, ref_conc_ws, ref_conc_dir)
    pred_lt = mcp_predict(fit, ref_lt_ws, ref_lt_dir)
    s_conc = sector_index(ref_conc_dir, n_sectors)
    s_lt = sector_index(ref_lt_dir, n_sectors)
    ws_lt = site_ws.copy()
    weight = np.ones_like(site_ws)
    rows = []
    for s in range(n_sectors):
        m = s_conc == s
        factor = pred_lt[s_lt == s].mean() / pred_conc[m].mean() if m.sum() >= min_points else pred_lt.mean() / pred_conc.mean()
        f_conc, f_lt = m.mean(), (s_lt == s).mean()
        ws_lt[m] *= factor
        if m.any():
            weight[m] = f_lt / f_conc
        rows.append({"sector": s, "centre_deg": s * 360.0 / n_sectors, "n_concurrent_hours": int(m.sum()),
                     "slope": fit.loc[s, "slope"], "intercept": fit.loc[s, "intercept"], "r2": fit.loc[s, "r2"],
                     "speed_factor": factor, "freq_concurrent": f_conc, "freq_long_term": f_lt})
    table = pd.DataFrame(rows).set_index("sector")
    return {"ws": ws_lt, "wd": np.asarray(site_dir, dtype=float), "weight": weight, "table": table,
            "mean_concurrent": float(site_ws.mean()), "mean_long_term": float(np.average(ws_lt, weights=weight))}


# ---------------------------------------------------------------- machine learning long-term correction
ML_FEATURES = ["ws100", "ws10", "dir_sin", "dir_cos", "shear", "t2m", "sp", "hour_sin", "hour_cos",
               "doy_sin", "doy_cos", "ws100_prev", "ws100_next"]


def ml_features(era5: pd.DataFrame) -> pd.DataFrame:
    """Hourly ERA5 inputs for the machine learning long-term correction.

    100 m and 10 m wind, direction (sine and cosine), the 10 to 100 m shear exponent and air
    temperature (both carry atmospheric stability), surface pressure, time of day and season,
    and the 100 m wind an hour before and after (ERA5 timing is not exact). Missing values at
    the ends of the record are left as NaN, which the gradient-boosted trees handle.
    """
    rad = np.radians(era5["wd100"])
    t = era5.index
    f = pd.DataFrame({"ws100": era5["ws100"], "ws10": era5["ws10"], "dir_sin": np.sin(rad), "dir_cos": np.cos(rad),
                      "shear": np.log(era5["ws100"].clip(lower=0.5) / era5["ws10"].clip(lower=0.5)) / np.log(10.0),
                      "t2m": era5["t2m"], "sp": era5["sp"],
                      "hour_sin": np.sin(2 * np.pi * t.hour / 24), "hour_cos": np.cos(2 * np.pi * t.hour / 24),
                      "doy_sin": np.sin(2 * np.pi * t.dayofyear / 365.25), "doy_cos": np.cos(2 * np.pi * t.dayofyear / 365.25)},
                     index=era5.index)
    f["ws100_prev"] = era5["ws100"].shift(1)
    f["ws100_next"] = era5["ws100"].shift(-1)
    return f[ML_FEATURES]


def mcp_ml_fit(ref_features: pd.DataFrame, site_ws, seed: int = 0):
    """Measure-correlate-predict with gradient-boosted trees (scikit-learn): hourly site wind from ERA5 inputs.

    Settings fixed in advance (the same family and settings as the turbine health model, with
    fewer rounds allowed); early stopping on a random 10 percent of the concurrent hours.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=seed)
    return model.fit(ref_features, np.asarray(site_ws, dtype=float))


def mcp_ml_predict(model, ref_features: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(ref_features[model.feature_names_in_]), 0.0, None)


def quantile_mapper(predicted, observed, n: int = 201):
    """Map predicted speeds onto the observed distribution (a spread correction for regression output).

    Built from out-of-fold predictions and the measured speeds of the fit period; beyond the
    end quantiles the mapping continues with slope one.
    """
    q = np.linspace(0.0, 1.0, n)
    xp, fp = np.quantile(np.asarray(predicted, float), q), np.quantile(np.asarray(observed, float), q)

    def apply(x):
        x = np.asarray(x, dtype=float)
        y = np.interp(x, xp, fp)
        lo, hi = x < xp[0], x > xp[-1]
        y[lo] = fp[0] + (x[lo] - xp[0])
        y[hi] = fp[-1] + (x[hi] - xp[-1])
        return np.clip(y, 0.0, None)
    return apply


def weighted_joint_frequency(ws, wd, weight, ws_edges, wd_step: float = 5.0):
    """Joint frequency table from weighted records (fractions summing to one)."""
    ws = np.asarray(ws, dtype=float)
    wd = np.asarray(wd, dtype=float) % 360.0
    w = np.asarray(weight, dtype=float)
    ok = np.isfinite(ws) & np.isfinite(wd)
    wd_centres = np.arange(0.0, 360.0, wd_step)
    wd_bin = (np.floor(((wd[ok] + wd_step / 2.0) % 360.0) / wd_step)).astype(int)
    ws_bin = np.clip(np.digitize(ws[ok], ws_edges) - 1, 0, len(ws_edges) - 2)
    freq = np.zeros((len(wd_centres), len(ws_edges) - 1))
    np.add.at(freq, (wd_bin, ws_bin), w[ok])
    freq /= freq.sum()
    return wd_centres, 0.5 * (ws_edges[:-1] + ws_edges[1:]), freq


def weibull_moments(v, w=None) -> tuple[float, float]:
    """Weibull A and k from the (weighted) mean and standard deviation (Justus approximation)."""
    from math import gamma

    v = np.asarray(v, dtype=float)
    w = np.ones_like(v) if w is None else np.asarray(w, dtype=float)
    mu = np.average(v, weights=w)
    sd = np.sqrt(np.average((v - mu) ** 2, weights=w))
    k = (sd / mu) ** -1.086
    return mu / gamma(1 + 1 / k), k


def weibull_by_sector(ws, wd, weight=None, n_sectors: int = 12) -> pd.DataFrame:
    """Sector frequency, mean speed and Weibull A and k (weighted records allowed)."""
    ws = np.asarray(ws, dtype=float)
    w = np.ones_like(ws) if weight is None else np.asarray(weight, dtype=float)
    ok = np.isfinite(ws) & np.isfinite(np.asarray(wd, dtype=float))
    sec = sector_index(np.nan_to_num(np.asarray(wd, dtype=float)), n_sectors)
    rows = []
    for s in range(n_sectors):
        m = ok & (sec == s)
        a, k = weibull_moments(ws[m], w[m])
        rows.append({"sector": s, "centre_deg": s * 360.0 / n_sectors, "frequency": w[m].sum() / w[ok].sum(),
                     "mean_ms": np.average(ws[m], weights=w[m]), "weibull_A": a, "weibull_k": k})
    a, k = weibull_moments(ws[ok], w[ok])
    out = pd.DataFrame(rows).set_index("sector")
    out.attrs.update({"all_A": a, "all_k": k, "all_mean": float(np.average(ws[ok], weights=w[ok]))})
    return out


def joint_frequency(ws, wd, ws_edges, wd_step: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint direction and speed frequency table (fractions summing to one).

    Returns (wd_centres, ws_centres, freq[wd, ws]).
    """
    ws = np.asarray(ws, dtype=float)
    wd = np.asarray(wd, dtype=float) % 360.0
    ok = np.isfinite(ws) & np.isfinite(wd)
    wd_centres = np.arange(0.0, 360.0, wd_step)
    wd_bin = (np.floor(((wd[ok] + wd_step / 2.0) % 360.0) / wd_step)).astype(int)
    ws_bin = np.digitize(ws[ok], ws_edges) - 1
    keep = (ws_bin >= 0) & (ws_bin < len(ws_edges) - 1)
    freq = np.zeros((len(wd_centres), len(ws_edges) - 1))
    np.add.at(freq, (wd_bin[keep], ws_bin[keep]), 1.0)
    freq /= ok.sum()
    ws_centres = 0.5 * (ws_edges[:-1] + ws_edges[1:])
    return wd_centres, ws_centres, freq
