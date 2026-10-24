"""Solar PV performance: expected energy model, performance ratio, availability, snow and clipping."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_pvdaq_system50(power_path, weather_path) -> tuple[pd.Series, pd.DataFrame]:
    """AC power (kW, 15-minute) and NSRDB PSM3 satellite weather interpolated to the same steps."""
    p = pd.read_parquet(power_path).set_index("measured_on")["ac_power_2"]
    p.index = pd.to_datetime(p.index)
    p = (p / 1000.0).asfreq("15min")  # the PVDAQ stream is in watts
    w = pd.read_parquet(weather_path).set_index("index")
    w.index = pd.to_datetime(w.index)
    w = w[["ghi", "ghi_clear", "dni_clear", "dhi_clear", "temp_air"]]
    w = w.reindex(pd.date_range(w.index[0], w.index[-1], freq="15min")).interpolate(limit=2)
    w.index = w.index.tz_convert(p.index.tz)
    return p, w.reindex(p.index)


def qc_power(p: pd.Series, daytime: pd.Series) -> pd.DataFrame:
    """Data checks on the power stream with PVAnalytics: stale values, negative values and gaps."""
    from pvanalytics.quality import gaps

    stale = gaps.stale_values_round(p, window=3, decimals=2) & daytime
    out = pd.DataFrame({"missing": p.isna(), "negative": p < -0.01, "stale": stale.fillna(False)})
    out["valid"] = ~(out["missing"] | out["negative"] | out["stale"])
    return out


def detect_time_shifts(p: pd.Series, lat: float, lon: float) -> pd.Series:
    """Minutes of clock shift per day, from the midpoint of the measured production day against solar noon.

    PVDAQ streams sometimes switch between standard and daylight saving time; PVAnalytics
    finds the change points with the ruptures library.
    """
    import pvlib
    from pvanalytics.features import daytime
    from pvanalytics.quality.time import shifts_ruptures

    modeled = pvlib.solarposition.sun_rise_set_transit_spa(p.index, lat, lon)
    modeled_mid = modeled["sunrise"] + (modeled["sunset"] - modeled["sunrise"]) / 2
    mask = daytime.power_or_irradiance(p, freq="15min", low_value_threshold=0.005)
    rise, sset = daytime.get_sunrise(mask), daytime.get_sunset(mask)
    mid = rise + (sset - rise) / 2

    def minutes(s):
        s = s.resample("D").mean()
        return s.dt.hour * 60 + s.dt.minute + s.dt.second / 60

    _, amount = shifts_ruptures(minutes(modeled_mid), minutes(mid), period_min=15, shift_min=15, zscore_cutoff=1.5)
    return amount


def correct_time_shifts(p: pd.Series, shift_min: pd.Series) -> pd.Series:
    """Move timestamps by the detected daily clock shift (minutes), back onto the regular grid.

    A shift of -60 means production is centred an hour after solar noon (clock on daylight
    saving time), so those timestamps move back one hour.
    """
    per_step = shift_min.reindex(p.index.normalize()).fillna(0).to_numpy()
    new_index = p.index + pd.to_timedelta(per_step, unit="min")
    moved = pd.Series(p.to_numpy(), index=new_index)
    moved = moved[~moved.index.duplicated(keep="first")]
    return moved.reindex(p.index)


def expected_power(index, ghi, temp_air, lat: float, lon: float, tilt: float, azimuth: float,
                   pdc0_kw: float, pac_max_kw: float, gamma: float = -0.0037, wind_ms: float = 1.0,
                   inverter_eff: float = 0.96) -> pd.DataFrame:
    """Modelled AC power from satellite irradiance with pvlib.

    Erbs decomposition of global horizontal irradiance, Perez transposition to the plane of
    array, physical incidence angle modifier, SAPM cell temperature (fixed wind speed, as the
    weather file has none), the PVWatts DC model and a flat inverter efficiency. The AC limit
    reproduces inverter clipping when one is set.
    """
    import pvlib

    sp = pvlib.solarposition.get_solarposition(index, lat, lon)
    dni_extra = pvlib.irradiance.get_extra_radiation(index)
    erbs = pvlib.irradiance.erbs(ghi, sp["zenith"], index)
    am = pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"])
    poa = pvlib.irradiance.get_total_irradiance(tilt, azimuth, sp["apparent_zenith"], sp["azimuth"],
                                                erbs["dni"], ghi, erbs["dhi"], dni_extra=dni_extra,
                                                airmass=am, model="perez")
    aoi = pvlib.irradiance.aoi(tilt, azimuth, sp["apparent_zenith"], sp["azimuth"])
    iam = pvlib.iam.physical(aoi)
    poa_eff = (poa["poa_direct"] * iam + poa["poa_diffuse"]).clip(lower=0).fillna(0)
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
    t_cell = pvlib.temperature.sapm_cell(poa["poa_global"].fillna(0), temp_air, wind_ms, **params)
    pdc = pvlib.pvsystem.pvwatts_dc(poa_eff, t_cell, pdc0_kw, gamma)
    pac_unclipped = (pdc * inverter_eff).clip(lower=0)
    pac = pac_unclipped.clip(upper=pac_max_kw)
    return pd.DataFrame({"poa_wm2": poa["poa_global"].fillna(0), "poa_eff_wm2": poa_eff, "t_cell": t_cell,
                         "pdc_kw": pdc, "pac_unclipped_kw": pac_unclipped, "pac_kw": pac,
                         "zenith": sp["zenith"]}, index=index)


def fit_dc_capacity(actual: pd.Series, model_unit: pd.DataFrame, mask: pd.Series) -> float:
    """DC capacity (kW) that best matches measured power on clear, unclipped, valid intervals.

    The model is linear in capacity, so the least-squares capacity is sum(a*m)/sum(m*m)
    for the model run at 1 kW.
    """
    a = actual[mask]
    m = model_unit.loc[mask, "pac_unclipped_kw"]
    return float((a * m).sum() / (m * m).sum())


def classify_losses(actual: pd.Series, expected: pd.DataFrame, valid: pd.Series, temp_air: pd.Series,
                    outage_frac: float = 0.05, snow_pi: float = 0.6) -> pd.DataFrame:
    """Daytime intervals classed as outage, suspected snow or normal, with lost energy (kWh)."""
    day = expected["poa_wm2"] > 50
    exp = expected["pac_kw"]
    state = pd.Series("night", index=actual.index)
    state[day & ~valid] = "no data"
    ok = day & valid
    outage = ok & (actual <= outage_frac * exp)
    # snow: low output in cold weather, judged on whole days so a passing cloud is not counted
    daily_pi = (actual.where(ok).resample("D").sum() / exp.where(ok).resample("D").sum())
    cold = temp_air.resample("D").min() < 2.0
    snow_day = (daily_pi < snow_pi) & cold
    snow = ok & ~outage & snow_day.reindex(actual.index, method="ffill").fillna(False)
    state[ok] = "normal"
    state[snow] = "suspected snow"
    state[outage] = "outage"
    lost = np.where(state.isin(["outage", "suspected snow"]), (exp - actual.fillna(0)).clip(lower=0) / 4.0, 0.0)
    return pd.DataFrame({"state": state, "lost_kwh": lost}, index=actual.index)


def performance_ratio(energy_kwh: pd.Series, poa_kwh_m2: pd.Series, pdc0_kw: float) -> pd.Series:
    """IEC 61724-1 performance ratio: final yield over reference yield."""
    return (energy_kwh / pdc0_kw) / poa_kwh_m2


# ---------------------------------------------------------------- independent (pre-construction style) energy estimate
# PVWatts default system losses (Dobos, 2014), as fractions of DC energy: the generic assumptions
# a desk study would use without site knowledge. Snow is zero in the PVWatts defaults.
PVWATTS_LOSSES = {"Soiling": 0.02, "Shading": 0.03, "Snow": 0.0, "Mismatch": 0.02, "Wiring": 0.02,
                  "Connections": 0.005, "Light-induced degradation": 0.015, "Nameplate rating": 0.01,
                  "Availability": 0.03}


def combined_loss(losses: dict) -> float:
    """Losses applied multiplicatively, as PVWatts does."""
    return float(1 - np.prod([1 - v for v in losses.values()]))


def ageing_factor(year_mid: float, installed: float, rate: float) -> float:
    """Share of rated power left after linear degradation at `rate` per year since installation."""
    return float(1 - rate * max(year_mid - installed, 0.0))


def long_term_month_factors(era5_monthly: pd.DataFrame, record_months) -> pd.Series:
    """Long-term over record-period irradiance, by calendar month (a ratio long-term correction).

    `era5_monthly` has a monthly PeriodIndex and a ghi_mean_wm2 column; `record_months` are the
    months of the short record. The factor for each calendar month is the long-term mean ERA5
    irradiance of that month over its mean in the record months.
    """
    g = era5_monthly["ghi_mean_wm2"]
    lt = g.groupby(g.index.month).mean()
    rec = g.reindex(pd.PeriodIndex(record_months, freq="M"))
    conc = rec.groupby(rec.index.month).mean()
    return (lt / conc).rename("factor")


def p_values_kwh(p50: float, historical_pct: float, iav_pct: float, years=(1, 10)) -> pd.DataFrame:
    """P50, P75 and P90 for averaging periods of 1 and 10 years, normal distribution."""
    from scipy import stats

    rows = []
    for n in years:
        total = float(np.sqrt(historical_pct ** 2 + iav_pct ** 2 / n))
        row = {"period_years": n, "total_uncertainty_pct": total}
        for p in (50, 75, 90):
            row[f"P{p}_kwh"] = p50 * (1 - stats.norm.ppf(p / 100.0) * total / 100.0)
        rows.append(row)
    return pd.DataFrame(rows)
