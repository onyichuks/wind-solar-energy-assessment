"""Penmanshiel wind farm: pre-construction style energy yield assessment against actual performance.

Usage:  python run_wind.py
Needs the processed data from scripts/prepare_data.py. Writes tables to outputs/tables,
a summary to outputs/wind_results.json and dashboard data to outputs/wind_dashboard.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from energyassess import eya, operational as op, resource as rs, turbine as tb, wakes as wk  # noqa: E402
from energyassess.outputs import to_json  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
CAMPAIGN_YEAR = 2017  # the 12-month "measurement campaign"
HUB = 59.0
DIAMETER = 82.0
RATED = 2050.0
WS_EDGES = np.arange(0.0, 30.5, 0.5)
WD_STEP = 5.0
HOURS = 8766.0
CENTRAL_MODEL = "jensen"  # Park model with the classic onshore wake decay constant
GAP_YEARS = (2017, 2018)  # the years with turbine SCADA, so every loss is measured over the same period


def r(x, n=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    turbines = pd.read_csv(ROOT / "static_data" / "penmanshiel_turbines.csv")
    names = turbines["turbine"].tolist()
    n_wt = len(names)
    capacity_mw = n_wt * RATED / 1000
    era5 = rs.load_era5_openmeteo(DATA / "raw" / "era5_penmanshiel_openmeteo.csv")
    scada = op.apply_qc(pd.read_parquet(DATA / "processed" / "turbine_scada_10min.parquet"))
    plant = pd.read_parquet(DATA / "processed" / "plant_meter_10min.parquet")
    # corrupt revenue meter days (checked against the summed turbine meters): replace the day's
    # export with the turbine energy less the median electrical loss, and keep it out of the
    # electrical loss measurement
    meter_days = op.meter_day_check(scada, plant)
    meter_days.round(4).to_csv(TABLES / "w00_meter_day_check.csv")
    turb_10 = scada.pivot(index="time", columns="turbine", values="energy_kwh")
    turb_10 = turb_10[turb_10.notna().all(axis=1)].sum(axis=1)
    for day in meter_days.index[meter_days["flag"]]:
        rows = (plant.index >= day) & (plant.index < day + pd.Timedelta("1D"))
        plant.loc[rows, "grid_kwh"] = np.nan
        plant.loc[rows, "export_kwh"] = turb_10.reindex(plant.index[rows]).to_numpy() * meter_days.attrs["median_ratio"]
        plant.loc[rows, "source"] = "turbine meters less electrical loss"
    results: dict = {"site": {"turbines": n_wt, "capacity_mw": capacity_mw, "hub_height_m": HUB,
                              "rotor_diameter_m": DIAMETER, "era5_grid_point": [era5.attrs["lat"], era5.attrs["lon"]]}}

    # ------------------------------------------------------------------ 1. data quality and virtual mast
    qc = op.qc_summary(scada)
    qc.to_csv(TABLES / "w01_scada_data_quality.csv")
    free = op.undisturbed_table(turbines)
    pd.DataFrame({"turbine": names, "undisturbed_share_pct": (100 * free.mean()).round(1).values}).to_csv(
        TABLES / "w02_undisturbed_sectors.csv", index=False)
    mast10 = op.virtual_mast(scada, free)
    mast = op.to_hourly(mast10)
    conc = mast.join(era5, how="inner").dropna(subset=["mast_ws", "mast_dir"])
    site_ti = op.free_stream_turbulence(scada, mast10, free)
    results["data_quality"] = {"scada_valid_pct": r(qc["valid_pct"].mean(), 2),
                               "virtual_mast_hourly_recovery_pct": r(100 * mast["mast_ws"].notna().mean(), 2),
                               "virtual_mast_mean_turbines": r(mast10["n_turbines"].mean(), 2),
                               "free_stream_turbulence_intensity_10ms": r(site_ti, 3),
                               "meter_to_turbine_daily_median_ratio": r(meter_days.attrs["median_ratio"], 4),
                               "meter_days_replaced": [f"{d:%Y-%m-%d}" for d in meter_days.index[meter_days["flag"]]]}

    # ------------------------------------------------------------------ 2. wind resource and long-term correction
    camp = conc[conc.index.year == CAMPAIGN_YEAR]
    mcp = rs.mcp_scaled_distribution(camp["mast_ws"], camp["mast_dir"], camp["ws100"], camp["wd100"],
                                     era5["ws100"], era5["wd100"])
    mcp["table"].round(4).to_csv(TABLES / "w03_mcp_sectors.csv")

    # cross-validation of long-term methods (fit on one year, predict the other)
    pc_std = tb.load_power_curve()
    rho_hub = rs.air_density(era5["t2m"], era5["sp"], era5.attrs["elevation_m"], turbines["elevation_m"].mean() + HUB)
    rho_lt = float(np.mean(rho_hub))
    pc_site = tb.density_adjusted(pc_std, rho_lt)

    def energy_of(ws, w=None):
        p = tb.power_from_curve(pc_site, ws)
        return float(np.average(p, weights=w)) if w is not None else float(p.mean())

    ml_x = rs.ml_features(era5)
    ml_subsets = {"Machine learning, wind speed and direction only": ["ws100", "dir_sin", "dir_cos"],
                  "Machine learning, adding 10 m wind and shear": ["ws100", "dir_sin", "dir_cos", "ws10", "shear"],
                  "Machine learning, all inputs except time of day and season":
                      [c for c in rs.ML_FEATURES if not c.startswith(("hour", "doy"))]}
    cv_rows, hourly_rows = [], []
    for a, b in [(2017, 2018), (2018, 2017)]:
        tr, te = conc[conc.index.year == a], conc[conc.index.year == b]
        y_tr, y_te = tr["mast_ws"].to_numpy(), te["mast_ws"].to_numpy()
        e_meas = energy_of(y_te)
        sc = rs.mcp_scaled_distribution(tr["mast_ws"], tr["mast_dir"], tr["ws100"], tr["wd100"], te["ws100"], te["wd100"])
        row = {"fit_year": a, "test_year": b,
               "Scaled measured distribution (used)": 100 * (energy_of(sc["ws"], sc["weight"]) / e_meas - 1)}
        series = {}
        for label, method in [("Hourly variance ratio", "variance_ratio"), ("Hourly linear regression", "linear")]:
            f = rs.mcp_fit(tr["ws100"], tr["wd100"], y_tr, 12, method)
            series[label] = rs.mcp_predict(f, te["ws100"], te["wd100"])
        ml = rs.mcp_ml_fit(ml_x.loc[tr.index], y_tr)
        series["Machine learning (gradient boosting)"] = rs.mcp_ml_predict(ml, ml_x.loc[te.index])
        for label in series:
            row[label] = 100 * (energy_of(series[label]) / e_meas - 1)
        # further checks on the machine learning method: fewer inputs, and a spread correction
        for label, cols in ml_subsets.items():
            series[label] = rs.mcp_ml_predict(rs.mcp_ml_fit(ml_x.loc[tr.index, cols], y_tr), ml_x.loc[te.index])
        oof = np.empty(len(tr))
        quarter = tr.index.quarter
        for k in (1, 2, 3, 4):
            mk = rs.mcp_ml_fit(ml_x.loc[tr.index[quarter != k]], y_tr[quarter != k])
            oof[quarter == k] = rs.mcp_ml_predict(mk, ml_x.loc[tr.index[quarter == k]])
        series["Machine learning with spread correction"] = rs.quantile_mapper(oof, y_tr)(series["Machine learning (gradient boosting)"])
        for label, p in series.items():
            hourly_rows.append({"fit_year": a, "test_year": b, "method": label, "energy_error_pct": 100 * (energy_of(p) / e_meas - 1),
                                "hourly_rmse_ms": float(np.sqrt(np.mean((p - y_te) ** 2))),
                                "hourly_r2": float(1 - np.sum((p - y_te) ** 2) / np.sum((y_te - y_te.mean()) ** 2)),
                                "spread_ratio": float(np.std(p) / np.std(y_te)), "mean_error_ms": float(np.mean(p) - np.mean(y_te))})
        cv_rows.append(row)
    cv = pd.DataFrame(cv_rows)
    cv.round(2).to_csv(TABLES / "w04_mcp_cross_validation_energy_error_pct.csv", index=False)
    pd.DataFrame(hourly_rows).round(4).to_csv(TABLES / "w17_long_term_method_checks.csv", index=False)

    # desk-study comparison: ERA5 alone, sheared down to hub height, no site data
    ws_era5_hub = rs.extrapolate_to_height(era5, HUB)
    dir_offset = float(((conc["mast_dir"] - conc["wd100"] + 180) % 360 - 180).median())
    weib = rs.weibull_by_sector(mcp["ws"], mcp["wd"], mcp["weight"])
    weib.round(4).to_csv(TABLES / "w05_long_term_wind_rose_weibull.csv")
    annual_ref = era5["ws100"].resample("YE").mean()
    results["wind_resource"] = {
        "campaign": f"{CAMPAIGN_YEAR} virtual met mast at {HUB:.0f} m",
        "campaign_mean_ms": r(mcp["mean_concurrent"]),
        "long_term_mean_ms": r(mcp["mean_long_term"]),
        "era5_desk_study_hub_mean_ms": r(ws_era5_hub.mean()),
        "era5_bias_vs_site_pct": r(100 * (ws_era5_hub.mean() / mcp["mean_long_term"] - 1), 2),
        "weibull_A": r(weib.attrs["all_A"]), "weibull_k": r(weib.attrs["all_k"]),
        "air_density_kgm3": r(rho_lt, 4),
        "long_term_period": f"{era5.index.min().year} to {era5.index.max().year}",
        "reference_interannual_variability_pct": r(100 * annual_ref.std() / annual_ref.mean(), 2),
        "campaign_year_wind_index": r(annual_ref.loc[str(CAMPAIGN_YEAR)].iloc[0] / annual_ref.mean(), 4),
        "hourly_r2_era5_100m": r(np.corrcoef(conc["mast_ws"], conc["ws100"])[0, 1] ** 2),
        "hourly_r2_era5_100m_2017": r(np.corrcoef(camp["mast_ws"], camp["ws100"])[0, 1] ** 2),
        "daily_r2_era5_100m": r(conc[["mast_ws", "ws100"]].resample("D").mean().dropna().corr().iloc[0, 1] ** 2),
        "monthly_r2_era5_100m": r(conc[["mast_ws", "ws100"]].resample("MS").mean().dropna().corr().iloc[0, 1] ** 2),
        "shear_exponent_median": r(float(np.nanmedian(rs.shear_exponent(era5["ws10"], era5["ws100"], 10.0, 100.0))), 3),
        "direction_offset_site_minus_era5_deg": r(dir_offset, 1),
        "direction_offset_mean_deg": r(float(((conc["mast_dir"] - conc["wd100"] + 180) % 360 - 180).mean()), 1),
        "mcp_cross_validation_energy_error_pct": cv.round(2).to_dict(orient="records"),
    }

    # ------------------------------------------------------------------ 3. energy: gross, wakes, losses
    x, y = wk.layout_xy(turbines)
    wt = wk.make_turbine(pc_site, DIAMETER, HUB, RATED)
    wdc, wsc, freq = rs.weighted_joint_frequency(mcp["ws"], mcp["wd"], mcp["weight"], WS_EDGES, WD_STEP)
    free_kw = wk.free_power_matrix(pc_site, n_wt, wdc, wsc)
    gross_t = wk.energy_mwh_per_year(free_kw, freq)
    gross = float(gross_t.sum())
    model_out, power_by_model = {}, {}
    for label, key in wk.MODELS.items():
        P = wk.power_matrix(wk.wind_farm_model(key, wt, ti=site_ti), x, y, wdc, wsc)
        power_by_model[key] = P
        net_t = wk.energy_mwh_per_year(P, freq)
        model_out[key] = {"label": label, "net_mwh": float(net_t.sum()), "wake_loss_pct": 100 * (1 - net_t.sum() / gross),
                          "per_turbine_wake_pct": (100 * (1 - net_t / gross_t)).tolist(), "net_t": net_t}
    central = model_out[CENTRAL_MODEL]
    chain = eya.loss_chain(gross, central["wake_loss_pct"] / 100)
    chain.round(3).to_csv(TABLES / "w06_loss_chain.csv", index=False)
    p50 = chain.attrs["net_mwh"]

    def energy_fn(scale: float) -> float:
        _, _, f2 = rs.weighted_joint_frequency(mcp["ws"] * scale, mcp["wd"], mcp["weight"], WS_EDGES, WD_STEP)
        return float(wk.energy_mwh_per_year(power_by_model[CENTRAL_MODEL], f2).sum())

    sens = eya.sensitivity_ratio(energy_fn)
    unc = eya.Uncertainty(
        wind_speed={
            "Measurement (nacelle anemometers as a virtual mast)": 3.0,
            "Long-term correction (12-month campaign, cross-validated)": 1.5,
            "Long-term reference consistency and period representativeness": 1.3,
            "Vertical extrapolation (measured at hub height)": 0.0,
            "Horizontal extrapolation (uniform flow across the site)": 2.5,
        },
        energy={
            "Power curve (library curve, not warranted)": 4.0,
            "Wake model (20 percent of the wake loss)": 0.2 * central["wake_loss_pct"],
            "Availability and other loss assumptions": 1.5,
            "Electrical losses": 0.5,
        },
        interannual_speed_pct=results["wind_resource"]["reference_interannual_variability_pct"],
        climate_energy_pct=1.0,
    )
    unc_table = eya.combine(unc, sens)
    unc_table.round(3).to_csv(TABLES / "w07_uncertainty.csv", index=False)
    pv = eya.p_values(p50, unc_table)
    pv.round(1).to_csv(TABLES / "w08_p_values.csv", index=False)

    # desk-study energy (ERA5 only) through the same chain, for comparison
    _, _, f_desk = rs.joint_frequency(ws_era5_hub, (era5["wd100"] + dir_offset) % 360, WS_EDGES, WD_STEP)
    gross_desk = float(wk.energy_mwh_per_year(free_kw, f_desk).sum())
    net_desk_wake = float(wk.energy_mwh_per_year(power_by_model[CENTRAL_MODEL], f_desk).sum())
    chain_desk = eya.loss_chain(gross_desk, 1 - net_desk_wake / gross_desk)

    # alternative long-term methods (for the dashboard selector): gross and wake loss per model
    def lt_option(ws_lt, wd_lt, w_lt=None):
        if w_lt is None:
            _, _, f = rs.joint_frequency(ws_lt, wd_lt, WS_EDGES, WD_STEP)
        else:
            _, _, f = rs.weighted_joint_frequency(ws_lt, wd_lt, w_lt, WS_EDGES, WD_STEP)
        g = float(wk.energy_mwh_per_year(free_kw, f).sum())
        return {"gross_mwh": g, "wake_loss_pct": {k: 100 * (1 - float(wk.energy_mwh_per_year(P, f).sum()) / g) for k, P in power_by_model.items()},
                "mean_ms": float(np.average(ws_lt, weights=w_lt)) if w_lt is not None else float(np.mean(ws_lt))}
    wd_lt_site = (era5["wd100"].to_numpy() + dir_offset) % 360
    lt_methods = {"scaled": {"label": "Scaled measured distribution (used)", **lt_option(mcp["ws"], mcp["wd"], mcp["weight"])}}
    for key, label in [("linear", "Hourly linear regression"), ("variance_ratio", "Hourly variance ratio")]:
        f_fit = rs.mcp_fit(camp["ws100"], camp["wd100"], camp["mast_ws"], 12, key)
        lt_methods[key] = {"label": label, **lt_option(rs.mcp_predict(f_fit, era5["ws100"], era5["wd100"]), wd_lt_site)}
    ml_lt = rs.mcp_ml_fit(ml_x.loc[camp.index], camp["mast_ws"])
    lt_methods["ml"] = {"label": "Machine learning (gradient boosting)", **lt_option(rs.mcp_ml_predict(ml_lt, ml_x), wd_lt_site)}
    lt_methods["era5"] = {"label": "ERA5 alone, no site data", **lt_option(ws_era5_hub.to_numpy(), wd_lt_site)}
    for v in lt_methods.values():
        v["p50_mwh"] = float(eya.loss_chain(v["gross_mwh"], v["wake_loss_pct"][CENTRAL_MODEL] / 100).attrs["net_mwh"])

    per_turbine = pd.DataFrame({"turbine": names, "x_m": x.round(1), "y_m": y.round(1),
                                "elevation_m": turbines["elevation_m"], "gross_mwh": gross_t.round(1)})
    for key, mo in model_out.items():
        per_turbine[f"wake_loss_pct_{key}"] = np.round(mo["per_turbine_wake_pct"], 2)
    results["energy_yield"] = {
        "gross_mwh": r(gross, 0), "gross_capacity_factor_pct": r(100 * gross / (capacity_mw * HOURS), 2),
        "wake_models": {k: {"label": v["label"], "wake_loss_pct": r(v["wake_loss_pct"], 2), "net_of_wake_mwh": r(v["net_mwh"], 0)}
                        for k, v in model_out.items()},
        "central_wake_model": central["label"],
        "loss_chain": chain.round(3).to_dict(orient="records"),
        "total_efficiency_pct": r(100 * chain.attrs["total_efficiency"], 2),
        "p50_mwh": r(p50, 0), "p50_capacity_factor_pct": r(100 * p50 / (capacity_mw * HOURS), 2),
        "sensitivity_ratio": r(sens, 3),
        "uncertainty": unc_table.round(3).to_dict(orient="records"),
        "historical_uncertainty_pct": r(unc_table.attrs["historical_pct"], 2),
        "climate_uncertainty_pct": r(unc_table.attrs["climate_pct"], 2),
        "interannual_energy_pct": r(unc_table.attrs["iav_energy_pct"], 2),
        "p_values": pv.round(1).to_dict(orient="records"),
        "desk_study_era5_only_p50_mwh": r(chain_desk.attrs["net_mwh"], 0),
        "lt_methods": {k: {"label": v["label"], "gross_mwh": r(v["gross_mwh"], 1), "mean_ms": r(v["mean_ms"], 3), "p50_mwh": r(v["p50_mwh"], 0),
                           "wake_loss_pct": {m: r(x, 3) for m, x in v["wake_loss_pct"].items()}} for k, v in lt_methods.items()},
    }

    # ------------------------------------------------------------------ 4. operational performance (2017 and 2018 SCADA)
    losses = op.turbine_losses(scada)
    elec = op.electrical_loss(scada, plant)
    ref_curves = op.measured_power_curves(scada)
    ref_curves.round(1).to_csv(TABLES / "w09_measured_power_curves_kw.csv")
    own = op.own_downtime_loss(scada, ref_curves.interpolate(limit_area="inside"))
    losses["own_downtime_mwh"] = own.groupby(scada["turbine"]).sum() / 1000
    losses.round(2).to_csv(TABLES / "w10_turbine_losses_2017_2018.csv")
    avail_meas = losses["energy_mwh"].sum() / (losses["energy_mwh"].sum() + losses["lost_downtime_mwh"].sum())

    # curtailment from the wind farm power setpoint (PMU, December 2018 onwards)
    sp = plant["plant_setpoint_kw"]
    limited = sp < 0.98 * capacity_mw * 1000
    curtailed_share = float(limited[sp.notna()].mean())

    # measured against modelled wake efficiency (same definition: farm mean over undisturbed-turbine mean)
    meas_eff, meas_turb = op.measured_sector_efficiency(scada, mast10, free)
    eff_rows = meas_eff[["sector", "centre_deg", "efficiency", "n"]].rename(columns={"efficiency": "measured"})
    for key in model_out:
        me = op.modelled_sector_efficiency(power_by_model[key], wdc, wsc, freq, free, names)
        eff_rows[key] = me.set_index("sector").reindex(eff_rows["sector"])["efficiency"].to_numpy()
    sector_energy = pd.DataFrame({"sector": (np.floor(((wdc + 15) % 360) / 30)).astype(int),
                                  "share": freq[:, (wsc >= 5) & (wsc <= 11)].sum(axis=1)}).groupby("sector")["share"].sum()
    eff_rows["lt_weight"] = sector_energy.reindex(eff_rows["sector"]).to_numpy()
    eff_rows["lt_weight"] /= eff_rows["lt_weight"].sum()
    eff_rows.round(4).to_csv(TABLES / "w11a_sector_efficiency_vs_undisturbed_turbines.csv", index=False)

    # terrain: measured speed-up factors, then wake efficiency with the terrain effect removed
    terr = op.terrain_factors(scada, mast10, free)
    terr = terr.join(turbines.set_index("turbine")["elevation_m"])
    fleet_curve = ref_curves.median(axis=1).interpolate(limit_area="inside")
    adj = op.terrain_adjusted_efficiency(scada, mast10, free, terr["speed_factor"], fleet_curve)
    eff = adj[["sector", "centre_deg", "efficiency", "n"]].rename(columns={"efficiency": "measured"})
    for key in model_out:
        me = op.model_wake_efficiency(power_by_model[key], free_kw, wdc, wsc, freq)
        eff[key] = me.set_index("sector").reindex(eff["sector"])["efficiency"].to_numpy()
    eff["lt_weight"] = sector_energy.reindex(eff["sector"]).to_numpy()
    eff["lt_weight"] /= eff["lt_weight"].sum()
    eff.round(4).to_csv(TABLES / "w11_sector_wake_efficiency_measured_vs_model.csv", index=False)
    deficit_meas = float((1 - eff["measured"]).mul(eff["lt_weight"]).sum())
    deficit_model = {k: float((1 - eff[k]).mul(eff["lt_weight"]).sum()) for k in model_out}
    wake_meas = {k: model_out[k]["wake_loss_pct"] * deficit_meas / deficit_model[k] for k in model_out}
    raw_deficit = float((1 - eff_rows["measured"]).mul(eff_rows["lt_weight"]).sum())
    raw_model = float((1 - eff_rows[CENTRAL_MODEL]).mul(eff_rows["lt_weight"]).sum())
    wake_meas_raw = central["wake_loss_pct"] * raw_deficit / raw_model
    terr_ratio = op.terrain_energy_ratio(pc_site, freq, wdc, wsc, terr["speed_factor"], free, names)
    terr["gross_energy_ratio"] = terr_ratio
    terr.round(4).to_csv(TABLES / "w16_terrain_speed_factors.csv")
    flow_eff = float((central["net_t"] * terr_ratio.reindex(names).to_numpy()).sum() / central["net_t"].sum())

    # per-turbine: predicted share of farm energy against actual (availability-corrected) share
    pot = losses["energy_mwh"] + losses["lost_downtime_mwh"]
    per_turbine["predicted_net_of_wake_mwh"] = central["net_t"].round(1)
    per_turbine["predicted_share_pct"] = (100 * central["net_t"] / central["net_t"].sum()).round(3)
    per_turbine["actual_share_pct"] = (100 * pot.reindex(names) / pot.sum()).round(3).to_numpy()
    per_turbine["actual_vs_predicted_share"] = (per_turbine["actual_share_pct"] / per_turbine["predicted_share_pct"]).round(4)
    per_turbine["availability_pct"] = losses["availability_pct"].reindex(names).round(2).to_numpy()
    per_turbine["mean_nacelle_wind_ms"] = losses["mean_wind_ms"].reindex(names).round(2).to_numpy()
    per_turbine.to_csv(TABLES / "w12_per_turbine.csv", index=False)

    # ------------------------------------------------------------------ 5. predicted against actual, month by month
    fit_lin = rs.mcp_fit(camp["ws100"], camp["wd100"], camp["mast_ws"], 12, "linear")
    ws_idx = rs.mcp_predict(fit_lin, era5["ws100"], era5["wd100"])
    wd_idx = (era5["wd100"].to_numpy() + dir_offset) % 360
    farm_kw = power_by_model[CENTRAL_MODEL].sum(axis=0)  # [wd, ws]
    wi = (np.round(wd_idx / WD_STEP).astype(int)) % len(wdc)
    index_kw = np.array([np.interp(v, wsc, farm_kw[k], left=0, right=0) for v, k in zip(ws_idx, wi)])
    idx = pd.Series(index_kw, index=era5.index)
    lt_mean_kw = idx.mean()
    monthly = pd.DataFrame({"index_kw": idx.resample("MS").mean(), "hours": idx.resample("MS").size()})
    monthly["predicted_mwh"] = p50 / HOURS * monthly["hours"] * monthly["index_kw"] / lt_mean_kw
    pm = plant["export_kwh"].resample("MS").agg(["sum", "count"])
    expected_n = plant["export_kwh"].resample("MS").size()
    monthly["actual_mwh"] = pm["sum"] / 1000
    monthly["coverage"] = pm["count"] / expected_n
    monthly["source"] = plant["source"].resample("MS").agg(lambda s: s.mode().iloc[0] if len(s) else "missing")
    op_months = monthly.loc["2016-09-01":"2021-06-01"].copy()  # commercial operation began 1 September 2016
    op_months = op_months[op_months["coverage"] >= 0.97]
    op_months["ratio"] = op_months["actual_mwh"] / op_months["predicted_mwh"]
    op_months.round(4).to_csv(TABLES / "w13_monthly_predicted_vs_actual.csv")
    ratio_all = op_months["actual_mwh"].sum() / op_months["predicted_mwh"].sum()
    yearly = op_months.groupby(op_months.index.year)[["actual_mwh", "predicted_mwh"]].sum()
    yearly["months"] = op_months.groupby(op_months.index.year).size()
    yearly["ratio"] = yearly["actual_mwh"] / yearly["predicted_mwh"]
    yearly.round(3).to_csv(TABLES / "w14_yearly_predicted_vs_actual.csv")

    # cross-check: OpenOA-style regression of monthly energy per hour on the monthly index
    xm, ym = op_months["index_kw"].to_numpy(), (op_months["actual_mwh"] / op_months["hours"]).to_numpy()
    slope, intercept = np.polyfit(xm, ym, 1)
    r2 = np.corrcoef(xm, ym)[0, 1] ** 2
    op_aep_regression = (intercept + slope * lt_mean_kw) * HOURS
    op_aep_ratio = p50 * ratio_all

    # ------------------------------------------------------------------ 6. gap waterfall (pre-construction P50 to actual, SCADA years)
    # Losses are measured from 2017 and 2018 SCADA, so the actual end point uses the same two years.
    gap_months = op_months[(op_months.index.year >= GAP_YEARS[0]) & (op_months.index.year <= GAP_YEARS[1])]
    ratio_gap = gap_months["actual_mwh"].sum() / gap_months["predicted_mwh"].sum()
    later = op_months[op_months.index.year > GAP_YEARS[1]]
    ratio_later = later["actual_mwh"].sum() / later["predicted_mwh"].sum()
    target = p50 * ratio_gap
    assumed = {g: float(np.prod(1 - chain.loc[chain["group"] == g, "loss_pct"].to_numpy() / 100))
               for g in chain["group"].unique()}  # efficiency per loss group
    steps, e = [], p50
    def step(label, eff_pred, eff_meas):
        nonlocal e
        new = e * eff_meas / eff_pred
        steps.append({"step": label, "assumed_loss_pct": 100 * (1 - eff_pred), "measured_loss_pct": 100 * (1 - eff_meas),
                      "change_mwh": new - e})
        e = new
    # power curve: the fleet's measured nacelle curve against the library curve over the long-term distribution
    fc = fleet_curve.dropna()
    fleet_full = pc_site.copy()
    inside = (pc_site.index >= fc.index.min()) & (pc_site.index <= fc.index.max())
    fleet_full[inside] = np.interp(pc_site.index[inside], fc.index.to_numpy(dtype=float), fc.to_numpy(dtype=float))
    pc_eff = float(np.average(tb.power_from_curve(fleet_full, mcp["ws"]), weights=mcp["weight"]) /
                   np.average(tb.power_from_curve(pc_site, mcp["ws"]), weights=mcp["weight"]))
    step("Site flow (terrain speed-up, measured)", 1.0, flow_eff)
    step("Power curve and turbine performance", assumed["Turbine performance"], pc_eff)
    step("Wakes (internal, measured on site)", 1 - central["wake_loss_pct"] / 100, 1 - wake_meas[CENTRAL_MODEL] / 100)
    step("Availability (turbine, balance of plant, grid)", assumed["Availability"], avail_meas)
    step("Electrical losses", assumed["Electrical"], 1 - elec["loss_pct"] / 100)
    # curtailment in 2017 and 2018 from the operator's turbine records, the same years as every
    # other step (the wind farm setpoint is only logged from December 2018)
    lost_c = float(scada["lost_curtailment_kwh"].sum())
    curt_loss = lost_c / (float(scada["energy_kwh"].sum()) + lost_c)
    step("Curtailment", assumed["Curtailment"], 1 - curt_loss)
    # December 2018 to June 2021, from the setpoint: expected power above the setpoint, in
    # intervals where the farm was running at the setpoint (so the setpoint was the limit)
    exp_kw = (idx * (1000 * p50 / HOURS / lt_mean_kw) * ratio_all).reindex(plant.index, method="nearest").clip(upper=capacity_mw * 1000)
    period = sp.notna()
    held = limited & period & (plant["plant_power_kw"] >= 0.9 * sp)
    lost_curt = float((exp_kw - sp).clip(lower=0)[held].sum() / 6)
    curt_loss_later = lost_curt / (float(plant.loc[period, "export_kwh"].sum()) + lost_curt)
    residual = target - e
    steps.append({"step": "Environmental allowance, long-term method and other (residual)", "assumed_loss_pct": np.nan,
                  "measured_loss_pct": np.nan, "change_mwh": residual})
    gap = pd.DataFrame(steps)
    gap.round(2).to_csv(TABLES / "w15_gap_waterfall.csv", index=False)

    # where the long-term actual sits in the estimate's distribution (uncertainty of a long-term mean)
    from scipy import stats as _st
    lt_unc = np.hypot(unc_table.attrs["historical_pct"], unc_table.attrs["climate_pct"]) / 100
    exceed_pct = 100 * (1 - _st.norm.cdf((op_aep_ratio - p50) / (p50 * lt_unc)))
    wake_unc = 0.2 * central["wake_loss_pct"]
    wake_error_sigma = (central["wake_loss_pct"] - wake_meas[CENTRAL_MODEL]) / wake_unc

    results["operational"] = {
        "scada_period": "2017 to 2018",
        "turbine_energy_mwh": r(losses["energy_mwh"].sum(), 0),
        "energy_availability_pct": r(100 * avail_meas, 2),
        "operator_downtime_loss_mwh": r(losses["lost_downtime_mwh"].sum(), 0),
        "own_downtime_loss_mwh": r(losses["own_downtime_mwh"].sum(), 0),
        "electrical": {k: r(v, 3) for k, v in elec.items()},
        "curtailed_time_pct": r(100 * curtailed_share, 3),
        "curtailment_loss_pct": r(100 * curt_loss, 3),
        "curtailment_loss_pct_from_setpoint_dec2018_jun2021": r(100 * curt_loss_later, 3),
        "curtailment_held_intervals_dec2018_jun2021": int(held.sum()),
        "measured_wake_loss_pct": {k: r(v, 2) for k, v in wake_meas.items()},
        "measured_wake_loss_without_terrain_adjustment_pct": r(wake_meas_raw, 2),
        "terrain_speed_factor_range": [r(terr["speed_factor"].min(), 3), r(terr["speed_factor"].max(), 3)],
        "terrain_vs_elevation_r": r(np.corrcoef(terr["speed_factor"], terr["elevation_m"])[0, 1], 3),
        "site_flow_energy_effect_pct": r(100 * (flow_eff - 1), 2),
        "power_curve_energy_effect_pct": r(100 * (pc_eff - 1), 2),
        "operator_performance_loss_pct": r(100 * losses["lost_performance_mwh"].sum() / (losses["energy_mwh"].sum() + losses["lost_performance_mwh"].sum()), 3),
        "wake_model_error_in_standard_deviations": r(wake_error_sigma, 2),
        "terrain_free_intervals_min": [str(terr["intervals"].idxmin()), int(terr["intervals"].min())],
        "sector_deficit_measured": r(deficit_meas, 4),
        "sector_deficit_model": {k: r(v, 4) for k, v in deficit_model.items()},
        "per_turbine_share_range": [r(per_turbine["actual_vs_predicted_share"].min(), 3), r(per_turbine["actual_vs_predicted_share"].max(), 3)],
    }
    results["predicted_vs_actual"] = {
        "months_compared": int(len(op_months)), "period": f"{op_months.index.min():%B %Y} to {op_months.index.max():%B %Y}",
        "actual_mwh": r(op_months["actual_mwh"].sum(), 0), "predicted_mwh": r(op_months["predicted_mwh"].sum(), 0),
        "actual_over_predicted": r(ratio_all, 4),
        "yearly": yearly.reset_index().rename(columns={"index": "year"}).round(3).to_dict(orient="records"),
        "operational_aep_ratio_method_mwh": r(op_aep_ratio, 0),
        "operational_aep_regression_mwh": r(op_aep_regression, 0), "regression_r2": r(r2, 3),
        "gap_mwh": r(op_aep_ratio - p50, 0), "gap_pct": r(100 * (op_aep_ratio / p50 - 1), 2),
        "exceedance_level_long_term_pct": r(exceed_pct, 1),
        "gap_basis": f"{GAP_YEARS[0]} to {GAP_YEARS[1]}",
        "actual_over_predicted_gap_years": r(ratio_gap, 4),
        "actual_over_predicted_after_gap_years": r(ratio_later, 4),
        "gap_target_mwh": r(target, 0),
        "waterfall": gap.round(1).to_dict(orient="records"),
    }

    (OUT / "wind_results.json").write_text(to_json(results, indent=2))

    # ------------------------------------------------------------------ dashboard data
    rose = rs.weibull_by_sector(mcp["ws"], mcp["wd"], mcp["weight"], n_sectors=16)
    speed_hist = np.histogram(mcp["ws"], bins=np.arange(0, 26, 1.0), weights=mcp["weight"])[0]
    cs = camp.sample(min(3000, len(camp)), random_state=1)
    dash = {
        "results": results,
        "layout": per_turbine.to_dict(orient="list"),
        "rose": {"centre": rose["centre_deg"].tolist(), "freq": (100 * rose["frequency"]).round(2).tolist(),
                 "mean": rose["mean_ms"].round(2).tolist()},
        "speed_hist": {"bins": list(range(0, 25)), "pct": (100 * speed_hist / speed_hist.sum()).round(2).tolist()},
        "mcp_scatter": {"ref": cs["ws100"].round(2).tolist(), "site": cs["mast_ws"].round(2).tolist()},
        "power_curve": {"ws": pc_site.index.tolist(), "reference_kw": pc_site.round(1).tolist(),
                        "measured_kw": ref_curves[[c for c in ["T01", "T15", "T12"] if c in ref_curves]].median(axis=1)
                        .reindex(pc_site.index).round(1).tolist()},
        "sector_eff": eff.round(4).to_dict(orient="list"),
        "terrain": terr.reset_index().round(4).to_dict(orient="list"),
        "monthly": {"month": [f"{d:%Y-%m}" for d in op_months.index], "actual": op_months["actual_mwh"].round(1).tolist(),
                    "predicted": op_months["predicted_mwh"].round(1).tolist(), "ratio": op_months["ratio"].round(4).tolist(),
                    "source": op_months["source"].tolist()},
        "gap": gap.round(1).to_dict(orient="list"),
        "measured": {"flow_eff": r(flow_eff, 5), "pc_eff": r(pc_eff, 5), "avail_eff": r(avail_meas, 5),
                     "elec_eff": r(1 - elec["loss_pct"] / 100, 5), "curt_eff": r(1 - curt_loss, 5),
                     "wake_loss_pct": {k: r(v, 3) for k, v in wake_meas.items()}, "target_mwh": r(target, 1)},
        "annual_index": {"year": [int(d.year) for d in annual_ref.index], "ws100": annual_ref.round(3).tolist()},
    }
    (OUT / "wind_dashboard.json").write_text(to_json(dash, compact=True))

    print(json.dumps({"long_term_ms": results["wind_resource"]["long_term_mean_ms"],
                      "era5_bias_pct": results["wind_resource"]["era5_bias_vs_site_pct"],
                      "gross": results["energy_yield"]["gross_mwh"], "wake": {k: v["wake_loss_pct"] for k, v in results["energy_yield"]["wake_models"].items()},
                      "p50": results["energy_yield"]["p50_mwh"], "p90_1y": pv.loc[0, "P90_mwh"], "p90_10y": pv.loc[1, "P90_mwh"],
                      "desk": results["energy_yield"]["desk_study_era5_only_p50_mwh"],
                      "avail": results["operational"]["energy_availability_pct"], "elec": elec["loss_pct"],
                      "wake_meas": results["operational"]["measured_wake_loss_pct"],
                      "ratio": ratio_all, "op_aep_ratio": op_aep_ratio, "op_aep_reg": op_aep_regression, "r2": r2,
                      "sens": sens, "unc_hist": unc_table.attrs["historical_pct"]}, indent=1, default=float))
    print(gap.round(1).to_string())
    print(yearly.round(3).to_string())
    print(cv.round(2).to_string())


if __name__ == "__main__":
    main()
