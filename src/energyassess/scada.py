"""Loading the Penmanshiel Greenbyte exports (turbine SCADA, status logs, grid meter and PMU)."""
from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Greenbyte column -> short name. Only the signals this project uses are kept.
TURBINE_COLUMNS = {
    "Date and time": "time",
    "Wind speed (m/s)": "wind_ms",
    "Wind speed, Standard deviation (m/s)": "wind_std_ms",
    "Wind speed Sensor 1 (m/s)": "wind1_ms",
    "Wind speed Sensor 2 (m/s)": "wind2_ms",
    "Wind direction (°)": "wind_dir_deg",
    "Nacelle position (°)": "nacelle_deg",
    "Power (kW)": "power_kw",
    "Power, Minimum (kW)": "power_min_kw",
    "Power, Maximum (kW)": "power_max_kw",
    "Energy Export (kWh)": "energy_kwh",
    "Potential power default PC (kW)": "potential_kw",
    "Turbine Power setpoint (kW)": "setpoint_kw",
    "Lost Production to Downtime (kWh)": "lost_downtime_kwh",
    "Lost Production to Performance (kWh)": "lost_performance_kwh",
    "Lost Production to Curtailment (Total) (kWh)": "lost_curtailment_kwh",
    "Lost Production to Curtailment (Grid) (kWh)": "lost_curtail_grid_kwh",
    "Energy Theoretical (kWh)": "energy_theoretical_kwh",
    "Nacelle ambient temperature (°C)": "temp_c",
    "Blade angle (pitch position) A (°)": "pitch_deg",
    "Rotor speed (RPM)": "rotor_rpm",
}


def _header(lines) -> list[str]:
    for line in lines:
        if line.startswith("# Date and time"):
            return next(csv.reader([line[2:].strip()]))
    raise ValueError("Greenbyte header line not found")


def read_greenbyte_csv(raw: bytes, columns: dict) -> pd.DataFrame:
    """Read one Greenbyte CSV export, keeping and renaming the listed columns.

    The bytes are parsed directly rather than decoded to one large string first, to save memory.
    """
    head = raw[:200_000].decode("utf-8-sig", errors="ignore").splitlines()
    names = _header(head)
    keep = [c for c in columns if c in names]
    df = pd.read_csv(io.BytesIO(raw), comment="#", header=None, names=names, usecols=keep,
                     na_values=["NaN"], low_memory=False, encoding="utf-8-sig")
    df = df.rename(columns=columns)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def turbine_from_name(name: str) -> str:
    m = re.search(r"Penmanshiel_(\d{2})_", name)
    return f"T{m.group(1)}"


def load_turbine_zip(zip_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All turbines in one Zenodo SCADA zip: (10-minute data, status events)."""
    data, status = [], []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = Path(name).name
            if base.startswith("Turbine_Data"):
                df = read_greenbyte_csv(z.read(name), TURBINE_COLUMNS)
                df.insert(0, "turbine", turbine_from_name(base))
                data.append(df)
            elif base.startswith("Status_"):
                st = pd.read_csv(io.BytesIO(z.read(name)), comment="#")
                st.insert(0, "turbine", turbine_from_name(base))
                status.append(st)
    return pd.concat(data, ignore_index=True), pd.concat(status, ignore_index=True)


def tidy_status(status: pd.DataFrame) -> pd.DataFrame:
    """Status log with parsed times and short column names."""
    s = status.rename(columns={"Timestamp start": "start", "Timestamp end": "end", "Status": "status",
                               "Code": "code", "Message": "message", "IEC category": "iec_category",
                               "Service contract category": "contract_category"})
    s["start"] = pd.to_datetime(s["start"], utc=True, errors="coerce")
    s["end"] = pd.to_datetime(s["end"], utc=True, errors="coerce")  # open events are written as "-"
    return s[["turbine", "start", "end", "status", "code", "message", "iec_category", "contract_category"]]


def flat_runs(s: pd.Series, min_len: int) -> pd.Series:
    """True for values inside runs of at least `min_len` identical, non-missing consecutive values."""
    v = s.to_numpy(dtype=float)
    new_run = np.r_[True, v[1:] != v[:-1]]  # NaN never equals NaN, so gaps always break a run
    run_id = np.cumsum(new_run)
    length = np.bincount(run_id)[run_id]
    return pd.Series((length >= min_len) & ~np.isnan(v), index=s.index)


def load_plant_meter(grid_zip, pmu_zip) -> pd.DataFrame:
    """Plant export (kWh per 10 minutes) from the revenue grid meter, gaps filled from the PMU.

    The PMU (wind farm controller meter) matches the grid meter closely where both report,
    so it is scaled by their overlap ratio and used where the grid meter has no data. PMU
    stretches where one energy value repeats for an hour or more (a counter gap spread evenly,
    as from 21 April to 16 May 2018) are left out of the ratio.
    Also returns the wind farm power setpoint from the PMU, used to flag curtailment.
    """
    def _device(zip_path, prefix, cols):
        with zipfile.ZipFile(zip_path) as z:
            name = [n for n in z.namelist() if Path(n).name.startswith(prefix)][0]
            return read_greenbyte_csv(z.read(name), cols)

    grid = _device(grid_zip, "Device_Data", {"Date and time": "time", "Grid Meter Energy Export (kWh)": "grid_kwh"})
    pmu = _device(pmu_zip, "Device_Data", {"Date and time": "time", "GMS Energy Export (kWh)": "pmu_kwh",
                                            "GMS Power setpoint (kW)": "plant_setpoint_kw",
                                            "GMS Power (kW)": "plant_power_kw"})
    df = grid.merge(pmu, on="time", how="outer").set_index("time").sort_index()
    spread = flat_runs(df["pmu_kwh"], 6) & (df["pmu_kwh"] > 1.0)
    both = df["grid_kwh"].notna() & df["pmu_kwh"].notna() & ~spread
    ratio = df.loc[both, "grid_kwh"].sum() / df.loc[both, "pmu_kwh"].sum()
    df["export_kwh"] = df["grid_kwh"].fillna(df["pmu_kwh"] * ratio)
    df["source"] = np.where(df["grid_kwh"].notna(), "grid meter", np.where(df["pmu_kwh"].notna(), "PMU", "missing"))
    df.attrs["pmu_to_grid_ratio"] = ratio
    return df
