"""Convert the raw Zenodo downloads in ./data/raw into compact files in ./data/processed.

The raw turbine exports are about 150 MB per turbine-year with 300 signals. This step keeps
the 20 signals the analysis needs and writes Parquet files, so the analysis runs in seconds.

Usage:  python scripts/prepare_data.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energyassess import scada  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SCADA_ZIPS = [
    "Penmanshiel_SCADA_2017_WT01-10_3114.zip",
    "Penmanshiel_SCADA_2017_WT11-15_3115.zip",
    "Penmanshiel_SCADA_2018_WT01-10_3113.zip",
    "Penmanshiel_SCADA_2018_WT11-15_3116.zip",
]


def _one(name: str):
    return scada.load_turbine_zip(RAW / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [n for n in SCADA_ZIPS + ["Penmanshiel_Grid_3153.zip", "Penmanshiel_PMU_3152.zip"] if not (RAW / n).exists()]
    if missing:
        sys.exit(f"Missing raw files in {RAW}: {missing}. Run python scripts/get_data.py first.")
    with ProcessPoolExecutor(max_workers=2) as pool:  # about 1.4 GB of memory per worker
        parts = list(pool.map(_one, SCADA_ZIPS))
    data = pd.concat([p[0] for p in parts], ignore_index=True)
    data = data.drop_duplicates(["turbine", "time"]).sort_values(["turbine", "time"]).reset_index(drop=True)
    for c in data.columns:
        if data[c].dtype == "float64":
            data[c] = data[c].astype("float32")
    data.to_parquet(OUT / "turbine_scada_10min.parquet", index=False)
    status = scada.tidy_status(pd.concat([p[1] for p in parts], ignore_index=True))
    status.to_parquet(OUT / "turbine_status.parquet", index=False)
    plant = scada.load_plant_meter(RAW / "Penmanshiel_Grid_3153.zip", RAW / "Penmanshiel_PMU_3152.zip")
    plant.to_parquet(OUT / "plant_meter_10min.parquet")
    print(f"Turbine SCADA: {len(data):,} rows, {data['turbine'].nunique()} turbines")
    print(f"Status events: {len(status):,}")
    print(f"Plant meter: {len(plant):,} rows, PMU to grid ratio {plant.attrs['pmu_to_grid_ratio']:.4f}")


if __name__ == "__main__":
    main()
