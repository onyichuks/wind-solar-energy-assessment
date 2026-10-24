"""Download the open datasets into ./data/raw. Nothing here is stored in the repository.

* Penmanshiel wind farm data, Zenodo record 5946808 (CC BY 4.0), about 1.05 GB.
* ERA5 hourly reanalysis for the site, 2000 to 2024, from the Open-Meteo archive API.
* NREL PVDAQ system 50 power and satellite weather, from the PVAnalytics example data.

Usage:  python scripts/get_data.py            (everything)
        python scripts/get_data.py --solar    (solar files only, about 1 MB)
See docs/DATASET.md for what each file holds.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ZENODO = "https://zenodo.org/records/5946808/files/{name}?download=1"  # record version 0.0.2
PENMANSHIEL = {  # file name: MD5 checksum as published on the Zenodo record page
    "Penmanshiel_SCADA_2017_WT01-10_3114.zip": "c8279e34a5e33d688e2219aa9abab74e",
    "Penmanshiel_SCADA_2017_WT11-15_3115.zip": "d41874e3d195e88506949492b5758cf6",
    "Penmanshiel_SCADA_2018_WT01-10_3113.zip": "6389e5c4ea59d6565f6712db43a504ce",
    "Penmanshiel_SCADA_2018_WT11-15_3116.zip": "7c40dc00df6d6301427213e15170b927",
    "Penmanshiel_Grid_3153.zip": "52168e766e26f4ba909fcef909b64e0e",
    "Penmanshiel_PMU_3152.zip": "c7e0bf5d1146d531160da0f0131e6944",
    "Penmanshiel_WT_static.csv": "c4cd4191234c1a67a391fe5d2978256b",
}
OPEN_METEO = ("https://archive-api.open-meteo.com/v1/archive?latitude=55.9035&longitude=-2.2915"
              "&start_date=2000-01-01&end_date=2024-12-31"
              "&hourly=wind_speed_10m,wind_speed_100m,wind_direction_100m,temperature_2m,surface_pressure"
              "&models=era5&wind_speed_unit=ms&timezone=GMT&format=csv")
PVA = "https://github.com/pvlib/pvanalytics/raw/main/pvanalytics/data/{name}"
SOLAR = ["system_50_ac_power_2_full_DST.parquet", "system_50_ac_power_2_full_DST_psm3.parquet"]


def fetch(url: str, target: Path) -> None:
    if target.exists():
        print("  already present:", target.name)
        return
    print("  downloading", target.name)
    tmp = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "wind-solar-energy-assessment"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(target)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solar", action="store_true", help="download only the solar files")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "solar").mkdir(exist_ok=True)
    print("Solar (NREL PVDAQ system 50)")
    for name in SOLAR:
        fetch(PVA.format(name=name), RAW / "solar" / name)
    if args.solar:
        return
    print("Penmanshiel wind farm (Zenodo 5946808)")
    for name, digest in PENMANSHIEL.items():
        fetch(ZENODO.format(name=name), RAW / name)
        if md5(RAW / name) != digest:
            sys.exit(f"Checksum mismatch for {name}: delete it and run again")
    print("ERA5 reanalysis (Open-Meteo archive)")
    fetch(OPEN_METEO, RAW / "era5_penmanshiel_openmeteo.csv")
    print("Done. Next: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
