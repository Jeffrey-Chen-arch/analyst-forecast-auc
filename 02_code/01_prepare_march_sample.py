from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd

from config import *
from src.common import setup_logging, ensure_dirs, read_csv_any, write_csv_gz, write_csv


def find_march_zip() -> Path | None:
    # Search project root and 01_input for a zip containing 3.23_rerestart.
    candidates = list(PROJECT_ROOT.glob("*.zip")) + list(INPUT_DIR.glob("*.zip"))
    for z in candidates:
        try:
            with zipfile.ZipFile(z, "r") as f:
                names = f.namelist()
            if any(name.startswith("3.23_rerestart/") for name in names):
                return z
        except zipfile.BadZipFile:
            continue
    return None


def main():
    logger = setup_logging(LOG_DIR / "01_prepare_march_sample.log")
    ensure_dirs([INPUT_DIR, INT_SAMPLE_DIR, INT_LINK_DIR, TABLE_DIR])

    if not MARCH_DIR.exists():
        z = find_march_zip()
        if z is None:
            raise FileNotFoundError(
                "Could not find a zip file containing 3.23_rerestart/. "
                "Place the March 3.23 zip in the project root or 01_input/."
            )
        logger.info("Unzipping March base package: %s", z)
        with zipfile.ZipFile(z, "r") as f:
            f.extractall(INPUT_DIR)
    else:
        logger.info("March base folder already exists: %s", MARCH_DIR)

    # Verify and copy the key base files into this project's own intermediate/raw folders.
    required = [MARCH_GAAP_SAMPLE, MARCH_RAW_SAMPLE, MARCH_CCM_LINKS]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required March base file not found: {p}")

    gaap = read_csv_any(MARCH_GAAP_SAMPLE)
    raw = read_csv_any(MARCH_RAW_SAMPLE)
    ccm = read_csv_any(MARCH_CCM_LINKS)

    # Basic date typing and timing variables for the AUC task.
    gaap["datadate"] = pd.to_datetime(gaap["datadate"], errors="coerce")
    gaap["formation_date"] = gaap["datadate"] + pd.DateOffset(months=FORMATION_MONTHS_AFTER_FYE)
    gaap["forecast_window_start"] = (gaap["formation_date"] + pd.offsets.MonthBegin(1)).dt.normalize()
    gaap["forecast_window_end"] = gaap["forecast_window_start"] + pd.offsets.MonthEnd(0)
    gaap["target_fpedats_t"] = gaap["datadate"]
    gaap["target_fpedats_tp1"] = gaap["datadate"] + pd.DateOffset(years=1)
    gaap["base_sample_name"] = BASE_SAMPLE_NAME

    write_csv_gz(gaap, INT_SAMPLE_DIR / "base_gaap_sample_with_timing.csv.gz")
    write_csv_gz(raw, INT_SAMPLE_DIR / "base_raw_sample.csv.gz")
    write_csv_gz(ccm, RAW_LINK_DIR / "ccm_links_full_from_march.csv.gz")

    summary = pd.DataFrame([
        {"file": "gaap_sample", "n_obs": len(gaap), "n_firms": gaap["gvkey"].nunique(),
         "min_fyear": gaap["fyear"].min(), "max_fyear": gaap["fyear"].max()},
        {"file": "raw_sample", "n_obs": len(raw), "n_firms": raw["gvkey"].nunique(),
         "min_fyear": raw["fyear"].min(), "max_fyear": raw["fyear"].max()},
    ])
    write_csv(summary, TABLE_DIR / "00_base_sample_summary.csv")
    logger.info("Prepared base sample: %s", TABLE_DIR / "00_base_sample_summary.csv")


if __name__ == "__main__":
    main()
