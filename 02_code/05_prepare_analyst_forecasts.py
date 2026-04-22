from __future__ import annotations

import pandas as pd

from config import *
from src.common import setup_logging, ensure_dirs, read_csv_any, write_csv_gz, write_csv


def main():
    logger = setup_logging(LOG_DIR / "05_prepare_analyst_forecasts.log")
    ensure_dirs([INT_IBES_DIR, TABLE_DIR])

    df = read_csv_any(
        RAW_IBES_DIR / "ibes_detail_forecasts_raw.csv.gz",
        force_str_cols=["ticker", "cusip", "oftic", "estimator", "analys", "broker"],
    )
    df.columns = [c.lower() for c in df.columns]

    needed = {"ticker", "fpedats", "anndats", "value"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing required columns {needed - set(df.columns)}")

    df["ticker"]           = df["ticker"].astype(str).str.strip()
    df["fpedats"]          = pd.to_datetime(df["fpedats"],  errors="coerce")
    df["forecast_anndats"] = pd.to_datetime(df["anndats"],  errors="coerce")
    df["forecast_revdats"] = pd.to_datetime(df["revdats"],  errors="coerce") \
                              if "revdats" in df.columns else pd.NaT
    df["forecast_eps"]     = pd.to_numeric(df["value"], errors="coerce")

    # Content filters — must accept both 'ANN' and 'A' for pdicity
    if "measure" in df.columns:
        df = df[df["measure"].astype(str).str.upper().eq(IBES_MEASURE)]
    if "pdicity" in df.columns:
        df = df[df["pdicity"].astype(str).str.upper().isin(["ANN", "A"])]

    # Currency filter: defensive re-apply.  In ibes.det_epsus the individual
    # forecast-level `curr` column is almost always NULL; the denominating
    # currency lives in `curr_act`.  `usfirm = 1` already restricts to the US
    # file, so we accept NULL as an effective USD and prefer `curr_act`.
    def _is_usd_or_null(s):
        t = s.astype(str).str.upper().str.strip()
        return t.eq(IBES_CURRENCY) | t.isin(["", "NAN", "NONE", "<NA>"])
    if "curr_act" in df.columns:
        df = df[_is_usd_or_null(df["curr_act"])]
    elif "curr" in df.columns:
        df = df[_is_usd_or_null(df["curr"])]
    elif "curcode" in df.columns:
        df = df[_is_usd_or_null(df["curcode"])]

    # fpi filter in pandas as secondary safety net
    if "fpi" in df.columns:
        df = df[df["fpi"].astype(str).str.strip().eq(IBES_FPI_ANNUAL)]

    df = df[
        df["ticker"].notna() & df["fpedats"].notna() &
        df["forecast_anndats"].notna() & df["forecast_eps"].notna()
    ].copy()

    # ── Analyst identifier construction ───────────────────────────────────────
    # Strategy: combine estimator + analys (+ broker) into one composite ID.
    # If all parts are missing/empty, assign a row-unique fallback to prevent
    # multiple distinct unknown analysts from being merged into one.
    id_parts = []
    for c in ["estimator", "analys", "broker"]:
        if c in df.columns:
            id_parts.append(df[c].astype(str).str.strip()
                            .replace({"nan": "", "None": "", "<NA>": ""}))

    if id_parts:
        combined = id_parts[0]
        for p in id_parts[1:]:
            combined = combined + "|" + p
        df["analyst_id"] = combined.str.strip("|").str.replace(r"\|+", "|", regex=True)
    else:
        df["analyst_id"] = pd.NA

    df["analyst_id"] = df["analyst_id"].replace({"": pd.NA, "|": pd.NA})
    df["analyst_id_quality"] = "observed"

    missing_mask = df["analyst_id"].isna()
    n_missing    = int(missing_mask.sum())
    pct_missing  = 100 * n_missing / len(df) if len(df) else 0.0

    # Assign unique row-level IDs for missing analysts — never collapse them
    if n_missing > 0:
        fallback_ids = "unknown_row_" + df.loc[missing_mask].index.astype(str)
        df.loc[missing_mask, "analyst_id"]         = fallback_ids
        df.loc[missing_mask, "analyst_id_quality"] = "missing_fallback_unique_row"

    logger.info("analyst_id: %d observed (%.1f%%), %d fallback (%.1f%%)",
                len(df) - n_missing, 100 - pct_missing, n_missing, pct_missing)

    # ── Save ──────────────────────────────────────────────────────────────────
    keep_cols = [c for c in [
        "ticker", "cusip", "oftic", "cname", "fpedats",
        "forecast_anndats", "forecast_revdats",
        "estimator", "analys", "broker", "analyst_id", "analyst_id_quality",
        "forecast_eps", "measure", "pdicity", "fpi", "usfirm", "curr", "curcode",
    ] if c in df.columns]
    out = df[keep_cols].drop_duplicates().copy()
    write_csv_gz(out, INT_IBES_DIR / "detail_forecasts_clean_all_rows.csv.gz")

    summary = pd.DataFrame([{
        "n_rows":                   len(out),
        "n_tickers":                out["ticker"].nunique(),
        "n_analyst_ids_observed":   int((out["analyst_id_quality"] == "observed").sum()),
        "n_analyst_ids_fallback":   n_missing,
        "pct_analyst_ids_fallback": f"{pct_missing:.1f}%",
        "min_forecast_anndats":     str(out["forecast_anndats"].min().date()),
        "max_forecast_anndats":     str(out["forecast_anndats"].max().date()),
        "min_fpedats":              str(out["fpedats"].min().date()),
        "max_fpedats":              str(out["fpedats"].max().date()),
    }])
    write_csv(summary, TABLE_DIR / "04_detail_forecasts_clean_summary.csv")
    logger.info("Clean detail forecasts saved: %d rows, %d tickers",
                len(out), out["ticker"].nunique())


if __name__ == "__main__":
    main()
