from __future__ import annotations

import pandas as pd

from config import *
from src.common import setup_logging, ensure_dirs, read_csv_any, write_csv_gz, write_csv


def main():
    logger = setup_logging(LOG_DIR / "04_prepare_ibes_actuals.log")
    ensure_dirs([INT_IBES_DIR, TABLE_DIR])

    df = read_csv_any(
        RAW_IBES_DIR / "ibes_actuals_raw.csv.gz",
        force_str_cols=["ticker", "cusip", "oftic"],
    )
    df.columns = [c.lower() for c in df.columns]

    if "ticker" not in df.columns or "value" not in df.columns:
        raise ValueError(f"IBES actuals missing ticker/value. Columns: {list(df.columns)}")

    # ── Period-end date ───────────────────────────────────────────────────────
    if "pends" in df.columns:
        df["fpedats"] = pd.to_datetime(df["pends"], errors="coerce")
    elif "fpedats" in df.columns:
        df["fpedats"] = pd.to_datetime(df["fpedats"], errors="coerce")
    else:
        raise ValueError("IBES actuals need a period-end date (pends or fpedats).")

    df["actual_eps"] = pd.to_numeric(df["value"], errors="coerce")

    # ── Announcement date: use anndats; fill missing with actdats ─────────────
    # actdats = date IBES recorded the actual; acceptable fallback for anndats.
    df["actual_anndats"] = pd.to_datetime(df.get("anndats"), errors="coerce") \
                            if "anndats" in df.columns else pd.NaT
    if "actdats" in df.columns:
        actdats = pd.to_datetime(df["actdats"], errors="coerce")
        df["actual_anndats"] = df["actual_anndats"].fillna(actdats)

    # ── Content filters (defensive re-apply) ─────────────────────────────────
    if "measure" in df.columns:
        df = df[df["measure"].astype(str).str.upper().eq(IBES_MEASURE)]

    if "pdicity" in df.columns:
        # Accept both 'ANN' and 'A' regardless of WRDS schema version
        df = df[df["pdicity"].astype(str).str.upper().isin(["ANN", "A"])]

    if "curr_act" in df.columns:
        df = df[df["curr_act"].astype(str).str.upper().eq(IBES_CURRENCY)]

    df = df[df["ticker"].notna() & df["fpedats"].notna() & df["actual_eps"].notna()].copy()

    # ── Duplicate audit ───────────────────────────────────────────────────────
    dup = (
        df.groupby(["ticker", "fpedats"]).size()
        .reset_index(name="n_actual_rows")
        .query("n_actual_rows > 1")
    )
    logger.info("Duplicate (ticker, fpedats) pairs: %d", len(dup))
    write_csv(dup, TABLE_DIR / "03_actuals_duplicate_audit.csv")

    # ── Save ──────────────────────────────────────────────────────────────────
    keep_cols = [c for c in [
        "ticker", "cusip", "oftic", "cname", "fpedats",
        "actual_anndats", "actual_eps", "measure", "pdicity", "curr_act", "usfirm",
    ] if c in df.columns]
    out = df[keep_cols].drop_duplicates().copy()
    write_csv_gz(out, INT_IBES_DIR / "actuals_clean_all_rows.csv.gz")

    n_with_anndats = int(out["actual_anndats"].notna().sum()) \
                     if "actual_anndats" in out.columns else "N/A"
    summary = pd.DataFrame([{
        "n_rows":                          len(out),
        "n_tickers":                       out["ticker"].nunique(),
        "min_fpedats":                     str(out["fpedats"].min().date()),
        "max_fpedats":                     str(out["fpedats"].max().date()),
        "duplicate_ticker_fpedats_pairs":  len(dup),
        "rows_with_actual_anndats":        n_with_anndats,
        "pct_with_actual_anndats":         f"{100*n_with_anndats/len(out):.1f}%"
                                           if isinstance(n_with_anndats, int) and len(out) else "N/A",
    }])
    write_csv(summary, TABLE_DIR / "03_actuals_clean_summary.csv")
    logger.info("Clean IBES actuals: %d rows, %d tickers, %s with anndats",
                len(out), out["ticker"].nunique(), n_with_anndats)


if __name__ == "__main__":
    main()
