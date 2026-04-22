"""
03_wrds_pull_ibes.py — Download IBES and linking data from WRDS.

Changes from v3:
  - Must Fix 2: Raise RuntimeError if WRDS link table is missing 'score' column.
    The main sample is defined by score ≤ 2; if score does not exist we cannot
    enforce that threshold and must stop immediately rather than silently
    producing an unscored "main" sample.
  - Suggest 2: Expand IBES ticker scope to include both scored-link tickers
    AND ibtic tickers from comp.funda.  This ensures that firms reachable only
    via the ibtic appendix route have their IBES actuals / forecasts downloaded,
    making the appendix coverage diagnostic meaningful.
    Important: the ibtic tickers only expand the *download scope*; they are NOT
    added to the main sample.  The main sample still uses only scored WRDS links.
"""
from __future__ import annotations

import pandas as pd

from config import *
from src.common import (
    setup_logging, ensure_dirs, read_csv_any, write_csv_gz, write_csv,
    infer_ibes_basis,
)


# ── WRDS helpers ──────────────────────────────────────────────────────────────

def get_columns(conn, schema: str, table: str) -> list[str]:
    q = f"""
        select column_name from information_schema.columns
        where table_schema = '{schema}' and table_name = '{table}'
        order by ordinal_position
    """
    try:
        return conn.raw_sql(q)["column_name"].str.lower().tolist()
    except Exception:
        return []


def resolve_table(conn, schema_candidates, table_candidates):
    for schema in schema_candidates:
        for table in table_candidates:
            cols = get_columns(conn, schema, table)
            if cols:
                return schema, table, cols
    raise RuntimeError(
        f"No accessible table found in schemas={schema_candidates} "
        f"tables={table_candidates}.  Check WRDS account permissions."
    )


def resolve_ibes_pair(conn):
    """Resolve actuals and detail tables as a MATCHED PAIR (same adjustment basis)."""
    for pair in IBES_TABLE_PAIRS:
        act_cols = get_columns(conn, IBES_LIBRARY, pair["actuals"])
        det_cols = get_columns(conn, IBES_LIBRARY, pair["detail"])
        if act_cols and det_cols:
            return (
                pair["basis"],
                IBES_LIBRARY, pair["actuals"], act_cols,
                pair["detail"], det_cols,
            )
    raise RuntimeError(
        "No matched IBES Actuals/Detail EPS table pair found. "
        "Tried: " + str([(p["actuals"], p["detail"]) for p in IBES_TABLE_PAIRS])
    )


def add_filter(filters, cols, col, expr):
    if col in cols:
        filters.append(expr)


def sql_date(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def tickers_sql_list(tickers) -> str:
    clean = [str(t).strip() for t in tickers if pd.notna(t) and str(t).strip()]
    return "','".join(clean)


# ── Main pull ─────────────────────────────────────────────────────────────────

def main():
    logger = setup_logging(LOG_DIR / "03_wrds_pull_ibes.log")
    ensure_dirs([RAW_IBES_DIR, RAW_LINK_DIR, TABLE_DIR])

    sample = read_csv_any(
        INT_LINK_DIR / "sample_with_crsp_permno_schemeB.csv.gz",
        force_str_cols=["gvkey", "permno"],
    )
    for c in ["target_fpedats_t", "target_fpedats_tp1",
              "forecast_window_start", "forecast_window_end"]:
        sample[c] = pd.to_datetime(sample[c], errors="coerce")

    fped_min = sample["target_fpedats_t"].min()   - pd.DateOffset(years=PULL_YEAR_PAD_BEFORE)
    fped_max = sample["target_fpedats_tp1"].max() + pd.DateOffset(years=PULL_YEAR_PAD_AFTER)
    ann_min  = sample["forecast_window_start"].min() - pd.DateOffset(months=2)
    ann_max  = sample["forecast_window_end"].max()   + pd.DateOffset(months=2)

    logger.info("Period-end pull range:  %s → %s", sql_date(fped_min), sql_date(fped_max))
    logger.info("Forecast anndats range: %s → %s", sql_date(ann_min),  sql_date(ann_max))

    try:
        import wrds
    except ImportError as exc:
        raise ImportError("Install WRDS: pip install wrds") from exc

    import os
    wrds_user = os.environ.get("WRDS_USERNAME")
    conn = wrds.Connection(wrds_username=wrds_user) if wrds_user else wrds.Connection()
    meta_rows = []

    # ── 1. IBES–CRSP historical link table ────────────────────────────────────
    link_schema, link_table, link_cols = resolve_table(
        conn, IBES_LINK_SCHEMA_CANDIDATES, [IBES_LINK_TABLE]
    )
    logger.info("IBES–CRSP link: %s.%s", link_schema, link_table)

    link_select = [c for c in ["ticker", "permno", "ncusip", "sdate", "edate", "score"]
                   if c in link_cols]

    # MUST FIX 2 (part A): validate required columns before proceeding
    if not {"ticker", "permno"}.issubset(link_select):
        raise RuntimeError(
            f"ibcrsphist is missing ticker/permno columns.  "
            f"Found: {link_cols}"
        )
    if "score" not in link_select:
        raise RuntimeError(
            f"ibcrsphist is missing the 'score' (ICLINK quality) column.  "
            f"Found: {link_cols}.  "
            "The main sample requires WRDS ICLINK score thresholds (score ≤ 2 for main, "
            "score ≤ 5 / ≤ 6 for sensitivity).  Cannot proceed without score."
        )

    q_link = f"select {', '.join(link_select)} from {link_schema}.{link_table}"
    iblink = conn.raw_sql(q_link,
                           date_cols=[c for c in ["sdate", "edate"] if c in link_select])
    write_csv_gz(iblink, RAW_LINK_DIR / "ibes_crsp_link_raw.csv.gz")
    logger.info("  → %d rows", len(iblink))

    # Force numeric types
    iblink["permno"] = pd.to_numeric(iblink["permno"], errors="coerce")
    iblink["score"]  = pd.to_numeric(iblink["score"],  errors="coerce")

    # Candidate tickers from scored WRDS links (for main + sensitivity pulls)
    sample_permnos = set(
        pd.to_numeric(sample["permno"], errors="coerce").dropna().astype(int).unique()
    )
    iblink_in_sample = iblink[iblink["permno"].isin(sample_permnos)]
    scored_tickers = (
        iblink_in_sample[iblink_in_sample["score"].le(SENSITIVITY_MAX_IBES_CRSP_SCORE2)]["ticker"]
        .dropna().unique().tolist()
    )

    n_permnos     = len(sample_permnos)
    n_tickers_le2 = len(
        iblink_in_sample[iblink_in_sample["score"].le(MAIN_MAX_IBES_CRSP_SCORE)]["ticker"]
        .dropna().unique()
    )
    logger.info("Sample permnos: %d | Scored tickers (score≤6): %d | (score≤2): %d",
                n_permnos, len(scored_tickers), n_tickers_le2)

    if not scored_tickers:
        raise RuntimeError(
            "No candidate IBES tickers found from sample permnos.  "
            "Stopping to avoid accidentally pulling the entire IBES database.  "
            "Check that step 02 (CCM Scheme B linking) produced valid permnos."
        )

    # SUGGEST 2: Pull ibtic from comp.funda NOW (before IBES downloads) so that
    # ibtic-only tickers are also included in the IBES actuals/forecasts scope.
    # This makes the appendix coverage diagnostic meaningful.
    ibtic_tickers: list[str] = []
    if USE_IBTIC_FALLBACK_APPENDIX:
        gaap_gvkeys = sample["gvkey"].dropna().unique().tolist()
        gvkeys_str  = "','".join(str(g) for g in gaap_gvkeys)
        q_ibtic = f"""
            select distinct gvkey, ibtic
            from comp.funda
            where gvkey in ('{gvkeys_str}') and ibtic is not null and ibtic != ''
        """
        try:
            ibtic_df = conn.raw_sql(q_ibtic)
            ibtic_df = ibtic_df.drop_duplicates(subset=["gvkey"], keep="first")
            write_csv_gz(ibtic_df, INT_IBES_DIR / "comp_funda_ibtic.csv.gz")
            ibtic_tickers = ibtic_df["ibtic"].dropna().unique().tolist()
            meta_rows.append({
                "dataset": "comp_funda_ibtic", "schema": "comp", "table": "funda",
                "n_rows": len(ibtic_df),
                "note": "appendix_coverage_only_NOT_in_main_sample",
            })
            logger.info("ibtic (appendix): %d gvkeys → %d unique ibtic tickers",
                        len(ibtic_df), len(ibtic_tickers))
        except Exception as e:
            logger.warning("Could not pull ibtic (appendix will be skipped): %s", e)

    # Combined ticker scope for IBES downloads:
    # scored tickers (main + sensitivity) ∪ ibtic tickers (appendix only)
    # Using a union ensures we download actuals/forecasts for all firms that
    # might appear in ANY result (main, sensitivity, or appendix).
    all_pull_tickers = list(set(scored_tickers) | set(ibtic_tickers))
    logger.info(
        "Combined IBES pull scope: %d tickers "
        "(%d scored-link + %d ibtic-only)",
        len(all_pull_tickers),
        len(scored_tickers),
        len(set(ibtic_tickers) - set(scored_tickers)),
    )

    tickers_str = tickers_sql_list(all_pull_tickers)
    meta_rows.append({
        "dataset": "ibes_crsp_link",
        "schema": link_schema, "table": link_table,
        "n_rows": len(iblink),
        "n_sample_permnos": n_permnos,
        "n_scored_tickers_le_2": n_tickers_le2,
        "n_scored_tickers_le_6": len(scored_tickers),
        "n_ibtic_tickers": len(ibtic_tickers),
        "n_total_pull_tickers": len(all_pull_tickers),
        "columns": ",".join(link_select),
    })

    # ── 2. IBES Actuals + Detail Forecasts as a MATCHED PAIR ─────────────────
    basis, act_schema, act_table, act_cols, det_table, det_cols = resolve_ibes_pair(conn)
    logger.info("IBES EPS basis: %s | actuals: %s | detail: %s",
                basis, act_table, det_table)

    if infer_ibes_basis(act_table) != infer_ibes_basis(det_table):
        raise RuntimeError(
            f"IBES table basis mismatch: actuals={act_table} vs detail={det_table}."
        )

    # ── 2a. IBES Actuals ──────────────────────────────────────────────────────
    act_select = [c for c in [
        "ticker", "cusip", "oftic", "cname", "pends", "fpedats", "anndats",
        "value", "measure", "pdicity", "curr_act", "usfirm", "actdats",
    ] if c in act_cols]
    filters = []
    add_filter(filters, act_cols, "measure",  f"measure = '{IBES_MEASURE}'")
    add_filter(filters, act_cols, "pdicity",
               f"pdicity in ('{IBES_PERIODICITY}', 'A', 'ANN')")
    add_filter(filters, act_cols, "curr_act", f"curr_act = '{IBES_CURRENCY}'")
    add_filter(filters, act_cols, "usfirm",   "usfirm = 1")
    period_col = "pends" if "pends" in act_cols else "fpedats"
    filters.append(
        f"{period_col} between '{sql_date(fped_min)}' and '{sql_date(fped_max)}'"
    )
    filters.append("value is not null")
    filters.append(f"ticker in ('{tickers_str}')")
    q_act = (f"select {', '.join(act_select)} from {IBES_LIBRARY}.{act_table} "
             f"where " + " and ".join(filters))
    actuals = conn.raw_sql(
        q_act,
        date_cols=[c for c in ["pends", "fpedats", "anndats", "actdats"]
                   if c in act_select],
    )
    write_csv_gz(actuals, RAW_IBES_DIR / "ibes_actuals_raw.csv.gz")
    logger.info("IBES actuals: %d rows, %d tickers", len(actuals), actuals["ticker"].nunique())
    meta_rows.append({
        "dataset": "ibes_actuals", "ibes_eps_basis": basis,
        "schema": act_schema, "table": act_table,
        "n_rows": len(actuals), "columns": ",".join(act_select),
    })

    # ── 2b. IBES Detail Forecasts ─────────────────────────────────────────────
    det_select = [c for c in [
        "ticker", "cusip", "oftic", "cname", "fpedats", "anndats", "revdats",
        "estimator", "analys", "broker", "value", "measure", "pdicity",
        "fpi", "usfirm", "curr", "curcode", "curr_act",
    ] if c in det_cols]
    filters = []
    add_filter(filters, det_cols, "measure", f"measure = '{IBES_MEASURE}'")
    add_filter(filters, det_cols, "pdicity",
               f"pdicity in ('{IBES_PERIODICITY}', 'A', 'ANN')")
    add_filter(filters, det_cols, "usfirm",  "usfirm = 1")
    # In ibes.det_epsus the `curr` column is NULL for virtually all individual
    # forecasts; the denominating currency is recorded in `curr_act`.
    # Using `curr = 'USD'` strictly zeroes out the pull. Accept NULL as USD
    # (usfirm = 1 already restricts to the US file) and prefer curr_act when
    # available.
    if "curr_act" in det_cols:
        filters.append(f"(curr_act = '{IBES_CURRENCY}' or curr_act is null)")
    elif "curr" in det_cols:
        filters.append(f"(curr = '{IBES_CURRENCY}' or curr is null)")
    elif "curcode" in det_cols:
        filters.append(f"(curcode = '{IBES_CURRENCY}' or curcode is null)")
    if "fpi" in det_cols:
        filters.append(f"cast(fpi as varchar) = '{IBES_FPI_ANNUAL}'")
    filters.append(
        f"fpedats between '{sql_date(fped_min)}' and '{sql_date(fped_max)}'"
    )
    filters.append(
        f"anndats between '{sql_date(ann_min)}' and '{sql_date(ann_max)}'"
    )
    filters.append("value is not null")
    filters.append(f"ticker in ('{tickers_str}')")
    q_det = (f"select {', '.join(det_select)} from {IBES_LIBRARY}.{det_table} "
             f"where " + " and ".join(filters))
    forecasts = conn.raw_sql(
        q_det,
        date_cols=[c for c in ["fpedats", "anndats", "revdats"] if c in det_select],
    )
    write_csv_gz(forecasts, RAW_IBES_DIR / "ibes_detail_forecasts_raw.csv.gz")
    logger.info("IBES detail: %d rows, %d tickers",
                len(forecasts), forecasts["ticker"].nunique())
    meta_rows.append({
        "dataset": "ibes_detail_forecasts", "ibes_eps_basis": basis,
        "schema": IBES_LIBRARY, "table": det_table,
        "n_rows": len(forecasts), "columns": ",".join(det_select),
    })

    write_csv(pd.DataFrame(meta_rows), TABLE_DIR / "02_wrds_pull_metadata.csv")
    logger.info("WRDS pull complete.  Raw data in 03_raw/")
    conn.close()


if __name__ == "__main__":
    main()
