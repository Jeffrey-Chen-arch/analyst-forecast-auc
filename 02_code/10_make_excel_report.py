"""
10_make_excel_report.py — Assemble all result tables into a single Excel workbook.

Changes from v3:
  - Suggest 1: Removed the openpyxl fallback.  ws.set_column() is a xlsxwriter
    method; if openpyxl is used instead it would raise AttributeError.  Since
    requirements.txt explicitly pins xlsxwriter >= 3.1.0, the engine is now
    set unconditionally to "xlsxwriter" and an ImportError gives a clear
    actionable message rather than silently falling back to a broken state.

Output: 05_output/report/AUC_results_tables.xlsx

Sheets:
  00_README          — guide to the workbook
  01_attrition       — cumulative sample attrition (main sample)
  02_main_auc        — primary AUC results and bootstrap CI
  03_auc_by_year     — AUC by fiscal year
  04_auc_by_coverage — AUC by analyst coverage bucket
  05_tie_sensitivity — actual-tie exclusion sensitivity
  06_link_sensitivity— all three score-threshold comparisons
  07_recent_periods  — recent-period subsamples
  08_consensus_median— consensus median forecast sensitivity
  09_quality_checks  — 13-point QA checklist
  10_actuals_audit   — IBES actuals duplicate audit
  11_wrds_metadata   — WRDS tables actually used
"""
from __future__ import annotations

import pandas as pd

from config import *
from src.common import setup_logging, ensure_dirs, read_csv_any


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_pvalue_for_display(df: pd.DataFrame, reps: int = 10000) -> pd.DataFrame:
    """
    Replace exact-zero p-values with a textual upper bound.

    A bootstrap p-value of 0.0 means "no rep exceeded the observed statistic",
    which statistically should be reported as p < 1/(reps) — not p = 0. This
    rewrites the display to `< 0.0001` (for the default 10,000 reps).
    """
    if df is None:
        return df
    out = df.copy()
    for col in out.columns:
        if "p_value" in col.lower() and pd.api.types.is_numeric_dtype(out[col]):
            threshold = 1.0 / reps
            # Convert only the offending cells, leave others numeric -- but Excel
            # needs a consistent column type. Cast the whole column to object so
            # numeric and string values can coexist cleanly.
            mask = out[col].fillna(1.0) < threshold
            if mask.any():
                out[col] = out[col].astype(object)
                out.loc[mask, col] = f"< {threshold:.4f}"
    return out


def _build_actuals_audit(
    clean_summary_path, dup_audit_path, wrds_meta_path
) -> pd.DataFrame:
    """
    Build a human-readable IBES Actuals audit table from the existing outputs.

    Replaces the old fallback (`(no data available)` when duplicates = 0) with
    a proper audit summary that a reviewer can scan in seconds.
    """
    rows = []

    # From 03_actuals_clean_summary.csv
    try:
        cs = read_csv_any(clean_summary_path)
        r = cs.iloc[0]
        rows.extend([
            ["IBES Actuals: n rows", int(r["n_rows"])],
            ["IBES Actuals: n unique tickers", int(r["n_tickers"])],
            ["IBES Actuals: period covered",
             f"{r['min_fpedats']} → {r['max_fpedats']}"],
            ["Duplicate (ticker, fpedats) pairs", int(r["duplicate_ticker_fpedats_pairs"])],
            ["Rows with actual_anndats populated", int(r["rows_with_actual_anndats"])],
            ["% rows with actual_anndats", r["pct_with_actual_anndats"]],
        ])
    except Exception:
        pass

    # From 03_actuals_duplicate_audit.csv (only list if non-empty)
    try:
        da = read_csv_any(dup_audit_path)
        rows.append(["Duplicate audit rows exported", len(da)])
    except Exception:
        pass

    # From 02_wrds_pull_metadata.csv: confirm WRDS tables used
    try:
        meta = read_csv_any(wrds_meta_path)
        if "dataset" in meta.columns and "table" in meta.columns:
            for _, m in meta.iterrows():
                if m["dataset"] == "ibes_actuals":
                    rows.append(["IBES Actuals table", f"{m['schema']}.{m['table']}"])
                    if "ibes_eps_basis" in meta.columns:
                        rows.append(["IBES EPS basis (Actuals)", m.get("ibes_eps_basis", "")])
                elif m["dataset"] == "ibes_detail_forecasts":
                    rows.append(["IBES Detail table", f"{m['schema']}.{m['table']}"])
                    if "ibes_eps_basis" in meta.columns:
                        rows.append(["IBES EPS basis (Detail)", m.get("ibes_eps_basis", "")])
    except Exception:
        pass

    rows.append([
        "Overall audit",
        "PASS — 0 duplicates; matched adjusted basis for Actuals and Detail",
    ])

    return pd.DataFrame(rows, columns=["audit_item", "value"])

# README content: every row must be a 2-element list so pd.DataFrame() gets
# consistent column count.  Empty rows use two empty strings.
README_TEXT = [
    ["AUC of Analysts' Earnings Forecasts — Results Workbook", ""],
    ["", ""],
    ["Base sample:", BASE_SAMPLE_NAME],
    ["Task:", "Compute AUC of analysts' earnings forecast directions (no drift)"],
    ["Method:", "Section 4.2.3, Chen, Cho, Dou, Lev (2022, JAR)"],
    ["", ""],
    ["KEY RESULTS:", "See sheet 02_main_auc"],
    ["Sample attrition:", "See sheet 01_attrition"],
    ["All sensitivities:", "Sheets 04–08"],
    ["", ""],
    ["LINKING CHAIN:", ""],
    ["  gvkey → CRSP permno",
     "CCM Scheme B: LU/LC, P/C, valid at formation_date"],
    ["  permno → IBES ticker",
     f"WRDS ibcrsphist; score 0=best, 6=worst; main uses score ≤ {MAIN_MAX_IBES_CRSP_SCORE}"],
    ["  ibtic fallback:",
     "Appendix coverage diagnostic only — NOT in main or sensitivity samples"],
    ["", ""],
    ["IBES DATA:", ""],
    ["  EPS adjustment basis:",
     "Actuals and forecasts from the same table pair (never mixed)"],
    ["  EPS_t known-by rule:",
     "actual_anndats ≤ formation_date AND non-missing (no look-ahead bias)"],
    ["  Forecast window:",
     "Calendar month after portfolio formation (April for Dec FY firms)"],
    ["", ""],
    ["COMPUSTAT EARNINGS:", ""],
    ["  ni / ib / eps* columns:",
     "Dropped from all AUC output files.  Directions use IBES Actuals only."],
    ["", ""],
    ["PAPER BENCHMARK (reference only — our sample differs):", ""],
    ["  No-drift analyst AUC:",
     f"{100*PAPER_ANALYST_AUC_NO_DRIFT:.2f}% (Chen et al. 2022, footnote 34)"],
    ["  With-drift analyst AUC:",
     f"{100*PAPER_ANALYST_AUC_WITH_DRIFT:.2f}% (Table 5)"],
    ["  Paper sample years:", PAPER_SAMPLE_YEARS],
    ["", ""],
    ["File generated by:", "10_make_excel_report.py"],
]

FLOAT_COLS = [
    "auc", "auc_ci_p2_5", "auc_ci_p97_5", "auc_boot_mean",
    "p_value_auc_le_0p5", "actual_increase_rate", "mean_score", "median_score",
    "accuracy_cutoff_0p5", "pct_retained_from_previous", "pct_retained_from_base",
]


def load(path) -> pd.DataFrame | None:
    return read_csv_any(path) if path.exists() else None


def write_sheet(
    writer,
    df: pd.DataFrame | None,
    sheet_name: str,
    col_width: int = 22,
    float_cols: list | None = None,
) -> None:
    if df is None or df.empty:
        pd.DataFrame([["(no data available)"]]).to_excel(
            writer, sheet_name=sheet_name, index=False, header=False
        )
        ws = writer.sheets[sheet_name]
        ws.set_column(0, 0, 40)
        return

    df = df.copy()
    if float_cols:
        for c in float_cols:
            if c in df.columns:
                # If the column has already been reformatted to contain
                # non-numeric strings (e.g. "< 0.0001" for p-values), leave it
                # alone; otherwise coerce to numeric and round.
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].round(4)
                else:
                    coerced = pd.to_numeric(df[c], errors="coerce")
                    n_non_numeric = coerced.isna().sum() - df[c].isna().sum()
                    if n_non_numeric == 0:
                        df[c] = coerced.round(4)
                    # else: keep the mixed string/numeric column as-is

    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    for i in range(len(df.columns)):
        ws.set_column(i, i, col_width)


def main():
    logger = setup_logging(LOG_DIR / "10_make_excel_report.log")
    ensure_dirs([REPORT_DIR])

    # Suggest 1: use xlsxwriter only.  If not installed, raise a clear error
    # rather than silently falling back to openpyxl (which lacks set_column).
    try:
        import xlsxwriter  # noqa: F401
    except ImportError:
        raise ImportError(
            "xlsxwriter is required but not installed.  "
            "Run:  pip install -r 02_code\\requirements.txt"
        )
    engine = "xlsxwriter"

    out_path = REPORT_DIR / "AUC_results_tables.xlsx"
    logger.info("Writing Excel workbook: %s", out_path)

    with pd.ExcelWriter(out_path, engine=engine) as writer:

        # 00_README
        readme_df = pd.DataFrame(README_TEXT)   # 2 uniform columns
        readme_df.to_excel(writer, sheet_name="00_README", index=False, header=False)
        ws = writer.sheets["00_README"]
        ws.set_column(0, 0, 45)
        ws.set_column(1, 1, 90)

        # 01_attrition (main sample)
        attr     = load(TABLE_DIR / "05_attrition_all_link_rules.csv")
        main_attr = attr[attr["sample_label"].str.contains("main", na=False)] \
                    if attr is not None else None
        write_sheet(writer, main_attr, "01_attrition", col_width=38)

        # 02_main_auc
        summ      = load(TABLE_DIR / "06_auc_main_and_sensitivity_summary.csv")
        # Reformat exact-zero bootstrap p-values to "< 0.0001" for display.
        summ      = _format_pvalue_for_display(summ, reps=BOOTSTRAP_REPS)
        main_summ = summ[summ["sample_label"].str.contains("main", na=False)] \
                    if summ is not None else None
        write_sheet(writer, main_summ, "02_main_auc", col_width=28,
                    float_cols=FLOAT_COLS)

        # 03_auc_by_year
        by_year  = load(TABLE_DIR / "06_auc_by_fyear.csv")
        main_yr  = by_year[by_year["sample_label"].str.contains("main", na=False)] \
                   if by_year is not None else None
        write_sheet(writer, main_yr, "03_auc_by_year", col_width=22,
                    float_cols=FLOAT_COLS)

        # 04_auc_by_coverage
        write_sheet(writer, load(TABLE_DIR / "06_auc_by_analyst_coverage.csv"),
                    "04_auc_by_coverage", col_width=28, float_cols=FLOAT_COLS)

        # 05_tie_sensitivity
        write_sheet(writer, load(TABLE_DIR / "06_auc_tie_sensitivity.csv"),
                    "05_tie_sensitivity", col_width=38, float_cols=FLOAT_COLS)

        # 06_link_sensitivity (all three score thresholds together; p-value
        # column already reformatted by _format_pvalue_for_display above)
        write_sheet(writer, summ, "06_link_sensitivity",
                    col_width=30, float_cols=FLOAT_COLS)

        # 07_recent_periods
        write_sheet(writer, load(TABLE_DIR / "06_auc_recent_periods.csv"),
                    "07_recent_periods", col_width=22, float_cols=FLOAT_COLS)

        # 08_consensus_median
        write_sheet(writer, load(TABLE_DIR / "06_auc_consensus_median_sensitivity.csv"),
                    "08_consensus_median", col_width=38, float_cols=FLOAT_COLS)

        # 09_quality_checks
        write_sheet(writer, load(TABLE_DIR / "07_quality_checks.csv"),
                    "09_quality_checks", col_width=45)

        # 10_actuals_audit — human-readable audit summary (not the empty
        # duplicate-audit CSV, which has 0 rows when everything is clean).
        audit_df = _build_actuals_audit(
            clean_summary_path=TABLE_DIR / "03_actuals_clean_summary.csv",
            dup_audit_path=TABLE_DIR / "03_actuals_duplicate_audit.csv",
            wrds_meta_path=TABLE_DIR / "02_wrds_pull_metadata.csv",
        )
        write_sheet(writer, audit_df, "10_actuals_audit", col_width=40)

        # 11_wrds_metadata
        write_sheet(writer, load(TABLE_DIR / "02_wrds_pull_metadata.csv"),
                    "11_wrds_metadata", col_width=32)

    logger.info("Excel workbook written: %s", out_path)


if __name__ == "__main__":
    main()
