"""
09_quality_checks.py — 13-point quality checklist.

Changes from v3:
  - Must Fix 1: Check 11 now tests for ALL Compustat earnings columns
    (ni, ib, ibc, oiadp, epspi, epspx, epsfi, epsfx, ni_lead, ni_lag, oibdp),
    not just 'ni'.  This matches the drop_compustat_earnings_columns() list in
    06_construct_auc_sample.py and properly enforces the "no Compustat earnings"
    requirement in the final output file.

Run AFTER 06_construct_auc_sample.py and 07_compute_auc.py.
Writes 05_output/tables/07_quality_checks.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import *
from src.common import setup_logging, ensure_dirs, read_csv_any, write_csv

# Keep in sync with COMPUSTAT_EARNINGS_COLS in 06_construct_auc_sample.py
COMPUSTAT_EARNINGS_COLS = {
    "ni", "ni_lead", "ni_lag",
    "ib", "ibc", "oiadp", "oibdp",
    "epspi", "epspx", "epsfi", "epsfx",
}


def check(name: str, passed: bool, value, expected, comment: str = "") -> dict:
    return {
        "check_name": name,
        "status":     "PASS" if passed else "FAIL",
        "value":      str(value),
        "expected":   str(expected),
        "comment":    comment,
    }


def main():
    logger = setup_logging(LOG_DIR / "09_quality_checks.log")
    ensure_dirs([TABLE_DIR])

    auc_path = INT_MERGE_DIR / "firm_year_auc_input_main_score_le_2.csv.gz"
    if not auc_path.exists():
        logger.error("Main AUC file not found.  Run 06_construct_auc_sample.py first.")
        return

    df   = read_csv_any(auc_path, force_str_cols=["gvkey"])
    meta = read_csv_any(TABLE_DIR / "02_wrds_pull_metadata.csv") \
           if (TABLE_DIR / "02_wrds_pull_metadata.csv").exists() else pd.DataFrame()

    results = []

    # ── 01: No duplicate (gvkey, fyear) ──────────────────────────────────────
    dup_count = int(df.duplicated(subset=["gvkey", "fyear"]).sum())
    results.append(check(
        "01_no_duplicate_gvkey_fyear", dup_count == 0, dup_count, 0,
        "Each firm-year must appear exactly once in the AUC sample"
    ))

    # ── 02: y_true only in {0, 1} ─────────────────────────────────────────────
    if "y_true" in df.columns:
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
        unique_yt = sorted(df["y_true"].dropna().unique().tolist())
        results.append(check(
            "02_y_true_binary", set(unique_yt).issubset({0, 1}),
            unique_yt, "[0, 1]",
            "y_true must be 0 or 1 only"
        ))

    # ── 03: analyst_score (score) in [0, 1] ───────────────────────────────────
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        out_of_range = int(((df["score"] < 0) | (df["score"] > 1)).sum())
        results.append(check(
            "03_score_in_unit_interval", out_of_range == 0, out_of_range, 0,
            "analyst_score is a proportion; must be in [0, 1]"
        ))

    # ── 04: actual_eps_t and actual_eps_tp1 not missing ───────────────────────
    for col in ["actual_eps_t", "actual_eps_tp1"]:
        if col in df.columns:
            n_miss = int(df[col].isna().sum())
            results.append(check(
                f"04_no_missing_{col}", n_miss == 0, n_miss, 0,
                "AUC sample must only contain rows where both EPS values are present"
            ))

    # ── 05: forecast_anndats within forecast window ───────────────────────────
    fcst_path = INT_MERGE_DIR / "forecast_level_directions_main_score_le_2.csv.gz"
    fdf = None
    if fcst_path.exists():
        fdf = read_csv_any(fcst_path)
        for c in ["forecast_anndats", "forecast_window_start", "forecast_window_end"]:
            if c in fdf.columns:
                fdf[c] = pd.to_datetime(fdf[c], errors="coerce")
        if all(c in fdf.columns for c in ["forecast_anndats",
                                            "forecast_window_start",
                                            "forecast_window_end"]):
            out_of_window = int(
                (~fdf["forecast_anndats"].between(
                    fdf["forecast_window_start"],
                    fdf["forecast_window_end"]
                )).sum()
            )
            results.append(check(
                "05_forecast_anndats_in_window", out_of_window == 0,
                out_of_window, 0,
                "All forecasts must fall within the designated forecast window"
            ))

    # ── 06: forecast fpedats within tolerance ─────────────────────────────────
    if fdf is not None and "forecast_fpedats_diff_days" in fdf.columns:
        beyond_tol = int(
            (pd.to_numeric(fdf["forecast_fpedats_diff_days"], errors="coerce")
               .gt(PENDS_MATCH_TOLERANCE_DAYS)
               .sum())
        )
        results.append(check(
            "06_forecast_fpedats_within_tolerance", beyond_tol == 0, beyond_tol, 0,
            f"All forecast fpedats within ±{PENDS_MATCH_TOLERANCE_DAYS} days of target"
        ))

    # ── 07: main sample uses Route A only (no ibtic) ──────────────────────────
    if "ibes_link_source" in df.columns:
        non_route_a = int(
            (~df["ibes_link_source"].str.contains("route_a", na=False)).sum()
        )
        results.append(check(
            "07_main_sample_route_a_only", non_route_a == 0, non_route_a, 0,
            "Main sample must use WRDS ibcrsphist (route_a) only; ibtic is appendix only"
        ))

    # ── 08: IBES–CRSP link score ≤ 2 in main sample ───────────────────────────
    # Uses the renamed 'ibes_crsp_score' column (not 'score', which is analyst_score).
    # Hardened: PASS only when (a) column exists, (b) zero missing values,
    # and (c) max score ≤ 2.  A missing max would silently pass in some pandas
    # versions, so we check n_missing explicitly.
    if "ibes_crsp_score" in df.columns:
        df["ibes_crsp_score"] = pd.to_numeric(df["ibes_crsp_score"], errors="coerce")
        n_missing_score = int(df["ibes_crsp_score"].isna().sum())
        max_link_score  = df["ibes_crsp_score"].max()   # NaN if all missing
        results.append(check(
            "08_ibes_crsp_link_score_le_2",
            n_missing_score == 0 and (not pd.isna(max_link_score))
            and max_link_score <= MAIN_MAX_IBES_CRSP_SCORE,
            f"missing={n_missing_score}, max={max_link_score}",
            f"missing=0, max≤{MAIN_MAX_IBES_CRSP_SCORE}",
            "WRDS ICLINK score: 0 = best, 6 = worst; "
            "main sample must have non-missing score ≤ 2 for every observation"
        ))
    else:
        results.append(check(
            "08_ibes_crsp_link_score_le_2",
            False,
            "ibes_crsp_score column absent from AUC input",
            f"non-missing ibes_crsp_score with max ≤ {MAIN_MAX_IBES_CRSP_SCORE}",
            "Main sample must retain the actual WRDS ICLINK score per observation; "
            "re-run 06_construct_auc_sample.py"
        ))

    # ── 09: actual_anndats_t ≤ formation_date (no look-ahead) ─────────────────
    for col in ["actual_anndats_t", "formation_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "actual_anndats_t" in df.columns and "formation_date" in df.columns:
        lookahead = int((df["actual_anndats_t"] > df["formation_date"]).sum())
        missing   = int(df["actual_anndats_t"].isna().sum())
        results.append(check(
            "09_no_lookahead_in_actual_eps_t",
            lookahead == 0 and missing == 0,
            f"lookahead={lookahead}, missing={missing}", "0, 0",
            "EPS_t announced before formation; missing anndats excluded from main"
        ))

    # ── 10: Both classes present ───────────────────────────────────────────────
    if "y_true" in df.columns:
        classes = set(df["y_true"].dropna().astype(int).unique())
        results.append(check(
            "10_both_classes_present_for_AUC",
            {0, 1}.issubset(classes), sorted(classes), "[0, 1]",
            "AUC requires both earnings-increase (y=1) and non-increase (y=0) observations"
        ))

    # ── 11: No Compustat earnings columns in AUC input ────────────────────────
    # MUST FIX 1: Check ALL earnings-related Compustat columns, not just 'ni'.
    # This matches the COMPUSTAT_EARNINGS_COLS set in 06_construct_auc_sample.py.
    present_comp_cols = sorted(
        c for c in df.columns if c.lower() in COMPUSTAT_EARNINGS_COLS
    )
    results.append(check(
        "11_no_compustat_earnings_columns_in_auc_input",
        len(present_comp_cols) == 0,
        present_comp_cols,
        "[]",
        "Final AUC input must not carry Compustat earnings columns "
        "(ni, ib, epspi, etc.); all earnings directions use IBES Actuals only"
    ))

    # ── 12: IBES EPS adjustment basis is consistent ────────────────────────────
    if not meta.empty and "ibes_eps_basis" in meta.columns:
        bases = meta[meta["dataset"].isin(["ibes_actuals", "ibes_detail_forecasts"])][
            "ibes_eps_basis"
        ]
        consistent = bases.nunique() == 1
        val = bases.iloc[0] if len(bases) > 0 else "N/A"
        results.append(check(
            "12_ibes_eps_basis_consistent", consistent, val,
            "same basis for both tables",
            "Actuals and forecasts must use the same IBES EPS split-adjustment basis"
        ))
    else:
        results.append(check(
            "12_ibes_eps_basis_consistent", False,
            "metadata not found", "run step 03 first",
            "Check ibes_eps_basis in 02_wrds_pull_metadata.csv"
        ))

    # ── 13: AUC in sensible range ──────────────────────────────────────────────
    summ_path = TABLE_DIR / "06_auc_main_and_sensitivity_summary.csv"
    if summ_path.exists():
        auc_df = read_csv_any(summ_path)
        main_r = auc_df[auc_df["sample_label"].str.contains("main", na=False)]
        if len(main_r) > 0:
            auc_val = float(main_r["auc"].iloc[0])
            results.append(check(
                "13_auc_in_sensible_range",
                0.45 <= auc_val <= 0.85,
                f"{auc_val:.4f}", "between 0.45 and 0.85",
                "AUC outside [0.45, 0.85] is unusual for analyst earnings forecast studies"
            ))

    # ── Write results ─────────────────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    write_csv(out_df, TABLE_DIR / "07_quality_checks.csv")

    n_pass = int((out_df["status"] == "PASS").sum())
    n_fail = int((out_df["status"] == "FAIL").sum())
    logger.info("Quality checks: %d PASS, %d FAIL", n_pass, n_fail)

    if n_fail > 0:
        fails = out_df[out_df["status"] == "FAIL"][["check_name", "value", "comment"]]
        logger.warning(
            "FAILED checks — investigate before sending results to professor:\n%s",
            fails.to_string(index=False)
        )
    else:
        logger.info("All %d quality checks PASSED.", n_pass)

    # Print a concise summary to stdout for quick review
    print("\n" + "="*60)
    print(f"QUALITY CHECKS:  {n_pass} PASS  /  {n_fail} FAIL")
    if n_fail > 0:
        print("FAILED:")
        for _, row in out_df[out_df["status"] == "FAIL"].iterrows():
            print(f"  - {row['check_name']}: got {row['value']}, expected {row['expected']}")
    print("See: 05_output/tables/07_quality_checks.csv")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
