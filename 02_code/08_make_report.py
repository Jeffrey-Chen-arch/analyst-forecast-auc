from __future__ import annotations

import pandas as pd

from config import *
from src.common import setup_logging, ensure_dirs, read_csv_any


def fmt_pct(x, d=2):
    try: return f"{100*float(x):.{d}f}%"
    except: return "N/A"

def fmt_f(x, d=4):
    try: return f"{float(x):.{d}f}"
    except: return "N/A"

def fmt_n(x):
    try: return f"{int(x):,}"
    except: return "N/A"

def df_to_md(df, max_rows=50):
    if df is None or df.empty: return "*(no data)*"
    try: return df.head(max_rows).to_markdown(index=False)
    except: return df.head(max_rows).to_string(index=False)

def load(path):
    return read_csv_any(path) if path.exists() else None


def main():
    logger = setup_logging(LOG_DIR / "08_make_report.log")
    ensure_dirs([REPORT_DIR])

    base    = load(TABLE_DIR / "00_base_sample_summary.csv")
    attr    = load(TABLE_DIR / "05_attrition_all_link_rules.csv")
    meta    = load(TABLE_DIR / "02_wrds_pull_metadata.csv")
    summ    = load(TABLE_DIR / "06_auc_main_and_sensitivity_summary.csv")
    by_year = load(TABLE_DIR / "06_auc_by_fyear.csv")
    by_cov  = load(TABLE_DIR / "06_auc_by_analyst_coverage.csv")
    ties    = load(TABLE_DIR / "06_auc_tie_sensitivity.csv")
    recent  = load(TABLE_DIR / "06_auc_recent_periods.csv")
    med_s   = load(TABLE_DIR / "06_auc_consensus_median_sensitivity.csv")
    dup_act = load(TABLE_DIR / "03_actuals_duplicate_audit.csv")
    act_sum = load(TABLE_DIR / "03_actuals_clean_summary.csv")
    fct_sum = load(TABLE_DIR / "04_detail_forecasts_clean_summary.csv")
    qa      = load(TABLE_DIR / "07_quality_checks.csv")

    main_row = summ[summ["sample_label"].str.contains("main", na=False)].iloc[0] \
               if summ is not None and len(summ) > 0 else None

    R = []

    R.append("# Analysts' Earnings Forecast AUC — Results Report\n")
    R.append(f"*Sample: {BASE_SAMPLE_NAME}  |  Bootstrap: {BOOTSTRAP_REPS:,} reps, "
             f"firm-clustered={BOOTSTRAP_CLUSTER_BY_FIRM}*\n")

    # ── 1. Task Interpretation ────────────────────────────────────────────────
    R.append("## 1. Task Interpretation\n")
    R.append(
        "This report computes the AUC of professional analysts' earnings forecast directions "
        "following Section 4.2.3 of Chen, Cho, Dou, and Lev (2022, *JAR*). "
        "It uses the March 2026 restarted GAAP replication sample as the base, "
        "merges through CRSP to IBES, and constructs raw (no-drift) earnings directions.\n\n"
        "**Key design choices per Professor Dou's April 7, 2026 instructions:**\n\n"
        "- **No drift adjustment** — raw EPS changes only\n"
        "- **IBES Actuals only** for actual EPS — Compustat earnings not used\n"
        "- **WRDS IBES–CRSP historical link table** — no unscored static ibtic in main sample\n"
        "- **Individual analyst forecasts** from IBES Detail History\n"
        "- **Firm-clustered bootstrap** for inference\n"
    )

    # ── 2. Methodology ────────────────────────────────────────────────────────
    R.append("## 2. Methodology\n")
    R.append(
        "### 2.1 Variable Definitions\n\n"
        "| Variable | Formula |\n|---|---|\n"
        "| `actual_increase_raw` | = 1 if IBES Actual EPS\\(_{t+1}\\) > IBES Actual EPS\\(_t\\), else 0 |\n"
        "| `forecast_increase_raw` | = 1 if Analyst Forecast\\(_{t+1}\\) > IBES Actual EPS\\(_t\\), else 0 |\n"
        "| `analyst_score` | Proportion of analysts predicting an EPS increase (∈ [0, 1]) |\n"
        "| `y_true` | = `actual_increase_raw` |\n"
        "| `score` | = `analyst_score` |\n\n"

        "### 2.2 Timing (December FY firms — entire sample)\n\n"
        "| Date | Description |\n|---|---|\n"
        "| Dec 31, year *t* | Fiscal year end (`datadate`) |\n"
        f"| Mar 31, year *t+1* | Portfolio formation (FYE + {FORMATION_MONTHS_AFTER_FYE} months) |\n"
        "| **Apr 1–Apr 30, year *t+1*** | **Forecast window** — forecasts used for analyst score |\n"
        "| Dec 31, year *t+1* | EPS_{t+1} fiscal year end |\n\n"
        "**EPS_t known-by rule:** IBES actual EPS for FY *t* must have a non-missing "
        "`actual_anndats` that is ≤ portfolio formation date. "
        "This prevents look-ahead bias.\n\n"

        "### 2.3 Linking Chain\n\n"
        "```\n"
        "Compustat gvkey\n"
        "  → CRSP permno  [CCM Scheme B: LU/LC, P/C, valid at formation_date]\n"
        "  → IBES ticker  [WRDS ibcrsphist; ICLINK score 0=best, 6=worst]\n"
        f"    Main sample:  score ≤ {MAIN_MAX_IBES_CRSP_SCORE}\n"
        f"    Sensitivity:  score ≤ {SENSITIVITY_MAX_IBES_CRSP_SCORE}\n"
        f"    Sensitivity:  score ≤ {SENSITIVITY_MAX_IBES_CRSP_SCORE2} (all scored)\n"
        "    Appendix:     + Compustat ibtic (unscored, coverage diagnostic only)\n"
        "```\n\n"
        "**Note on WRDS ICLINK score:** The score ranges from 0 (best match) to 6 "
        "(worst match). Lower scores indicate better link quality. The main sample uses "
        "score ≤ 2 to include only the highest-quality links.\n\n"

        "### 2.4 IBES EPS Adjustment Basis\n\n"
        "Analyst forecasts and IBES Actuals are pulled from the **same adjustment-family "
        "table pair** (either adjusted `act_epsus`/`det_epsus`, or unadjusted "
        "`actu_epsus`/`detu_epsus`). The tables are never mixed, preventing "
        "split-adjustment basis mismatches in EPS comparisons.\n\n"

        "### 2.5 AUC\n\n"
        "AUC = Pr(analyst score for earnings-increase firm > analyst score for non-increase firm). "
        "AUC = 0.50 is the random-guess baseline. "
        f"Inference: firm-clustered bootstrap ({BOOTSTRAP_REPS:,} reps).\n"
    )

    # ── 3. Data Sources ───────────────────────────────────────────────────────
    R.append("## 3. Data Sources\n")
    if meta is not None:
        R.append(df_to_md(meta))
    R.append("\n")

    # ── 4. Base Sample ────────────────────────────────────────────────────────
    R.append("## 4. Base Sample\n")
    if base is not None:
        R.append(df_to_md(base))
    R.append("\n")

    # ── 5. Sample Attrition ───────────────────────────────────────────────────
    R.append("## 5. Sample Attrition\n")
    R.append("Each step is cumulative. `pct_retained_from_previous` = fraction of the "
             "prior step's observations that survive this step.\n\n")
    if attr is not None:
        main_attr = attr[attr["sample_label"].str.contains("main", na=False)]
        R.append(df_to_md(main_attr, max_rows=20))
    R.append("\n")

    # ── 6. IBES Data Quality ──────────────────────────────────────────────────
    R.append("## 6. IBES Data Quality\n")
    R.append("### 6.1 Actuals coverage\n")
    if act_sum is not None: R.append(df_to_md(act_sum))
    R.append("\n### 6.2 Duplicate actuals\n")
    if dup_act is not None:
        R.append(f"- {len(dup_act):,} (ticker, fpedats) pairs have more than one row. "
                 "Resolved by taking the row with the latest `actual_anndats`.\n")
    R.append("\n### 6.3 Detail forecasts coverage\n")
    if fct_sum is not None: R.append(df_to_md(fct_sum))
    R.append("\n")

    # ── 7. Compustat Earnings Audit ───────────────────────────────────────────
    R.append("## 7. Compustat Earnings Audit\n")
    R.append(
        "| Item | Status |\n|---|---|\n"
        "| Base sample constructed from Compustat | Yes |\n"
        "| CRSP link via CCM (Compustat–CRSP) | Yes |\n"
        "| Compustat `ni` used in AUC actual direction | **No** |\n"
        "| Compustat EPS used in AUC actual direction | **No** |\n"
        "| IBES Actual EPS_t used | **Yes** |\n"
        "| IBES Actual EPS_{t+1} used | **Yes** |\n"
        "| Analyst forecast EPS from IBES Detail | **Yes** |\n\n"
        "> Compustat earnings numbers are used **only** to construct the replication "
        "sample (balance sheet / income statement filters for Panel B). "
        "All earnings-change directions are defined using IBES Actuals exclusively.\n"
    )

    # ── 8. Main AUC Results ───────────────────────────────────────────────────
    R.append("## 8. Main AUC Results\n")
    if main_row is not None:
        R.append(
            f"| Metric | Value |\n|---|---|\n"
            f"| N firm-years | {fmt_n(main_row['n_obs'])} |\n"
            f"| N unique firms | {fmt_n(main_row['n_firms'])} |\n"
            f"| N fiscal years | {fmt_n(main_row['n_years'])} |\n"
            f"| Earnings increases | {fmt_n(main_row['n_actual_increase'])} "
            f"({fmt_pct(main_row['actual_increase_rate'])}) |\n"
            f"| Mean analyst score | {fmt_f(main_row['mean_score'])} |\n"
            f"| Median analyst score | {fmt_f(main_row['median_score'])} |\n"
            f"| **AUC (main, score ≤ 2, no drift)** | **{fmt_pct(main_row['auc'])}** |\n"
            f"| Bootstrap 95% CI | [{fmt_pct(main_row['auc_ci_p2_5'])}, "
            f"{fmt_pct(main_row['auc_ci_p97_5'])}] |\n"
            f"| p-value (AUC ≤ 0.50) | {fmt_f(main_row['p_value_auc_le_0p5'])} |\n"
            f"| Accuracy at 0.5 cutoff | {fmt_pct(main_row['accuracy_cutoff_0p5'])} |\n"
        )
    R.append("\n")

    # ── 9. Comparison with Paper ──────────────────────────────────────────────
    R.append("## 9. Comparison with Paper Benchmark\n")
    R.append(
        f"| | AUC |\n|---|---|\n"
        f"| **This analysis (main, no drift)** | "
        f"**{fmt_pct(main_row['auc']) if main_row is not None else 'N/A'}** |\n"
        f"| Chen et al. (2022), analysts, no drift (fn. 34) | {fmt_pct(PAPER_ANALYST_AUC_NO_DRIFT)} |\n"
        f"| Chen et al. (2022), analysts, with drift (Table 5) | {fmt_pct(PAPER_ANALYST_AUC_WITH_DRIFT)} |\n"
        f"| Random guess baseline | {fmt_pct(PAPER_RANDOM_BASELINE)} |\n\n"
        f"> The paper's {fmt_pct(PAPER_ANALYST_AUC_NO_DRIFT)} (no drift) is a reference point, "
        "not a target. Differences reflect sample period (this study: 2005–2023 vs. "
        f"paper: {PAPER_SAMPLE_YEARS}) and sample definition (March replication sample "
        "vs. paper's XBRL sample).\n"
    )

    # ── 10. Sensitivity Analyses ──────────────────────────────────────────────
    R.append("## 10. Sensitivity Analyses\n")
    R.append("### 10.1 IBES link score threshold\n")
    if summ is not None:
        cols = [c for c in ["sample_label","n_obs","n_firms","auc","auc_ci_p2_5",
                             "auc_ci_p97_5","p_value_auc_le_0p5"] if c in summ.columns]
        R.append(df_to_md(summ[cols]))
    R.append("\n")

    R.append("### 10.2 Actual-tie exclusion sensitivity\n")
    if ties is not None: R.append(df_to_md(ties))
    R.append("\n")

    R.append("### 10.3 AUC by analyst coverage\n")
    if by_cov is not None: R.append(df_to_md(by_cov))
    R.append("\n")

    R.append("### 10.4 Consensus median score sensitivity\n")
    R.append("*Uses 1[median consensus forecast > Actual EPS_t] instead of proportion score.*\n\n")
    if med_s is not None: R.append(df_to_md(med_s))
    R.append("\n")

    R.append("### 10.5 Recent-period subsamples\n")
    if recent is not None:
        main_recent = recent[recent["sample_label"].str.contains("main", na=False)]
        R.append(df_to_md(main_recent))
    R.append("\n")

    # ── 11. By Fiscal Year ────────────────────────────────────────────────────
    R.append("## 11. AUC by Fiscal Year\n")
    if by_year is not None:
        main_yr = by_year[by_year["sample_label"].str.contains("main", na=False)]
        R.append(df_to_md(main_yr, max_rows=25))
    R.append("\n")

    # ── 12. Quality Checks ────────────────────────────────────────────────────
    R.append("## 12. Quality Checks\n")
    if qa is not None:
        R.append(df_to_md(qa))
    else:
        R.append("*Run `09_quality_checks.py` to generate.*\n")
    R.append("\n")

    # ── 13. Files Produced ────────────────────────────────────────────────────
    R.append("## 13. Files Produced\n")
    R.append(
        "| File | Description |\n|---|---|\n"
        "| `05_output/tables/05_attrition_all_link_rules.csv` | **Full cumulative attrition** |\n"
        "| `05_output/tables/06_auc_main_and_sensitivity_summary.csv` | **Main AUC + all sensitivities** |\n"
        "| `05_output/tables/06_auc_by_fyear.csv` | AUC by fiscal year |\n"
        "| `05_output/tables/06_auc_by_analyst_coverage.csv` | AUC by coverage bucket |\n"
        "| `05_output/tables/06_auc_tie_sensitivity.csv` | Tie-exclusion sensitivity |\n"
        "| `05_output/tables/06_auc_recent_periods.csv` | Recent-period AUC |\n"
        "| `05_output/tables/06_auc_consensus_median_sensitivity.csv` | Consensus median |\n"
        "| `05_output/tables/07_quality_checks.csv` | 13-point QA checklist |\n"
        "| `05_output/figures/roc_curve_main_score_le_2.png` | **Main ROC curve** |\n"
        "| `04_intermediate/merge/firm_year_auc_input_main_score_le_2.csv.gz` | **AUC input data** |\n"
        "| `05_output/report/AUC_results_tables.xlsx` | **Excel workbook** |\n"
    )

    out = REPORT_DIR / "analysts_forecast_auc_report.md"
    out.write_text("\n".join(R), encoding="utf-8")
    logger.info("Report written: %s", out)


if __name__ == "__main__":
    main()
