# Task Requirements and Implementation Decisions

## Professor Dou's April 7, 2026 Instructions

1. Compute the AUC of analysts' earnings forecasts.
2. Use the sample replicated from the November 24, 2025 assignment.
3. Merge that sample with IBES.
4. Use the WRDS IBES–CRSP historical link table.
5. Follow the methodology in Section 4.2.3 of Chen et al. (2022, JAR).
6. Do **not** follow that paper's Table 1 to construct the sample.
7. Do **not** de-drift earnings changes or analyst forecasts.
8. Use raw earnings-change directions and raw forecasted earnings-change directions.
9. Measure earnings using IBES Actuals street/pro-forma EPS.
10. Do **not** use earnings numbers from Compustat.

---

## Base Sample

**File:** `3.23_rerestart/03_intermediate/table1/gaap_sample.csv.gz`

- 50,402 firm-years / 6,396 firms, fyear 2005–2023
- All December fiscal-year ends (fyr = 12)
- Source: Panel B / GAAP sample from the March 2026 full revision

---

## Variable Definitions

### Actual earnings direction (no drift)
```
actual_increase_raw = 1[IBES_Actual_EPS_{t+1} > IBES_Actual_EPS_t]
```
- Both values exclusively from IBES Actuals (never Compustat)
- Ties coded as 0 in main; excluded in tie sensitivity
- EPS_t: `actual_anndats` must be **non-missing AND ≤ formation_date**

### Analyst forecast direction (no drift)
```
forecast_increase_raw = 1[Forecast_EPS_{t+1} > IBES_Actual_EPS_t]
```
- Forecasts from IBES Detail History
- Only forecasts in the calendar month after portfolio formation
- One forecast per analyst (latest revision within window)

### Firm-year analyst score
```
analyst_score = mean(forecast_increase_raw)
             = N_forecasts_predicting_increase / N_total_forecasts
```

### AUC
```
AUC(y_true = actual_increase_raw, score = analyst_score)
```

---

## Timing

| Variable | Value |
|---|---|
| Fiscal year end (`datadate`) | Dec 31, year *t* |
| Portfolio formation | Mar 31, year *t+1* (= datadate + 3 months) |
| Forecast window | Apr 1–Apr 30, year *t+1* (month after formation) |
| EPS_t target date | ≈ Dec 31, year *t* (±7-day tolerance) |
| EPS_{t+1} target date | ≈ Dec 31, year *t+1* (±7-day tolerance) |

---

## Linking Chain

```
Compustat gvkey (GAAP sample)
  ↓ CCM Scheme B: linktype ∈ {LU, LC}, linkprim ∈ {P, C}
  ↓ valid at formation_date (NOT datadate)
CRSP permno
  ↓ WRDS ibcrsphist (WRDS ICLINK score: 0=best, 6=worst)
IBES ticker
  [main: score ≤ 2] [sensitivity 1: score ≤ 5] [sensitivity 2: score ≤ 6]

Compustat funda.ibtic → APPENDIX ONLY (not in any scored sample)
```

**IMPORTANT:** `ibtic` from Compustat is a static, unscored field. It must NOT
enter the main or sensitivity samples, which are defined by the WRDS IBES–CRSP
historical link table score. `ibtic` is used only in an appendix coverage diagnostic.

---

## IBES Table Pairs (must be matched)

| Basis | Actuals table | Detail table |
|---|---|---|
| Adjusted | `ibes.act_epsus` | `ibes.det_epsus` |
| Unadjusted | `ibes.actu_epsus` | `ibes.detu_epsus` |

The pipeline resolves the pair together and raises an error if the two tables
are on different adjustment bases. Never mix adjusted actuals with unadjusted forecasts.

---

## Parameters Set by This Project

| Parameter | Value | Reason |
|---|---|---|
| Formation months | 3 | Section 4.2.3: portfolio = FYE + 3 months |
| Forecast window | Month after formation | Section 4.2.3: "month following portfolio formation" |
| IBES link score (main) | ≤ 2 | High-quality links (ICLINK score 0=best, 6=worst) |
| IBES link score (sensitivity 1) | ≤ 5 | Standard broader threshold |
| IBES link score (sensitivity 2) | ≤ 6 | All scored links (maximum coverage check) |
| ibtic fallback | Appendix only | Unscored static field; not professor's specified link |
| Same-analyst dedup | Keep latest | Avoid counting within-window revisions multiple times |
| Actual tie | = 0 (main) | Standard binary classification convention |
| Tie exclusion | Separate sensitivity | Verify ties don't drive results |
| EPS_t known-by | anndats.notna() AND ≤ formation_date | No look-ahead bias |
| fpedats tolerance | ±7 days | Guards against minor Compustat–IBES date discrepancies |
| Applied to: | Both actuals AND forecasts | Symmetric treatment |
| Bootstrap | Firm-clustered, 2,000 reps | Correct for panel data; upgrade to 10,000 for submission |
| Random seed | 20260413 | Fixed for reproducibility |
| Consensus median | Appendix sensitivity | Compare to individual-forecast proportion (main) |

---

## Deliverables

### Required tables (`05_output/tables/`)

| File | Step | Description |
|---|---|---|
| `00_base_sample_summary.csv` | 01 | March GAAP sample counts |
| `01_ccm_schemeB_link_summary.csv` | 02 | CCM Scheme B match counts |
| `02_wrds_pull_metadata.csv` | 03 | WRDS tables actually used |
| `03_actuals_clean_summary.csv` | 04 | IBES actuals coverage |
| `03_actuals_duplicate_audit.csv` | 04 | Duplicate actual EPS audit |
| `04_detail_forecasts_clean_summary.csv` | 05 | Forecast coverage + analyst ID quality |
| `05_attrition_main_score_le_2.csv` | 06 | **Cumulative attrition (main)** |
| `05_attrition_all_link_rules.csv` | 06 | All score-threshold attritions |
| `06_auc_main_and_sensitivity_summary.csv` | 07 | **Primary AUC results** |
| `06_auc_by_fyear.csv` | 07 | AUC by fiscal year |
| `06_auc_by_analyst_coverage.csv` | 07 | AUC by coverage bucket |
| `06_auc_tie_sensitivity.csv` | 07 | Actual-tie exclusion sensitivity |
| `06_auc_recent_periods.csv` | 07 | 2019–2023, 2015–2023, FY2023 only |
| `06_auc_consensus_median_sensitivity.csv` | 07 | Consensus median vs. proportion score |
| `07_quality_checks.csv` | 09 | **13-point QA checklist** |

### Intermediate data (`04_intermediate/merge/`)

| File | Description |
|---|---|
| `firm_year_auc_input_main_score_le_2.csv.gz` | **Final AUC input — one row per firm-year** |
| `forecast_level_directions_main_score_le_2.csv.gz` | One row per analyst per firm-year |
| `firm_year_auc_input_sensitivity_score_le_5.csv.gz` | Sensitivity version |
| `firm_year_auc_input_sensitivity_score_le_6.csv.gz` | Sensitivity version (all scored) |
| `firm_year_auc_input_appendix_ibtic.csv.gz` | Appendix coverage check only |

### Report (`05_output/report/`)
- `analysts_forecast_auc_report.md` — draft markdown report
- `AUC_results_tables.xlsx` — **complete Excel workbook** (primary deliverable to professor)

---

## Verbatim Methodology Paragraph for Report

> I use the March restarted GAAP replication sample as the base sample. I link
> each firm-year to a CRSP permno using the CCM Scheme B design (linktype ∈ {LU, LC},
> linkprim ∈ {P, C}, valid at the portfolio formation date). I then link CRSP
> permnos to IBES tickers using the WRDS IBES–CRSP historical link table (ibcrsphist),
> retaining only links with ICLINK match score ≤ 2 in the main sample (score ≤ 5
> and score ≤ 6 in sensitivity analyses). Scores range from 0 (best) to 6 (worst).
> The Compustat `ibtic` field is used only in an appendix coverage diagnostic and
> is not included in the main or sensitivity samples.
>
> For each firm-year, I define the realized earnings-change direction using IBES
> Actuals street/pro-forma EPS only. I require the current-year actual EPS (EPS_t)
> to have been announced by the portfolio formation date (no look-ahead bias).
> I do not use Compustat earnings in any direction variable.
>
> Following Section 4.2.3 of Chen et al. (2022), I take individual analyst forecasts
> issued in the calendar month after portfolio formation (April for December fiscal-year
> firms) and define each forecast as predicting an earnings increase if the forecasted
> EPS for fiscal year t+1 exceeds IBES Actual EPS for fiscal year t. I do not subtract
> a drift term. Within each firm-year, I retain only the most recent forecast per analyst
> to avoid double-counting within-window revisions. The firm-year analyst score is the
> proportion of individual forecasts predicting an increase. Both analyst forecasts and
> IBES Actuals are taken from the same IBES EPS adjustment-family table pair to ensure
> a consistent per-share basis. I compute AUC of this score against the realized raw
> earnings-change direction, with statistical inference based on firm-clustered bootstrap
> confidence intervals.
