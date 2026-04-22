# Analysts' Earnings Forecast AUC — Results Report

*Sample: march_restarted_strict_gaap_panel_b  |  Bootstrap: 10,000 reps, firm-clustered=True*

## 1. Task Interpretation

This report computes the AUC of professional analysts' earnings forecast directions following Section 4.2.3 of Chen, Cho, Dou, and Lev (2022, *JAR*). It uses the March 2026 restarted GAAP replication sample as the base, merges through CRSP to IBES, and constructs raw (no-drift) earnings directions.

**Key design choices per Professor Dou's April 7, 2026 instructions:**

- **No drift adjustment** — raw EPS changes only
- **IBES Actuals only** for actual EPS — Compustat earnings not used
- **WRDS IBES–CRSP historical link table** — no unscored static ibtic in main sample
- **Individual analyst forecasts** from IBES Detail History
- **Firm-clustered bootstrap** for inference

## 2. Methodology

### 2.1 Variable Definitions

| Variable | Formula |
|---|---|
| `actual_increase_raw` | = 1 if IBES Actual EPS\(_{t+1}\) > IBES Actual EPS\(_t\), else 0 |
| `forecast_increase_raw` | = 1 if Analyst Forecast\(_{t+1}\) > IBES Actual EPS\(_t\), else 0 |
| `analyst_score` | Proportion of analysts predicting an EPS increase (∈ [0, 1]) |
| `y_true` | = `actual_increase_raw` |
| `score` | = `analyst_score` |

### 2.2 Timing (December FY firms — entire sample)

| Date | Description |
|---|---|
| Dec 31, year *t* | Fiscal year end (`datadate`) |
| Mar 31, year *t+1* | Portfolio formation (FYE + 3 months) |
| **Apr 1–Apr 30, year *t+1*** | **Forecast window** — forecasts used for analyst score |
| Dec 31, year *t+1* | EPS_{t+1} fiscal year end |

**EPS_t known-by rule:** IBES actual EPS for FY *t* must have a non-missing `actual_anndats` that is ≤ portfolio formation date. This prevents look-ahead bias.

### 2.3 Linking Chain

```
Compustat gvkey
  → CRSP permno  [CCM Scheme B: LU/LC, P/C, valid at formation_date]
  → IBES ticker  [WRDS ibcrsphist; ICLINK score 0=best, 6=worst]
    Main sample:  score ≤ 2
    Sensitivity:  score ≤ 5
    Sensitivity:  score ≤ 6 (all scored)
    Appendix:     + Compustat ibtic (unscored, coverage diagnostic only)
```

**Note on WRDS ICLINK score:** The score ranges from 0 (best match) to 6 (worst match). Lower scores indicate better link quality. The main sample uses score ≤ 2 to include only the highest-quality links.

### 2.4 IBES EPS Adjustment Basis

Analyst forecasts and IBES Actuals are pulled from the **same adjustment-family table pair** (either adjusted `act_epsus`/`det_epsus`, or unadjusted `actu_epsus`/`detu_epsus`). The tables are never mixed, preventing split-adjustment basis mismatches in EPS comparisons.

### 2.5 AUC

AUC = Pr(analyst score for earnings-increase firm > analyst score for non-increase firm). AUC = 0.50 is the random-guess baseline. Inference: firm-clustered bootstrap (10,000 reps).

## 3. Data Sources

| dataset               | schema                  | table      |   n_rows |   n_sample_permnos |   n_scored_tickers_le_2 |   n_scored_tickers_le_6 |   n_ibtic_tickers |   n_total_pull_tickers | columns                                                                                                  | ibes_eps_basis   |
|:----------------------|:------------------------|:-----------|---------:|-------------------:|------------------------:|------------------------:|------------------:|-----------------------:|:---------------------------------------------------------------------------------------------------------|:-----------------|
| ibes_crsp_link        | wrdsapps_link_crsp_ibes | ibcrsphist |    37662 |               5962 |                    5893 |                    5910 |                 0 |                   5910 | ticker,permno,ncusip,sdate,edate,score                                                                   | nan              |
| ibes_actuals          | ibes                    | act_epsus  |    65844 |                nan |                     nan |                     nan |               nan |                    nan | ticker,cusip,oftic,cname,pends,anndats,value,measure,pdicity,curr_act,usfirm,actdats                     | adjusted         |
| ibes_detail_forecasts | ibes                    | det_epsus  |  2301017 |                nan |                     nan |                     nan |               nan |                    nan | ticker,cusip,oftic,cname,fpedats,anndats,revdats,estimator,analys,value,measure,fpi,usfirm,curr,curr_act | adjusted         |


## 4. Base Sample

| file        |   n_obs |   n_firms |   min_fyear |   max_fyear |
|:------------|--------:|----------:|------------:|------------:|
| gaap_sample |   50402 |      6396 |        2005 |        2023 |
| raw_sample  |   80446 |     10076 |        2003 |        2024 |


## 5. Sample Attrition

Each step is cumulative. `pct_retained_from_previous` = fraction of the prior step's observations that survive this step.


| sample_label    | step                                        |   n_obs |   n_firms |   min_fyear |   max_fyear |   dropped_from_previous |   pct_retained_from_previous |   pct_retained_from_base |
|:----------------|:--------------------------------------------|--------:|----------:|------------:|------------:|------------------------:|-----------------------------:|-------------------------:|
| main_score_le_2 | 01_base_gaap_sample                         |   50402 |      6396 |        2005 |        2023 |                       0 |                       1      |                   1      |
| main_score_le_2 | 02_has_CCM_SchemeB_permno_at_formation      |   45968 |      5908 |        2005 |        2023 |                    4434 |                       0.912  |                   0.912  |
| main_score_le_2 | 03_has_WRDS_IBES_link_score_le_2            |   44347 |      5763 |        2005 |        2023 |                    1621 |                       0.9647 |                   0.8799 |
| main_score_le_2 | 04_has_IBES_actual_EPS_t_known_by_formation |   40542 |      5499 |        2005 |        2023 |                    3805 |                       0.9142 |                   0.8044 |
| main_score_le_2 | 05_has_IBES_actual_EPS_tp1                  |   39632 |      5431 |        2005 |        2023 |                     910 |                       0.9776 |                   0.7863 |
| main_score_le_2 | 06_has_analyst_forecasts_in_window          |   29465 |      4503 |        2005 |        2023 |                   10167 |                       0.7435 |                   0.5846 |
| main_score_le_2 | 07_final_AUC_sample                         |   29465 |      4503 |        2005 |        2023 |                       0 |                       1      |                   0.5846 |


## 6. IBES Data Quality

### 6.1 Actuals coverage

|   n_rows |   n_tickers | min_fpedats   | max_fpedats   |   duplicate_ticker_fpedats_pairs |   rows_with_actual_anndats | pct_with_actual_anndats   |
|---------:|------------:|:--------------|:--------------|---------------------------------:|---------------------------:|:--------------------------|
|    65844 |        5757 | 2004-12-31    | 2025-12-31    |                                0 |                      65844 | 100.0%                    |

### 6.2 Duplicate actuals

- 0 (ticker, fpedats) pairs have more than one row. Resolved by taking the row with the latest `actual_anndats`.


### 6.3 Detail forecasts coverage

|   n_rows |   n_tickers |   n_analyst_ids_observed |   n_analyst_ids_fallback | pct_analyst_ids_fallback   | min_forecast_anndats   | max_forecast_anndats   | min_fpedats   | max_fpedats   |
|---------:|------------:|-------------------------:|-------------------------:|:---------------------------|:-----------------------|:-----------------------|:--------------|:--------------|
|  2300899 |        5512 |                  2300899 |                        0 | 0.0%                       | 2006-02-01             | 2024-06-30             | 2004-12-31    | 2025-12-31    |


## 7. Compustat Earnings Audit

| Item | Status |
|---|---|
| Base sample constructed from Compustat | Yes |
| CRSP link via CCM (Compustat–CRSP) | Yes |
| Compustat `ni` used in AUC actual direction | **No** |
| Compustat EPS used in AUC actual direction | **No** |
| IBES Actual EPS_t used | **Yes** |
| IBES Actual EPS_{t+1} used | **Yes** |
| Analyst forecast EPS from IBES Detail | **Yes** |

> Compustat earnings numbers are used **only** to construct the replication sample (balance sheet / income statement filters for Panel B). All earnings-change directions are defined using IBES Actuals exclusively.

## 8. Main AUC Results

| Metric | Value |
|---|---|
| N firm-years | 29,465 |
| N unique firms | 4,503 |
| N fiscal years | 19 |
| Earnings increases | 17,942 (60.89%) |
| Mean analyst score | 0.6541 |
| Median analyst score | 1.0000 |
| **AUC (main, score ≤ 2, no drift)** | **80.79%** |
| Bootstrap 95% CI | [80.24%, 81.35%] |
| p-value (AUC ≤ 0.50) | 0.0000 |
| Accuracy at 0.5 cutoff | 79.62% |



## 9. Comparison with Paper Benchmark

| | AUC |
|---|---|
| **This analysis (main, no drift)** | **80.79%** |
| Chen et al. (2022), analysts, no drift (fn. 34) | 64.71% |
| Chen et al. (2022), analysts, with drift (Table 5) | 65.09% |
| Random guess baseline | 50.00% |

> The paper's 64.71% (no drift) is a reference point, not a target. Differences reflect sample period (this study: 2005–2023 vs. paper: 2015–2018) and sample definition (March replication sample vs. paper's XBRL sample).

## 10. Sensitivity Analyses

### 10.1 IBES link score threshold

| sample_label           |   n_obs |   n_firms |      auc |   auc_ci_p2_5 |   auc_ci_p97_5 |   p_value_auc_le_0p5 |
|:-----------------------|--------:|----------:|---------:|--------------:|---------------:|---------------------:|
| main_score_le_2        |   29465 |      4503 | 0.807946 |      0.802396 |       0.813461 |                    0 |
| sensitivity_score_le_5 |   29465 |      4503 | 0.807946 |      0.802396 |       0.813461 |                    0 |
| sensitivity_score_le_6 |   29465 |      4503 | 0.807946 |      0.802396 |       0.813461 |                    0 |


### 10.2 Actual-tie exclusion sensitivity

| sample_label                               |   n_obs |   n_firms |   n_years |   n_actual_increase |   n_actual_decrease_or_equal |   actual_increase_rate |   mean_score |   median_score |     auc |   accuracy_cutoff_0p5 |   predicted_increase_rate_cutoff_0p5 |
|:-------------------------------------------|--------:|----------:|----------:|--------------------:|-----------------------------:|-----------------------:|-------------:|---------------:|--------:|----------------------:|-------------------------------------:|
| main_score_le_2_exclude_actual_ties        |   29222 |      4494 |        19 |               17942 |                        11280 |               0.613989 |     0.654576 |              1 | 0.81097 |              0.799227 |                             0.644959 |
| sensitivity_score_le_5_exclude_actual_ties |   29222 |      4494 |        19 |               17942 |                        11280 |               0.613989 |     0.654576 |              1 | 0.81097 |              0.799227 |                             0.644959 |
| sensitivity_score_le_6_exclude_actual_ties |   29222 |      4494 |        19 |               17942 |                        11280 |               0.613989 |     0.654576 |              1 | 0.81097 |              0.799227 |                             0.644959 |


### 10.3 AUC by analyst coverage

| sample_label           | analyst_coverage_bucket   |   n_obs |   n_firms |   n_years |   n_actual_increase |   n_actual_decrease_or_equal |   actual_increase_rate |   mean_score |   median_score |      auc |   accuracy_cutoff_0p5 |   predicted_increase_rate_cutoff_0p5 |
|:-----------------------|:--------------------------|--------:|----------:|----------:|--------------------:|-----------------------------:|-----------------------:|-------------:|---------------:|---------:|----------------------:|-------------------------------------:|
| main_score_le_2        | 1                         |    6618 |      2802 |        19 |                3803 |                         2815 |               0.574645 |     0.638561 |              1 | 0.729828 |              0.745391 |                             0.638561 |
| main_score_le_2        | 2                         |    4182 |      2117 |        19 |                2426 |                         1756 |               0.580105 |     0.639766 |              1 | 0.7802   |              0.758011 |                             0.569106 |
| main_score_le_2        | 3-4                       |    5576 |      2173 |        19 |                3390 |                         2186 |               0.607963 |     0.644533 |              1 | 0.81085  |              0.79089  |                             0.638451 |
| main_score_le_2        | 5+                        |   13089 |      2164 |        19 |                8323 |                         4766 |               0.635877 |     0.670707 |              1 | 0.868013 |              0.836275 |                             0.673848 |
| sensitivity_score_le_5 | 1                         |    6618 |      2802 |        19 |                3803 |                         2815 |               0.574645 |     0.638561 |              1 | 0.729828 |              0.745391 |                             0.638561 |
| sensitivity_score_le_5 | 2                         |    4182 |      2117 |        19 |                2426 |                         1756 |               0.580105 |     0.639766 |              1 | 0.7802   |              0.758011 |                             0.569106 |
| sensitivity_score_le_5 | 3-4                       |    5576 |      2173 |        19 |                3390 |                         2186 |               0.607963 |     0.644533 |              1 | 0.81085  |              0.79089  |                             0.638451 |
| sensitivity_score_le_5 | 5+                        |   13089 |      2164 |        19 |                8323 |                         4766 |               0.635877 |     0.670707 |              1 | 0.868013 |              0.836275 |                             0.673848 |
| sensitivity_score_le_6 | 1                         |    6618 |      2802 |        19 |                3803 |                         2815 |               0.574645 |     0.638561 |              1 | 0.729828 |              0.745391 |                             0.638561 |
| sensitivity_score_le_6 | 2                         |    4182 |      2117 |        19 |                2426 |                         1756 |               0.580105 |     0.639766 |              1 | 0.7802   |              0.758011 |                             0.569106 |
| sensitivity_score_le_6 | 3-4                       |    5576 |      2173 |        19 |                3390 |                         2186 |               0.607963 |     0.644533 |              1 | 0.81085  |              0.79089  |                             0.638451 |
| sensitivity_score_le_6 | 5+                        |   13089 |      2164 |        19 |                8323 |                         4766 |               0.635877 |     0.670707 |              1 | 0.868013 |              0.836275 |                             0.673848 |


### 10.4 Consensus median score sensitivity

*Uses 1[median consensus forecast > Actual EPS_t] instead of proportion score.*


| sample_label                            |   n_obs |   n_firms |   n_years |   n_actual_increase |   n_actual_decrease_or_equal |   actual_increase_rate |   mean_score |   median_score |      auc |   accuracy_cutoff_0p5 |   predicted_increase_rate_cutoff_0p5 |
|:----------------------------------------|--------:|----------:|----------:|--------------------:|-----------------------------:|-----------------------:|-------------:|---------------:|---------:|----------------------:|-------------------------------------:|
| main_score_le_2_consensus_median        |   29465 |      4503 |        19 |               17942 |                        11523 |               0.608926 |     0.660003 |              1 | 0.775717 |              0.797489 |                             0.660003 |
| sensitivity_score_le_5_consensus_median |   29465 |      4503 |        19 |               17942 |                        11523 |               0.608926 |     0.660003 |              1 | 0.775717 |              0.797489 |                             0.660003 |
| sensitivity_score_le_6_consensus_median |   29465 |      4503 |        19 |               17942 |                        11523 |               0.608926 |     0.660003 |              1 | 0.775717 |              0.797489 |                             0.660003 |


### 10.5 Recent-period subsamples

| sample_label    | period      |   fyear_start |   fyear_end |   n_obs |   n_firms |   n_years |   n_actual_increase |   n_actual_decrease_or_equal |   actual_increase_rate |   mean_score |   median_score |      auc |   accuracy_cutoff_0p5 |   predicted_increase_rate_cutoff_0p5 |
|:----------------|:------------|--------------:|------------:|--------:|----------:|----------:|--------------------:|-----------------------------:|-----------------------:|-------------:|---------------:|---------:|----------------------:|-------------------------------------:|
| main_score_le_2 | full_sample |          2005 |        2023 |   29465 |      4503 |        19 |               17942 |                        11523 |               0.608926 |     0.654142 |       1        | 0.807946 |              0.796165 |                             0.644358 |
| main_score_le_2 | 2015_2023   |          2015 |        2023 |   14242 |      3072 |         9 |                8806 |                         5436 |               0.618312 |     0.631507 |       1        | 0.807422 |              0.794832 |                             0.621823 |
| main_score_le_2 | 2019_2023   |          2019 |        2023 |    7924 |      2495 |         5 |                4643 |                         3281 |               0.585941 |     0.567662 |       0.857143 | 0.812424 |              0.787986 |                             0.557673 |
| main_score_le_2 | fy2023_only |          2023 |        2023 |    1408 |      1408 |         1 |                 848 |                          560 |               0.602273 |     0.629116 |       1        | 0.815049 |              0.807528 |                             0.628551 |


## 11. AUC by Fiscal Year

| sample_label    |   fyear |   n_obs |   n_firms |   n_years |   n_actual_increase |   n_actual_decrease_or_equal |   actual_increase_rate |   mean_score |   median_score |      auc |   accuracy_cutoff_0p5 |   predicted_increase_rate_cutoff_0p5 |
|:----------------|--------:|--------:|----------:|----------:|--------------------:|-----------------------------:|-----------------------:|-------------:|---------------:|---------:|----------------------:|-------------------------------------:|
| main_score_le_2 |    2005 |    1423 |      1423 |         1 |                 995 |                          428 |               0.699227 |     0.796    |       1        | 0.751555 |              0.798313 |                             0.795502 |
| main_score_le_2 |    2006 |    1427 |      1427 |         1 |                 810 |                          617 |               0.567624 |     0.684036 |       1        | 0.775967 |              0.765242 |                             0.672039 |
| main_score_le_2 |    2007 |    1428 |      1428 |         1 |                 655 |                          773 |               0.458683 |     0.628781 |       1        | 0.819068 |              0.770308 |                             0.618347 |
| main_score_le_2 |    2008 |    1493 |      1493 |         1 |                 616 |                          877 |               0.412592 |     0.373149 |       0        | 0.838407 |              0.815137 |                             0.364367 |
| main_score_le_2 |    2009 |    1508 |      1508 |         1 |                1085 |                          423 |               0.719496 |     0.740774 |       1        | 0.828351 |              0.842175 |                             0.735411 |
| main_score_le_2 |    2010 |    1420 |      1420 |         1 |                 993 |                          427 |               0.699296 |     0.76527  |       1        | 0.792747 |              0.814085 |                             0.758451 |
| main_score_le_2 |    2011 |    1637 |      1637 |         1 |                 987 |                          650 |               0.602932 |     0.709201 |       1        | 0.796888 |              0.785583 |                             0.696396 |
| main_score_le_2 |    2012 |    1637 |      1637 |         1 |                1043 |                          594 |               0.637141 |     0.697667 |       1        | 0.817902 |              0.809407 |                             0.682346 |
| main_score_le_2 |    2013 |    1624 |      1624 |         1 |                1058 |                          566 |               0.651478 |     0.729441 |       1        | 0.782306 |              0.788793 |                             0.722291 |
| main_score_le_2 |    2014 |    1626 |      1626 |         1 |                 894 |                          732 |               0.549815 |     0.630447 |       1        | 0.813703 |              0.784748 |                             0.612546 |
| main_score_le_2 |    2015 |    1611 |      1611 |         1 |                 980 |                          631 |               0.608318 |     0.634242 |       1        | 0.797278 |              0.779019 |                             0.617008 |
| main_score_le_2 |    2016 |    1579 |      1579 |         1 |                1075 |                          504 |               0.680811 |     0.735254 |       1        | 0.78961  |              0.817606 |                             0.731476 |
| main_score_le_2 |    2017 |    1578 |      1578 |         1 |                1198 |                          380 |               0.759189 |     0.813831 |       1        | 0.785676 |              0.840938 |                             0.809252 |
| main_score_le_2 |    2018 |    1550 |      1550 |         1 |                 910 |                          640 |               0.587097 |     0.663751 |       1        | 0.777734 |              0.776129 |                             0.652258 |
| main_score_le_2 |    2019 |    1782 |      1782 |         1 |                 780 |                         1002 |               0.43771  |     0.274467 |       0        | 0.731044 |              0.70651  |                             0.251964 |
| main_score_le_2 |    2020 |    1603 |      1603 |         1 |                1287 |                          316 |               0.80287  |     0.80161  |       1        | 0.779179 |              0.84529  |                             0.80287  |
| main_score_le_2 |    2021 |    1578 |      1578 |         1 |                 917 |                          661 |               0.581115 |     0.615625 |       1        | 0.803827 |              0.78834  |                             0.601394 |
| main_score_le_2 |    2022 |    1553 |      1553 |         1 |                 811 |                          742 |               0.522215 |     0.558159 |       0.846154 | 0.833128 |              0.80425  |                             0.546684 |
| main_score_le_2 |    2023 |    1408 |      1408 |         1 |                 848 |                          560 |               0.602273 |     0.629116 |       1        | 0.815049 |              0.807528 |                             0.628551 |


## 12. Quality Checks

| check_name                                    | status   | value                      | expected                   | comment                                                                                                                        |
|:----------------------------------------------|:---------|:---------------------------|:---------------------------|:-------------------------------------------------------------------------------------------------------------------------------|
| 01_no_duplicate_gvkey_fyear                   | PASS     | 0                          | 0                          | Each firm-year must appear exactly once in the AUC sample                                                                      |
| 02_y_true_binary                              | PASS     | [0, 1]                     | [0, 1]                     | y_true must be 0 or 1 only                                                                                                     |
| 03_score_in_unit_interval                     | PASS     | 0                          | 0                          | analyst_score is a proportion; must be in [0, 1]                                                                               |
| 04_no_missing_actual_eps_t                    | PASS     | 0                          | 0                          | AUC sample must only contain rows where both EPS values are present                                                            |
| 04_no_missing_actual_eps_tp1                  | PASS     | 0                          | 0                          | AUC sample must only contain rows where both EPS values are present                                                            |
| 05_forecast_anndats_in_window                 | PASS     | 0                          | 0                          | All forecasts must fall within the designated forecast window                                                                  |
| 06_forecast_fpedats_within_tolerance          | PASS     | 0                          | 0                          | All forecast fpedats within ±7 days of target                                                                                  |
| 07_main_sample_route_a_only                   | PASS     | 0                          | 0                          | Main sample must use WRDS ibcrsphist (route_a) only; ibtic is appendix only                                                    |
| 08_ibes_crsp_link_score_le_2                  | PASS     | missing=0, max=2.0         | missing=0, max≤2           | WRDS ICLINK score: 0 = best, 6 = worst; main sample must have non-missing score ≤ 2 for every observation                      |
| 09_no_lookahead_in_actual_eps_t               | PASS     | lookahead=0, missing=0     | 0, 0                       | EPS_t announced before formation; missing anndats excluded from main                                                           |
| 10_both_classes_present_for_AUC               | PASS     | [np.int64(0), np.int64(1)] | [0, 1]                     | AUC requires both earnings-increase (y=1) and non-increase (y=0) observations                                                  |
| 11_no_compustat_earnings_columns_in_auc_input | PASS     | []                         | []                         | Final AUC input must not carry Compustat earnings columns (ni, ib, epspi, etc.); all earnings directions use IBES Actuals only |
| 12_ibes_eps_basis_consistent                  | PASS     | adjusted                   | same basis for both tables | Actuals and forecasts must use the same IBES EPS split-adjustment basis                                                        |
| 13_auc_in_sensible_range                      | PASS     | 0.8079                     | between 0.45 and 0.85      | AUC outside [0.45, 0.85] is unusual for analyst earnings forecast studies                                                      |


## 13. Files Produced

| File | Description |
|---|---|
| `05_output/tables/05_attrition_all_link_rules.csv` | **Full cumulative attrition** |
| `05_output/tables/06_auc_main_and_sensitivity_summary.csv` | **Main AUC + all sensitivities** |
| `05_output/tables/06_auc_by_fyear.csv` | AUC by fiscal year |
| `05_output/tables/06_auc_by_analyst_coverage.csv` | AUC by coverage bucket |
| `05_output/tables/06_auc_tie_sensitivity.csv` | Tie-exclusion sensitivity |
| `05_output/tables/06_auc_recent_periods.csv` | Recent-period AUC |
| `05_output/tables/06_auc_consensus_median_sensitivity.csv` | Consensus median |
| `05_output/tables/07_quality_checks.csv` | 13-point QA checklist |
| `05_output/figures/roc_curve_main_score_le_2.png` | **Main ROC curve** |
| `04_intermediate/merge/firm_year_auc_input_main_score_le_2.csv.gz` | **AUC input data** |
| `05_output/report/AUC_results_tables.xlsx` | **Excel workbook** |
