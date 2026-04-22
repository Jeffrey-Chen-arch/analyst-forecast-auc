# Analyst Earnings-Forecast AUC

AUC of analysts' earnings-forecast directions against actual earnings-change
directions, following **Section 4.2.3 of Chen, Cho, Dou & Lev (2022, JAR)**.

## Result

- **Main AUC = 0.8079** (N = 29,465 firm-years, 4,503 firms, FY 2005–2023)
- 95% firm-clustered bootstrap CI: **[0.8024, 0.8135]** (10,000 reps)
- 14/14 QA PASS · 19/0 forensic PASS/BLOCKING · 16/0/0 diagnostics OK/WARN/SKIP

## Key files

| File | Description |
|---|---|
| `05_output/report/AUC_results_tables.xlsx` | Main Excel workbook (11 sheets) |
| `05_output/tables/07_quality_checks.csv` | 14 QA checks, all PASS |
| `05_output/additional_diagnostics/MASTER_TABLE_professor_ready.csv` | Validation summary (17 rows) |
| `05_output/forensic_validation/FORENSIC_SUMMARY.csv` | Forensic stress-test results |
| `05_output/figures/roc_curve_main_score_le_2.png` | Main ROC curve |

## Design

- Base sample: March 2026 GAAP replication sample merged to IBES (not the paper's XBRL Table 1 sample)
- No de-drifting; raw earnings-change directions only
- Actual EPS: IBES Actuals street/pro-forma (`act_epsus`); no Compustat earnings
- Forecast EPS: IBES Detail Forecasts (`det_epsus`, fpi=1, annual, usfirm=1)
- IBES–CRSP link: WRDS `ibcrsphist`, ICLINK score ≤ 2
- Score = proportion of analysts' latest April forecasts predicting an increase

## Re-run

**Requirements:** Python 3.10+, WRDS account (IBES + CRSP + Compustat), `3.23_rerestart.zip`

```bash
git clone https://github.com/Jeffrey-Chen-arch/analyst-forecast-auc
cd analyst-forecast-auc
pip install -r 02_code/requirements.txt
# Place 3.23_rerestart.zip at project root
cd 02_code
python run_all.py          # prompts for WRDS credentials on first run
```

Step-by-step (expected outputs):

| Step | Script | Expected |
|---:|---|---|
| 0 | `00_setup_folders.py` | — |
| 1 | `01_prepare_march_sample.py` | 50,402 firm-years |
| 2 | `02_link_sample_to_crsp.py` | CCM Scheme B |
| 3 | `03_wrds_pull_ibes.py` | needs WRDS login |
| 4 | `04_prepare_ibes_actuals.py` | — |
| 5 | `05_prepare_analyst_forecasts.py` | — |
| 6 | `06_construct_auc_sample.py` | 29,465 firm-years |
| 7 | `07_compute_auc.py` | ~4 min |
| 8 | `09_quality_checks.py` | 14 PASS / 0 FAIL |
| 9–10 | `08_make_report.py`, `10_make_excel_report.py` | — |
| 11 | `11_forensic_validation.py` | ~2 min |
| 12 | `12_additional_diagnostics.py` | ~4 min |

On Windows, run steps 11–12 with:
```bash
PYTHONIOENCODING=utf-8 python -X utf8 02_code/11_forensic_validation.py
PYTHONIOENCODING=utf-8 python -X utf8 02_code/12_additional_diagnostics.py
```

Total: ~15 min end-to-end.

**Not in repo** (WRDS licensing): `03_raw/`, `04_intermediate/`,
`01_input/.../02_raw/`, `01_input/.../03_intermediate/`, `06_logs/`.

## Code fixes applied during execution

1. `03_wrds_pull_ibes.py` — `curr_act` used instead of `curr` (virtually all NULL in accessible schema); verified no effect on AUC (strict-USD vs. USD-OR-NULL diff = 0.0000)
2. `src/common.py :: bootstrap_auc` — numpy-vectorised rewrite (200× faster); sampling logic unchanged, verified by T4-B equivalence test
3. `11_forensic_validation.py` — two `_x/_y` suffix-collision merge bugs fixed (T2-A, T2-B2)
4. `03_wrds_pull_ibes.py` — accepts `WRDS_USERNAME` env var for non-interactive runs
