# README: AUC of Analysts' Earnings Forecasts — v4

## One-sentence task
Compute the AUC of analysts' earnings forecast direction against actual
earnings-change direction, using the March 2026 GAAP replication sample merged
with IBES.  **No drift adjustment.  IBES Actuals only (no Compustat earnings).**

---

## Before you start — read this

1. **You need the March 2026 zip.**  Place `3_23_rerestart.zip` (or the
   already-unzipped folder `3.23_rerestart/`) in the project root **or** inside
   `01_input/`.  Step 1 auto-unzips if the folder is absent.

2. **No hard-coded WRDS username.**  `wrds.Connection()` prompts for your
   credentials on first run and caches them on the machine.  Nothing to edit.

3. **Do not skip steps.**  Each step creates files the next step needs.  If a
   step fails, open `06_logs/<script_name>.log` for the error.

4. **Run order matters.**  Step 8 (`09_quality_checks.py`) must run *before*
   step 9 (`08_make_report.py`) so the report includes the QA results.
   This is already enforced in `run_all.py`.

---

## Directory layout (after running)

```
4.13 计算分析师盈利预测的AUC/
├── 00_admin/                  ← this README + decisions doc
├── 01_input/
│   └── 3.23_rerestart/        ← auto-unzipped here if absent
├── 02_code/                   ← all scripts
│   ├── config.py              ← all parameters (no edits required)
│   ├── run_all.py             ← full one-shot pipeline
│   └── RUN_ALL.bat            ← Windows double-click shortcut
├── 03_raw/                    ← raw WRDS downloads
├── 04_intermediate/           ← cleaned + merged files
├── 05_output/
│   ├── tables/                ← all CSV result tables
│   ├── figures/               ← ROC curve PNGs
│   └── report/                ← markdown draft + Excel workbook
└── 06_logs/                   ← one .log file per script
```

---

## Setup (one time)

```bash
cd "C:\Users\18925\Desktop\research\Yiwei Dou\4.13 计算分析师盈利预测的AUC"
python -m venv .venv
.venv\Scripts\activate
pip install -r 02_code\requirements.txt
```

---

## Running

**Option A — full one-shot run (recommended):**
```bash
cd 02_code
python run_all.py
```
Or double-click `02_code\RUN_ALL.bat`.

**Option B — step by step (for debugging):**

| Step | Script | Key output to verify |
|---|---|---|
| 0 | `00_setup_folders.py` | Creates empty folder structure |
| 1 | `01_prepare_march_sample.py` | `00_base_sample_summary.csv` → ~50,402 firm-years, ~6,396 firms |
| 2 | `02_link_sample_to_crsp.py` | `01_ccm_schemeB_link_summary.csv` |
| **3** | **`03_wrds_pull_ibes.py`** | **Needs WRDS internet** → `02_wrds_pull_metadata.csv`; both `ibes_eps_basis` must be identical |
| 4 | `04_prepare_ibes_actuals.py` | `03_actuals_clean_summary.csv` |
| 5 | `05_prepare_analyst_forecasts.py` | `04_detail_forecasts_clean_summary.csv` |
| 6 | `06_construct_auc_sample.py` | `05_attrition_all_link_rules.csv` |
| 7 | `07_compute_auc.py` | `06_auc_main_and_sensitivity_summary.csv` + ROC curve PNGs |
| **8** | **`09_quality_checks.py`** | **`07_quality_checks.csv` → target: 13 PASS, 0 FAIL** |
| 9 | `08_make_report.py` | `analysts_forecast_auc_report.md` |
| 10 | `10_make_excel_report.py` | `AUC_results_tables.xlsx` |

> **Note on step numbering:** The script `09_quality_checks.py` runs as pipeline
> step 8 and `08_make_report.py` runs as step 9.  This ordering ensures the QA
> results are included in both the markdown report and the Excel workbook.

---

## Mandatory checks after step 8

Open `05_output/tables/07_quality_checks.csv` and confirm:

| Check | Must be |
|---|---|
| `07_main_sample_route_a_only` | PASS |
| `08_ibes_crsp_link_score_le_2` | PASS |
| `09_no_lookahead_in_actual_eps_t` | PASS |
| `11_no_compustat_earnings_columns_in_auc_input` | PASS |
| `12_ibes_eps_basis_consistent` | PASS |
| All 13 checks | PASS |

**Do not send results to professor if any check FAILS.  Investigate the log
file first.**

---

## Key numbers to verify after step 6

| Check | Expected |
|---|---|
| GAAP base sample | ~50,402 firm-years, ~6,396 firms |
| After CCM Scheme B | High 40,000s with valid permno |
| After IBES link (score ≤ 2) | See attrition table |
| After both EPS actuals | See attrition table |
| After analyst forecasts in window | Final AUC sample |
| Analyst AUC (no drift) | Comparable to paper's ~64.71% (reference only; samples differ) |

> **WRDS ICLINK score convention:** 0 = best match, 6 = worst match.
> Lower is better.  Main sample: score ≤ 2.  Sensitivities: ≤ 5 and ≤ 6.

---

## Final deliverables to send to professor

1. `05_output/report/AUC_results_tables.xlsx` — **Excel workbook (all results)**
2. `05_output/report/analysts_forecast_auc_report.md` — draft report
   (convert to PDF in Typora / VS Code before sending)
3. `05_output/tables/07_quality_checks.csv` — QA evidence
4. `05_output/figures/roc_curve_main_score_le_2.png` — ROC curve
5. `04_intermediate/merge/firm_year_auc_input_main_score_le_2.csv.gz` — reproducible data

---

## Troubleshooting

**WRDS login fails:**
Run `python -c "import wrds; wrds.Connection()"` to test.  Check your account
has IBES data access (contact wrds-support@wharton.upenn.edu if not).

**Step 3 raises "ibcrsphist is missing 'score'":**
The WRDS account may be accessing a schema that lacks the score column.  Try
logging in to WRDS directly and checking `wrdsapps.ibcrsphist` vs
`wrdsapps_link_crsp_ibes.ibcrsphist`.  Update `IBES_LINK_SCHEMA_CANDIDATES`
in `config.py` if needed.

**Step 3 raises "No candidate IBES tickers found":**
Step 2 (CCM Scheme B) did not produce valid permnos.  Check
`05_output/tables/01_ccm_schemeB_link_summary.csv`.

**Step 1 shows wrong counts:**
Verify `01_input/3.23_rerestart/03_intermediate/table1/gaap_sample.csv.gz` exists.

**Step 10 raises ImportError for xlsxwriter:**
Run `pip install -r 02_code\requirements.txt` again.

**QA Check 11 FAILS (Compustat earnings columns present):**
This means `06_construct_auc_sample.py` did not drop ni/ib/epspi etc.
Check that you are running v4 of that script (not an older version).
