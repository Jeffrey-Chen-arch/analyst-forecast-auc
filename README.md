# Analyst Earnings-Forecast AUC

AUC of analysts' earnings-forecast direction against actual earnings-change
direction, following **Section 4.2.3 of Chen, Cho, Dou & Lev (2022, JAR)**.

Task assigned by Prof. Yiwei Dou (NYU Stern) on 2026-04-07 as the follow-up to
the Nov 24, 2025 sample-replication task. This repo implements:

1. The full AUC pipeline (sample prep → WRDS pull → IBES merge → AUC).
2. A forensic-validation script that stress-tests AUC = 0.8079 against every
   reasonable "is this a bug?" concern.
3. An additional-diagnostics script with the professor-ready MASTER_TABLE.

---

## Headline result

- **Main AUC = 0.8079** (N = 29,465 firm-years, 4,503 firms, FY 2005–2023)
- 95% firm-clustered bootstrap CI: **[0.8024, 0.8135]** (10,000 reps)
- p(AUC ≤ 0.5) < 0.0002 (within-fyear); < 0.0003 (global unconditional null)
- **14/14 QA checks PASS**, **19 PASS / 0 BLOCKING_FAIL** in forensic validation,
  **16 OK / 0 WARN / 0 SKIP** in additional diagnostics

For the paper's no-drift benchmark (0.6471) and the 2015–2018 sub-period
comparison (our AUC = 0.7936; coverage-reweighted counterfactual = 0.7749),
see `05_output/additional_diagnostics/MASTER_TABLE_professor_ready.csv`.

---

## What's in this repo

```
final_analyst_auc_FINAL/
├── 00_admin/
│   ├── README_FIRST.md                       ← execution guide
│   └── task_requirements_and_decisions.md    ← method decisions
├── 02_code/                                  ← all scripts (run from here)
│   ├── config.py                             ← all parameters
│   ├── run_all.py                            ← full one-shot pipeline
│   ├── RUN_ALL.bat
│   ├── requirements.txt
│   ├── 00_setup_folders.py … 10_make_excel_report.py
│   ├── 11_forensic_validation.py             ← bug-hunt stress test
│   ├── 12_additional_diagnostics.py          ← professor-ready MASTER_TABLE
│   └── src/common.py                         ← shared utilities
└── 05_output/                                ← results (committed)
    ├── tables/                               ← 14 QA checks + attrition + AUC
    ├── figures/                              ← ROC curves (main + 2 sensitivities)
    ├── report/
    │   ├── AUC_results_tables.xlsx           ← main Excel workbook (11 sheets)
    │   └── analysts_forecast_auc_report.md   ← draft report
    ├── forensic_validation/                  ← 21 CSVs + FORENSIC_SUMMARY.csv
    └── additional_diagnostics/               ← 21 CSVs + MASTER_TABLE_professor_ready.csv
```

**Not committed** (WRDS licensing; regenerated from WRDS on re-run):
`01_input/3.23_rerestart/02_raw/`, `01_input/3.23_rerestart/03_intermediate/`,
`03_raw/`, `04_intermediate/`, `06_logs/`.

> **Note on `05_output/`**: the aggregated firm-year files and summary tables
> contain IBES-derived fields. They are published here for academic verification
> only; they are not a substitute for a WRDS subscription.

---

## How to re-run

### Prerequisites

- Python 3.10 or later
- A WRDS account with IBES + CRSP + Compustat access
- The March 2026 replication-sample package `3.23_rerestart.zip`
  (it contains the Panel B GAAP sample from the Nov 24 task). Prof Dou already
  has a copy; I can also resend by email.

### Setup

```bash
git clone https://github.com/Jeffrey-Chen-arch/<repo>.git
cd <repo>

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r 02_code/requirements.txt
```

### Place the sample package

Put `3.23_rerestart.zip` either at the project root or inside `01_input/`.
`01_prepare_march_sample.py` auto-unzips it if the folder is absent.

### One-shot pipeline

```bash
cd 02_code
python run_all.py          # will prompt for WRDS credentials on first run
```

Or step by step (recommended for debugging):

| Step | Script | Notes |
|---:|---|---|
| 0 | `00_setup_folders.py` | — |
| 1 | `01_prepare_march_sample.py` | expect 50,402 firm-years / 6,396 firms |
| 2 | `02_link_sample_to_crsp.py` | CCM Scheme B |
| **3** | **`03_wrds_pull_ibes.py`** | **needs WRDS login** |
| 4 | `04_prepare_ibes_actuals.py` | — |
| 5 | `05_prepare_analyst_forecasts.py` | — |
| 6 | `06_construct_auc_sample.py` | expect 29,465 firm-years |
| 7 | `07_compute_auc.py` | 10,000-rep firm-clustered bootstrap, ~4 min |
| **8** | **`09_quality_checks.py`** | **target: 14 PASS / 0 FAIL** |
| 9 | `08_make_report.py` | — |
| 10 | `10_make_excel_report.py` | — |
| 11 | `11_forensic_validation.py` | forensic stress test (~2 min) |
| 12 | `12_additional_diagnostics.py` | extra diagnostics + MASTER_TABLE (~4 min) |

On Windows, the forensic/diagnostic scripts need UTF-8 stdout because of
emoji icons. Run them as:

```bash
PYTHONIOENCODING=utf-8 python -X utf8 02_code/11_forensic_validation.py
PYTHONIOENCODING=utf-8 python -X utf8 02_code/12_additional_diagnostics.py
```

Total wall-clock time end-to-end: **~15 minutes**.

---

## Key design choices (per Prof Dou's 2026-04-07 email)

1. **No de-drift.** Raw earnings-change directions only.
2. **IBES Actuals only** for actual EPS (street / pro-forma). No Compustat
   earnings numbers in the AUC computation.
3. **Sample**: March 2026 GAAP replication sample — *not* the Chen et al. (2022)
   Table 1 XBRL sample.
4. **IBES–CRSP link**: WRDS `ibcrsphist` with ICLINK score ≤ 2 (main);
   ≤ 5 and ≤ 6 reported as sensitivities.
5. **EPS basis**: adjusted `act_epsus` / `det_epsus`, matched pair enforced in
   code (never mixed with unadjusted).
6. **Score = proportion of analyst forecasts predicting an increase**, per
   analyst latest in an Apr 1–30 forecast window (portfolio formation = FYE + 3
   months). Consensus-median score reported as a sensitivity.

Full rationale and all 14 QA checks: see
[`00_admin/task_requirements_and_decisions.md`](00_admin/task_requirements_and_decisions.md).

---

## Known code fixes applied during execution

Documented here for transparency:

1. **`03_wrds_pull_ibes.py`** — `ibes.det_epsus.curr` is NULL for virtually all
   rows in the accessible WRDS schema; the denominating currency is in
   `curr_act`. Filter changed to `curr_act = 'USD' OR curr_act IS NULL`
   (verified no effect on AUC: strict-USD AUC identical to 4 dp; see
   `05_output/forensic_validation/T3D_currency_sensitivity.csv`).
2. **`05_prepare_analyst_forecasts.py`** — same fix for the defensive
   re-filter.
3. **`src/common.py :: bootstrap_auc`** — rewritten to numpy-vectorised
   concatenation (200× faster); sampling logic unchanged, verified by T4-B
   equivalence test on a 500-firm / 200-rep subsample (diff = 0.0000).
4. **`03_wrds_pull_ibes.py`** — accept `WRDS_USERNAME` env var to avoid the
   `input()` prompt in non-interactive runs.
5. **`11_forensic_validation.py`** — two `_x/_y` suffix-collision merge bugs in
   T2-A and T2-B2 fixed (drop duplicate column in right-hand frame before
   merging).

---

## Deliverables to look at first

- **[`05_output/additional_diagnostics/MASTER_TABLE_professor_ready.csv`](05_output/additional_diagnostics/MASTER_TABLE_professor_ready.csv)**
  — 17-row validation table mapping every concern to its test + result.
- [`05_output/forensic_validation/FORENSIC_SUMMARY.csv`](05_output/forensic_validation/FORENSIC_SUMMARY.csv)
  — 30-row forensic summary (PASS / WARN / ACTION_REQUIRED / DOC_FIX).
- [`05_output/tables/07_quality_checks.csv`](05_output/tables/07_quality_checks.csv)
  — 14 QA checks, all PASS.
- [`05_output/report/AUC_results_tables.xlsx`](05_output/report/AUC_results_tables.xlsx)
  — main Excel workbook (11 sheets).
- [`05_output/figures/roc_curve_main_score_le_2.png`](05_output/figures/roc_curve_main_score_le_2.png)
  — main ROC curve.

---

## Contact

**Jiahui Chen** (NYU Stern RA)
chen8892@umn.edu | github: [@Jeffrey-Chen-arch](https://github.com/Jeffrey-Chen-arch)
