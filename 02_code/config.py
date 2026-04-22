"""
config.py — Configuration for the IBES analysts' earnings-forecast AUC task.

Project root:
  C:/Users/18925/Desktop/research/Yiwei Dou/4.13 计算分析师盈利预测的AUC

HOW TO USE
----------
No WRDS_USERNAME variable is hard-coded here. WRDS login is handled by
wrds.Connection() at runtime; you will be prompted for credentials on the
first run on each machine.

The ONLY parameters you might want to adjust:
  BOOTSTRAP_REPS  – increase from 2000 to 10000 for the final submission
  MAIN_MAX_IBES_CRSP_SCORE – default 2 is the published standard
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR    = PROJECT_ROOT / "01_input"
CODE_DIR     = PROJECT_ROOT / "02_code"
RAW_DIR      = PROJECT_ROOT / "03_raw"
INTERMEDIATE_DIR = PROJECT_ROOT / "04_intermediate"
OUTPUT_DIR   = PROJECT_ROOT / "05_output"
LOG_DIR      = PROJECT_ROOT / "06_logs"

MARCH_DIR         = INPUT_DIR / "3.23_rerestart"
MARCH_ZIP_PATTERN = "*3.23*.zip"

MARCH_GAAP_SAMPLE = MARCH_DIR / "03_intermediate" / "table1" / "gaap_sample.csv.gz"
MARCH_RAW_SAMPLE  = MARCH_DIR / "03_intermediate" / "table1" / "raw_sample.csv.gz"
MARCH_CCM_LINKS   = MARCH_DIR / "02_raw" / "ccm_links_full.csv.gz"

RAW_IBES_DIR   = RAW_DIR      / "ibes"
RAW_LINK_DIR   = RAW_DIR      / "links"
INT_SAMPLE_DIR = INTERMEDIATE_DIR / "sample"
INT_LINK_DIR   = INTERMEDIATE_DIR / "links"
INT_IBES_DIR   = INTERMEDIATE_DIR / "ibes"
INT_MERGE_DIR  = INTERMEDIATE_DIR / "merge"
TABLE_DIR      = OUTPUT_DIR / "tables"
FIG_DIR        = OUTPUT_DIR / "figures"
REPORT_DIR     = OUTPUT_DIR / "report"

# ── Base sample ───────────────────────────────────────────────────────────────
BASE_SAMPLE_NAME = "march_restarted_strict_gaap_panel_b"

# ── Fiscal timing ─────────────────────────────────────────────────────────────
# FYE + 3 months = portfolio formation date (Section 4.2.3)
# Forecast window = single calendar month immediately after formation
# For December FY: formation = Mar 31; forecast window = Apr 1–Apr 30
FORMATION_MONTHS_AFTER_FYE = 3
FORECAST_WINDOW = "month_after_formation"

# ── CCM Scheme B ──────────────────────────────────────────────────────────────
CCM_LINKTYPES = {"LU", "LC"}
CCM_LINKPRIMS = {"P", "C"}

# ── IBES–CRSP link quality ────────────────────────────────────────────────────
# WRDS ICLINK score: 0 = best match, 6 = worst match.
# Lower scores indicate better link quality; high scores need further checking.
# References: WRDS ICLINK documentation; WRDS IBES–CRSP linking matrix page.
MAIN_MAX_IBES_CRSP_SCORE         = 2   # high-quality main result
SENSITIVITY_MAX_IBES_CRSP_SCORE  = 5   # broader sensitivity
SENSITIVITY_MAX_IBES_CRSP_SCORE2 = 6   # all valid scored links (maximum coverage)

# ibtic fallback: Compustat funda.ibtic is a STATIC, UNSCORED ticker field.
# It must NOT enter the main sample (which is defined by the scored WRDS link).
# It is used only in an appendix coverage check (separate label).
USE_IBTIC_FALLBACK_IN_MAIN     = False  # ← never change to True for main result
USE_IBTIC_FALLBACK_APPENDIX    = True   # build appendix sample for coverage audit

# ── IBES table pairs (adjusted / unadjusted must match) ───────────────────────
# CRITICAL: actuals and detail forecasts MUST come from the same adjustment basis.
# Never mix det_epsus (adjusted) actuals with detu_epsus (unadjusted) forecasts.
# Code resolves the pair together and raises an error if they differ.
IBES_TABLE_PAIRS = [
    {"basis": "adjusted",   "actuals": "act_epsus",  "detail": "det_epsus"},
    {"basis": "unadjusted", "actuals": "actu_epsus", "detail": "detu_epsus"},
]
IBES_LIBRARY                 = "ibes"
IBES_LINK_SCHEMA_CANDIDATES  = ["wrdsapps_link_crsp_ibes", "wrdsapps"]
IBES_LINK_TABLE              = "ibcrsphist"

# ── IBES content filters ──────────────────────────────────────────────────────
IBES_MEASURE     = "EPS"
IBES_PERIODICITY = "ANN"   # WRDS uses "ANN"; code also accepts "A" defensively
IBES_CURRENCY    = "USD"
IBES_FPI_ANNUAL  = "1"     # one fiscal year ahead

# ── Date-matching tolerance ───────────────────────────────────────────────────
# Applied to BOTH actuals (fpedats vs target) AND forecasts (fpedats vs target).
# ±7 days guards against minor Compustat–IBES fiscal-year-end date discrepancies.
PENDS_MATCH_TOLERANCE_DAYS = 7

# ── Direction definitions ─────────────────────────────────────────────────────
# NO drift adjustment (per professor's instruction).
# actual_increase_raw   = 1[Actual EPS_{t+1}  > Actual EPS_t]
# forecast_increase_raw = 1[Forecast EPS_{t+1} > Actual EPS_t]
# Ties coded as 0 in main; excluded in sensitivity.
EXCLUDE_TIES_IN_SENSITIVITY = True

# ── EPS_t known-by rule ───────────────────────────────────────────────────────
# Main sample: actual_anndats_t must be NON-MISSING and ≤ formation_date.
# Prevents look-ahead bias (analyst cannot use EPS_t they have not yet seen).
# Sensitivity: allow missing actual_anndats_t when fpedats_t ≤ formation_date-90d
REQUIRE_ACTUAL_ANNDATS_NONMISSING = True

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# Firm-clustered bootstrap is correct for panel data.
# Set to 10000 for the final submission (2000 for drafts).
BOOTSTRAP_REPS            = 10000
BOOTSTRAP_CLUSTER_BY_FIRM = True
RANDOM_SEED               = 20260413

# ── WRDS pull date range padding ──────────────────────────────────────────────
PULL_YEAR_PAD_BEFORE = 1
PULL_YEAR_PAD_AFTER  = 2

# ── Identifier dtypes ─────────────────────────────────────────────────────────
# Force string on read to preserve leading zeros in IBES tickers, CUSIPs, etc.
ID_DTYPES = {
    "gvkey":  str, "ticker": str, "cusip":  str,
    "oftic":  str, "ncusip": str, "permno": str,
    "permco": str, "liid":   str, "ibtic":  str,
}

# ── Paper benchmark ───────────────────────────────────────────────────────────
# Chen, Cho, Dou, Lev (2022, JAR) — for reference only, NOT a target.
# Our sample differs (March replication vs paper's XBRL sample).
PAPER_ANALYST_AUC_NO_DRIFT   = 0.6471
PAPER_ANALYST_AUC_WITH_DRIFT = 0.6509
PAPER_RANDOM_BASELINE        = 0.5000
PAPER_SAMPLE_YEARS           = "2015–2018"
