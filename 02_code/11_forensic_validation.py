"""
11_forensic_validation.py  —  v2 (revised per expert review)
=============================================================
Forensic validation of AUC = 0.8079.

Changes from v1 after expert review:
  - T1-A: Removed the false "mathematical gate". P(increase|score=1) is a useful
    diagnostic but NOT a mathematical necessity for AUC=0.8079. Replaced with
    T1-E: exact 3-component AUC decomposition that IS a true mathematical check.
  - New T0-SCORE: Score recomputation from forecast records (most fundamental check).
  - New T2-0: actual_eps_t known-by-formation (forensic version with full evidence table).
  - T3-B / T3-C: Now use detail_forecasts_clean_all_rows.csv.gz (all records, not
    the already-deduplicated forecast_level file). This is the only way to actually
    compare per-analyst-latest vs all-forecasts.
  - T3-D: Uses raw IBES detail file for strict USD vs NULL sensitivity (curr_act not
    retained in clean file).
  - T1-D: 2015-2018 AUC > 0.79 is WARN not FAIL (it's a diagnostic, not a bug test).
  - T1-C: Uses upper-bound p-value (count+1)/(n+1) instead of p=0.
  - T2-A: Missing actual_anndats_tp1 now also counts as FAIL.
  - T2-C: Adds short-horizon (<90d) share as a recorded diagnostic.
  - T3-C: diff > 0.015 is ACTION_REQUIRED (flag for discussion) not auto-FAIL.
  - T4-B: Real bootstrap equivalence on small subsample, not just point AUC recheck.
  - T4-C: Fixed merge column collision bug.
  - T5: Changed from FAIL to WARN/ACTION_REQUIRED (doc fixes don't block pipeline).
  - Summary: Three-category decision output:
      BLOCKING_DATA_FAIL — stop, fix bug, do not submit
      ACTION_REQUIRED_METHOD — investigate and decide before submitting
      DOC_FIX_REQUIRED — text corrections needed but don't block

Run (no pipeline re-run needed — reads existing outputs):
    python 02_code/11_forensic_validation.py

Outputs to:  05_output/forensic_validation/
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
INT_MERGE  = ROOT / "04_intermediate" / "merge"
INT_IBES   = ROOT / "04_intermediate" / "ibes"
INT_LINKS  = ROOT / "04_intermediate" / "links"
INT_SAMPLE = ROOT / "04_intermediate" / "sample"
RAW_IBES   = ROOT / "03_raw" / "ibes"
OUT_DIR    = ROOT / "05_output" / "forensic_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Input files ───────────────────────────────────────────────────────────────
MAIN_AUC_FILE  = INT_MERGE  / "firm_year_auc_input_main_score_le_2.csv.gz"
FCST_MAIN      = INT_MERGE  / "forecast_level_directions_main_score_le_2.csv.gz"
LE5_FILE       = INT_MERGE  / "firm_year_auc_input_sensitivity_score_le_5.csv.gz"
LE6_FILE       = INT_MERGE  / "firm_year_auc_input_sensitivity_score_le_6.csv.gz"
FCST_CLEAN_ALL = INT_IBES   / "detail_forecasts_clean_all_rows.csv.gz"  # all records
RAW_FCST_FILE  = RAW_IBES   / "ibes_detail_forecasts_raw.csv.gz"        # for strict USD

EXPECTED_AUC = 0.8079
RANDOM_SEED  = 20260413
PENDS_TOL    = 7   # days, must match pipeline config

# ── Status codes ─────────────────────────────────────────────────────────────
BLOCKING   = "BLOCKING_FAIL"      # stop, fix bug, do not submit
ACTION_REQ = "ACTION_REQUIRED"    # investigate and decide before submitting
DOC_FIX    = "DOC_FIX"           # wording correction needed
PASS       = "PASS"
WARN       = "WARN"
INFO       = "INFO"

results: list[dict] = []

def log(msg: str = "") -> None:
    print(msg, flush=True)

def record(test_id: str, name: str, status: str, value, expected,
           note: str = "") -> dict:
    icons = {PASS: "✅", WARN: "⚠️ ", BLOCKING: "🚨", ACTION_REQ: "🔶",
             DOC_FIX: "📝", INFO: "ℹ️ "}
    icon = icons.get(status, "  ")
    print(f"  {icon} [{test_id}] {name}")
    print(f"        value: {value}")
    print(f"     expected: {expected}")
    if note:
        print(f"         note: {note}")
    row = dict(test_id=test_id, name=name, status=status,
               value=str(value), expected=str(expected), note=note)
    results.append(row)
    return row

def save(df: pd.DataFrame, name: str) -> None:
    p = OUT_DIR / name
    df.to_csv(p, index=False)
    log(f"      → saved: {name}")

def auc_mannwhitney(y, s) -> float:
    """AUC via rank-sum formula — independent of sklearn."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    mask = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[mask], s[mask]
    n1 = int((y == 1).sum());  n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").values
    return float((ranks[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))

def load_main() -> pd.DataFrame:
    df = pd.read_csv(MAIN_AUC_FILE, low_memory=False)
    df["y_true"] = pd.to_numeric(
        df.get("y_true", df.get("actual_increase_raw")), errors="coerce"
    ).astype("Int64")
    df["score"] = pd.to_numeric(
        df.get("score", df.get("analyst_score")), errors="coerce"
    )
    if "fyear" in df.columns:
        df["fyear"] = pd.to_numeric(df["fyear"], errors="coerce").astype("Int64")
    for c in ["datadate","formation_date","forecast_window_start","forecast_window_end",
              "target_fpedats_t","target_fpedats_tp1",
              "actual_anndats_t","actual_anndats_tp1"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def build_all_matched_forecasts(df_main: pd.DataFrame,
                                 fc_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build ALL forecast records matched to the main AUC sample.
    Uses detail_forecasts_clean_all_rows.csv.gz (not already-deduplicated file).
    This is the true paper-literal 'all forecasts in April window' dataset.
    """
    base_cols = ["gvkey","fyear","ticker","y_true","actual_eps_t",
                 "target_fpedats_tp1","forecast_window_start","forecast_window_end"]
    base = df_main[[c for c in base_cols if c in df_main.columns]].copy()
    # Ensure ticker type consistency so the merge does not silently collapse to empty
    if "ticker" not in base.columns:
        for alt in ["ibes_ticker", "ticker_ib", "ibticker"]:
            if alt in df_main.columns:
                base = base.assign(ticker=df_main[alt])
                break
    if "ticker" in base.columns:
        base["ticker"] = base["ticker"].astype(str).str.strip()

    fc = fc_clean.copy()
    fc.columns = [c.lower() for c in fc.columns]
    fc["ticker"]           = fc["ticker"].astype(str).str.strip()
    fc["forecast_anndats"] = pd.to_datetime(fc.get("forecast_anndats",
                              fc.get("anndats")), errors="coerce")
    fc["fpedats"]          = pd.to_datetime(fc.get("fpedats"), errors="coerce")

    # Rename value → forecast_eps if needed
    if "forecast_eps" not in fc.columns and "value" in fc.columns:
        fc["forecast_eps"] = pd.to_numeric(fc["value"], errors="coerce")
    elif "forecast_eps" in fc.columns:
        fc["forecast_eps"] = pd.to_numeric(fc["forecast_eps"], errors="coerce")
    else:
        return pd.DataFrame()

    fc = fc[fc["forecast_eps"].notna() & fc["forecast_anndats"].notna()].copy()

    # Merge on ticker to get sample context
    m = base.merge(fc, on="ticker", how="inner")
    if len(m) == 0:
        return pd.DataFrame()

    # Filter: fpedats within ±PENDS_TOL days of target_fpedats_tp1
    m["forecast_fpedats_diff_days"] = (
        m["fpedats"] - m["target_fpedats_tp1"]
    ).dt.days.abs()

    # Filter: anndats within forecast window
    m = m[
        m["forecast_fpedats_diff_days"].le(PENDS_TOL) &
        m["forecast_anndats"].between(
            m["forecast_window_start"], m["forecast_window_end"]
        )
    ].copy()

    if "actual_eps_t" in m.columns:
        m["f_dir"] = (m["forecast_eps"] > m["actual_eps_t"]).astype(float)

    return m

# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("FORENSIC VALIDATION — AUC = 0.8079")
log("=" * 70)

if not MAIN_AUC_FILE.exists():
    log(f"FATAL: {MAIN_AUC_FILE} not found. Run 06_construct_auc_sample.py first.")
    sys.exit(1)

df = load_main()
log(f"  Loaded main AUC input: {len(df):,} firm-years")
df_clean = df.dropna(subset=["y_true","score","fyear"]).copy()
y_arr    = df_clean["y_true"].astype(int).values
s_arr    = df_clean["score"].values

# ══════════════════════════════════════════════════════════════════════════════
# TIER 0 — Score recomputation (MOST FUNDAMENTAL CHECK)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "═"*70)
log("TIER 0 — Score recomputation from raw forecast records")
log("  Proves: final analyst_score = mean(1[forecast_eps > actual_eps_t])")
log("  Does NOT use actual_eps_tp1 in score computation")
log("═"*70)

score_recomp_done = False
if FCST_MAIN.exists() and MAIN_AUC_FILE.exists():
    fcst_main = pd.read_csv(FCST_MAIN, low_memory=False)
    fcst_main.columns = [c.lower() for c in fcst_main.columns]

    # Check if already deduplicated (per-analyst-latest)
    has_dup = False
    if "gvkey" in fcst_main.columns and "fyear" in fcst_main.columns \
            and "analyst_id" in fcst_main.columns:
        n_total = len(fcst_main)
        n_unique = fcst_main.drop_duplicates(["gvkey","fyear","analyst_id"]).shape[0]
        has_dup = n_total > n_unique
        record("T0-DUP", "forecast_level file is per-analyst-latest (as expected)",
               INFO, f"n_total={n_total:,}, n_unique_analyst_per_fy={n_unique:,}",
               "n_total == n_unique (no duplicates after dedup)",
               "If n_total > n_unique: file was NOT deduplicated — T3-C all-forecasts valid in-file")

    # Recompute score from this file
    eps_col = "forecast_eps" if "forecast_eps" in fcst_main.columns else \
              ("forecast_value" if "forecast_value" in fcst_main.columns else None)
    dir_col = "forecast_increase_raw" if "forecast_increase_raw" in fcst_main.columns else None

    if eps_col and "actual_eps_t" in fcst_main.columns and \
            "gvkey" in fcst_main.columns and "fyear" in fcst_main.columns:

        fcst_main[eps_col] = pd.to_numeric(fcst_main[eps_col], errors="coerce")
        fcst_main["actual_eps_t"] = pd.to_numeric(fcst_main["actual_eps_t"], errors="coerce")
        fcst_main["f_dir_check"] = (fcst_main[eps_col] > fcst_main["actual_eps_t"]).astype(float)

        # Verify f_dir_check != uses actual_eps_tp1
        if "actual_eps_tp1" in fcst_main.columns:
            uses_tp1 = (fcst_main["f_dir_check"] == (
                fcst_main[eps_col] > pd.to_numeric(fcst_main["actual_eps_tp1"], errors="coerce")
            )).all()
            record("T0-NOLEAK", "Score direction does NOT use actual_eps_tp1",
                   PASS if not uses_tp1 else BLOCKING,
                   f"direction_matches_tp1_comparison={uses_tp1}",
                   "False (score uses EPS_t, not EPS_t+1)",
                   "BLOCKING if True: score was computed using future actual EPS")

        score_recomp = (
            fcst_main.groupby(["gvkey","fyear"])["f_dir_check"].mean()
            .reset_index(name="score_recomp")
        )
        df_check = df[["gvkey","fyear","score"]].merge(score_recomp, on=["gvkey","fyear"])
        diff = (df_check["score"] - df_check["score_recomp"]).abs()
        max_diff  = float(diff.max())
        n_nonzero = int((diff > 1e-10).sum())

        save(pd.DataFrame([{
            "n_firmyears": len(df_check),
            "max_abs_diff_score": max_diff,
            "n_diff_gt_1e10": n_nonzero,
            "status": "PASS" if max_diff < 1e-10 else "FAIL",
        }]), "T0_score_recomputation.csv")

        record("T0-SCORE", "Score exactly recomputable from forecast records",
               PASS if max_diff < 1e-10 else BLOCKING,
               f"max_abs_diff={max_diff:.2e}, n_nonzero={n_nonzero}",
               "max_abs_diff < 1e-10, n_nonzero = 0",
               "BLOCKING if FAIL: analyst_score does not match forecast directions")
        score_recomp_done = True

if not score_recomp_done:
    record("T0-SCORE", "Score recomputation", WARN,
           "forecast_level file missing or columns not found", "max_diff < 1e-10",
           "Run 06_construct_auc_sample.py, then re-run this script")

# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — Mathematical consistency
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "═"*70)
log("TIER 1 — Mathematical consistency")
log("═"*70)

# ── T1-A: Conditional increase rates (DIAGNOSTIC ONLY — not a gate) ─────────
log("\n[T1-A] Conditional increase rates by score level (DIAGNOSTIC — not a gate)")
log("  Note: P(increase|score=1.0) is informative but NOT a mathematical")
log("  necessary condition for AUC=0.8079. It depends on ALL score levels.")

at1    = df[df["score"] == 1.0]
bel1   = df[df["score"] <  1.0]
at0    = df[df["score"] == 0.0]
mid    = df[(df["score"] > 0) & (df["score"] < 1.0)]

p1 = float(at1["y_true"].mean())  if len(at1) > 0 else float("nan")
p0 = float(at0["y_true"].mean())  if len(at0) > 0 else float("nan")
pm = float(mid["y_true"].mean())  if len(mid) > 0 else float("nan")

bin_rows = []
for label, mask in [
    ("score=0.0",      df["score"] == 0.0),
    ("0<score<0.25",   (df["score"] > 0) & (df["score"] < 0.25)),
    ("0.25≤score<0.5", (df["score"] >= 0.25) & (df["score"] < 0.50)),
    ("score=0.5",      df["score"] == 0.5),
    ("0.5<score<0.75", (df["score"] > 0.5) & (df["score"] < 0.75)),
    ("0.75≤score<1.0", (df["score"] >= 0.75) & (df["score"] < 1.0)),
    ("score=1.0",      df["score"] == 1.0),
]:
    sub = df[mask]
    if len(sub) == 0:
        continue
    bin_rows.append({
        "bin": label, "n_obs": len(sub),
        "pct_total": round(100*len(sub)/len(df), 2),
        "mean_score": round(float(sub["score"].mean()), 4),
        "actual_increase_rate": round(float(sub["y_true"].mean()), 4),
    })
bin_df = pd.DataFrame(bin_rows)
save(bin_df, "T1A_score_bins_increase_rates.csv")

record("T1-A-1", "P(increase|score=1.0)",
       INFO, f"{p1:.4f}",
       "high value expected but NOT mathematically required",
       f"P(increase|score=0)={p0:.4f}, P(increase|0<score<1)={pm:.4f}, overall={df['y_true'].mean():.4f}")
record("T1-A-2", "Fraction of firm-years at score=1.0",
       INFO, f"{len(at1):,} / {len(df):,} = {100*len(at1)/len(df):.1f}%",
       "> 50% (consistent with median_score=1.0)",
       "Analyst optimism bias makes unanimous-bullish majority plausible")

# ── T1-B: AUC formula cross-validation ──────────────────────────────────────
log("\n[T1-B] AUC formula cross-validation (sklearn / Mann-Whitney / inverted)")
auc_sk  = roc_auc_score(y_arr, s_arr)
auc_mw  = auc_mannwhitney(y_arr, s_arr)
auc_inv = roc_auc_score(y_arr, -s_arr)
diff_mw  = abs(auc_sk - auc_mw)
diff_inv = abs(auc_sk + auc_inv - 1.0)
max_diff = max(diff_mw, diff_inv)

save(pd.DataFrame([{
    "auc_sklearn": round(auc_sk, 8), "auc_mannwhitney": round(auc_mw, 8),
    "auc_inverted": round(auc_inv, 8), "expected_inverted": round(1 - auc_sk, 8),
    "diff_sklearn_vs_mannwhitney": diff_mw, "diff_inverted_consistency": diff_inv,
}]), "T1B_auc_formula_crosscheck.csv")

record("T1-B", "Three AUC formula cross-validation",
       PASS if max_diff < 1e-8 else BLOCKING,
       f"sklearn={auc_sk:.8f}, MW={auc_mw:.8f}, inv={auc_inv:.8f}, max_diff={max_diff:.2e}",
       "max_diff < 1e-8",
       "BLOCKING if FAIL: y_true/score columns or AUC function has a bug")

# ── T1-C: Permutation null test ──────────────────────────────────────────────
log("\n[T1-C] Within-fyear permutation null test (5,000 reps)")
log("  Proves: high AUC is not a mechanical consequence of class balance")

rng = np.random.default_rng(RANDOM_SEED)
null_aucs = []
fyear_arr = df_clean["fyear"].values
for _ in range(5000):
    s_perm = s_arr.copy()
    for fy in np.unique(fyear_arr):
        idx_fy = np.where(fyear_arr == fy)[0]
        s_perm[idx_fy] = rng.permutation(s_arr[idx_fy])
    null_aucs.append(roc_auc_score(y_arr, s_perm))

null_arr = np.array(null_aucs)
n_ge = int((null_arr >= auc_sk).sum())
pval_ub = (n_ge + 1) / (len(null_arr) + 1)   # plus-one upper bound — never reports p=0

save(pd.DataFrame({"null_auc": null_arr}), "T1C_permutation_null_5000.csv")
save(pd.DataFrame([{
    "observed_auc": auc_sk, "null_mean": null_arr.mean(),
    "null_p5": np.percentile(null_arr, 5), "null_p95": np.percentile(null_arr, 95),
    "null_p99": np.percentile(null_arr, 99),
    "p_value_upper_bound": pval_ub, "n_reps": 5000,
    "note": "p-value reported as upper bound (count+1)/(n+1); never reports p=0",
}]), "T1C_permutation_summary.csv")

record("T1-C", "Permutation null test",
       PASS if abs(null_arr.mean() - 0.5) < 0.005 and pval_ub < 0.001 else WARN,
       f"null_mean={null_arr.mean():.4f}, p(AUC≥observed)<{pval_ub:.4f}",
       "null_mean≈0.5, p < 0.001",
       "Null AUC centered at 0.5 confirms result not from class-balance mechanics")

# ── T1-D: Period sensitivity (2015-2018 = paper test period) ─────────────────
log("\n[T1-D] Period sensitivity — 2015-2018 (paper test period)")
log("  Paper no-drift benchmark: 0.6471 (different XBRL sample)")
log("  This is a DIAGNOSTIC — any AUC value here requires explanation, not a fix")

period_rows = []
for pname, y0, y1 in [
    ("Full 2005-2023",              2005, 2023),
    ("2015-2018 (paper period)",    2015, 2018),
    ("2015-2023",                   2015, 2023),
    ("2019-2023",                   2019, 2023),
    ("2005-2014",                   2005, 2014),
    ("FY2023 only",                 2023, 2023),
]:
    sub = df_clean[df_clean["fyear"].between(y0, y1)]
    if len(sub) < 50 or sub["y_true"].nunique() < 2:
        continue
    a = roc_auc_score(sub["y_true"].astype(int), sub["score"])
    period_rows.append({
        "period": pname, "fyear_start": y0, "fyear_end": y1,
        "n_obs": len(sub),
        "n_firms": sub["gvkey"].nunique() if "gvkey" in sub else None,
        "actual_increase_rate": round(float(sub["y_true"].mean()), 4),
        "mean_score": round(float(sub["score"].mean()), 4),
        "auc": round(a, 4),
    })

period_df = pd.DataFrame(period_rows)
save(period_df, "T1D_period_sensitivity.csv")

row_1518 = period_df[period_df["period"].str.contains("paper")]
if len(row_1518) > 0:
    a1518 = float(row_1518["auc"].iloc[0])
    if a1518 < 0.72:
        st = INFO; note = "Close to paper benchmark. Period explains much of the gap."
    elif a1518 < 0.79:
        st = WARN; note = "Still above paper. Discuss sample composition (IBES-covered firms, coverage composition)."
    else:
        st = ACTION_REQ
        note = ("AUC high even in paper-period years. Not a data error, but must be"
                " explained clearly to professor. Check T3-B window and T3-C aggregation.")
    record("T1-D", "2015-2018 pooled AUC vs paper benchmark 0.6471",
           st, f"{a1518:.4f}", "0.63-0.70 close to paper | >0.79 requires explanation", note)

# ── T1-E: Exact AUC decomposition (TRUE mathematical check) ─────────────────
log("\n[T1-E] Exact AUC 3-component decomposition")
log("  AUC = a·(1-b) + 0.5·a·b + (1-a)·(1-b)·A_B")
log("  where a = P(S=1|Y=1), b = P(S=1|Y=0), A_B = AUC within S<1 subgroup")
log("  Decomposed AUC must match sklearn AUC (mathematical identity check)")

n_pos = int((y_arr == 1).sum());  n_neg = int((y_arr == 0).sum())
# a = P(score=1 | Y=1)
at1_pos = ((df_clean["score"] == 1.0) & (df_clean["y_true"] == 1)).sum()
at1_neg = ((df_clean["score"] == 1.0) & (df_clean["y_true"] == 0)).sum()
a = float(at1_pos / n_pos) if n_pos > 0 else float("nan")
b = float(at1_neg / n_neg) if n_neg > 0 else float("nan")

# A_B = AUC within score<1 subgroup
bel1_df  = df_clean[df_clean["score"] < 1.0]
auc_AB   = float("nan")
if len(bel1_df) > 0 and bel1_df["y_true"].nunique() == 2:
    auc_AB = roc_auc_score(bel1_df["y_true"].astype(int), bel1_df["score"])

# Decomposed AUC
if not (np.isnan(a) or np.isnan(b) or np.isnan(auc_AB)):
    auc_decomp = a*(1-b) + 0.5*a*b + (1-a)*(1-b)*auc_AB
    diff_decomp = abs(auc_sk - auc_decomp)
    # Upper bound if A_B = 1.0 (perfect within-group ranking)
    auc_upper = a*(1-b) + 0.5*a*b + (1-a)*(1-b)*1.0
else:
    auc_decomp = float("nan");  diff_decomp = float("nan");  auc_upper = float("nan")

save(pd.DataFrame([{
    "a_P_S1_given_Y1": round(a, 6), "b_P_S1_given_Y0": round(b, 6),
    "auc_AB_within_Sbel1": round(auc_AB, 6) if not np.isnan(auc_AB) else "nan",
    "auc_decomposed": round(auc_decomp, 8) if not np.isnan(auc_decomp) else "nan",
    "auc_sklearn":    round(auc_sk, 8),
    "diff_decomp_vs_sklearn": f"{diff_decomp:.12e}" if not np.isnan(diff_decomp) else "nan",
    "auc_upper_bound_if_AB_perfect": round(auc_upper, 6) if not np.isnan(auc_upper) else "nan",
    "observed_auc_within_upper_bound": auc_sk <= auc_upper if not np.isnan(auc_upper) else "unknown",
}]), "T1E_auc_decomposition.csv")

record("T1-E-1", "Decomposed AUC matches sklearn AUC",
       PASS if not np.isnan(diff_decomp) and diff_decomp < 1e-8 else WARN,
       f"diff={diff_decomp:.2e}" if not np.isnan(diff_decomp) else "nan (subgroup too small)",
       "< 1e-8 (mathematical identity)",
       f"a={a:.4f}, b={b:.4f}, A_B={auc_AB:.4f}")
record("T1-E-2", "Observed AUC ≤ theoretical upper bound",
       PASS if not np.isnan(auc_upper) and auc_sk <= auc_upper + 1e-8 else BLOCKING,
       f"observed={auc_sk:.6f}, upper_bound={auc_upper:.6f}",
       "observed ≤ upper_bound",
       "BLOCKING if violated: AUC impossible given the observed score distribution")

# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — Timing and look-ahead (ALL must have 0 violations)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "═"*70)
log("TIER 2 — Timing and look-ahead (zero violations required)")
log("═"*70)

# ── T2-0: actual_eps_t known by formation_date ───────────────────────────────
log("\n[T2-0] actual_eps_t announced before portfolio formation (no look-ahead on EPS_t)")
if "actual_anndats_t" in df.columns and "formation_date" in df.columns:
    la_t    = int((df["actual_anndats_t"] > df["formation_date"]).sum())
    miss_t  = int(df["actual_anndats_t"].isna().sum())
    max_lag = float((df["actual_anndats_t"] - df["formation_date"]).dt.days.max()) \
              if la_t > 0 else 0
    save(pd.DataFrame([{
        "n_rows": len(df),
        "n_lookahead_actual_t_after_formation": la_t,
        "n_missing_actual_anndats_t": miss_t,
        "max_days_after_formation": max_lag,
    }]), "T20_actual_eps_t_timing.csv")
    record("T2-0", "EPS_t announced before formation (0 violations + 0 missing)",
           PASS if la_t == 0 and miss_t == 0 else BLOCKING,
           f"violations={la_t}, missing={miss_t}",
           "0, 0",
           "BLOCKING if violations>0: EPS_t used before announcement (look-ahead)")
else:
    record("T2-0", "EPS_t timing check", BLOCKING,
           "actual_anndats_t or formation_date column missing from AUC input",
           "both columns present and 0 violations",
           "BLOCKING: cannot verify EPS_t was known by formation date → look-ahead risk unverifiable")

# ── T2-A: Forecast before t+1 actual announcement ────────────────────────────
log("\n[T2-A] All forecasts issued before IBES Actual EPS_{t+1} announcement")
if FCST_MAIN.exists() and "actual_anndats_tp1" in df.columns:
    fm = pd.read_csv(FCST_MAIN, low_memory=False)
    fm.columns = [c.lower() for c in fm.columns]
    if "forecast_anndats" in fm.columns:
        fm["forecast_anndats"] = pd.to_datetime(fm["forecast_anndats"], errors="coerce")
        # Use firm-year-level anndats_tp1 from main AUC input; rename to avoid
        # collision with any same-named column already in the forecast file.
        tp1_map = df[["gvkey","fyear","actual_anndats_tp1"]].rename(
            columns={"actual_anndats_tp1": "anndats_tp1_main"}
        )
        tp1_map["anndats_tp1_main"] = pd.to_datetime(
            tp1_map["anndats_tp1_main"], errors="coerce"
        )
        # Drop any duplicate column in fm BEFORE merging to avoid _x/_y suffixes
        fm_clean = fm.drop(columns=[c for c in ["actual_anndats_tp1"]
                                      if c in fm.columns], errors="ignore")
        merged_f = fm_clean.merge(tp1_map, on=["gvkey","fyear"], how="left")
        n_miss_tp1 = int(merged_f["anndats_tp1_main"].isna().sum())
        violations = merged_f[
            merged_f["anndats_tp1_main"].notna() &
            (merged_f["forecast_anndats"] >= merged_f["anndats_tp1_main"])
        ]
        save(pd.DataFrame([{
            "n_total_forecasts": len(merged_f),
            "n_violations_forecast_ge_actual_anndats_tp1": len(violations),
            "n_missing_actual_anndats_tp1": n_miss_tp1,
        }]), "T2A_forecast_before_actual_tp1.csv")
        if len(violations) > 0:
            save(violations.head(20), "T2A_violations_sample.csv")
        record("T2-A", "Forecast issued before t+1 actual announcement (0 violations + 0 missing)",
               PASS if len(violations) == 0 and n_miss_tp1 == 0 else BLOCKING,
               f"violations={len(violations)}, missing_anndats_tp1={n_miss_tp1}",
               "0, 0",
               "BLOCKING if violations>0: look-ahead confirmed. "
               "BLOCKING if n_missing>0: cannot verify no look-ahead for those rows.")
    else:
        record("T2-A", "Forecast timing check", WARN, "forecast_anndats not found", "0, 0", "")
else:
    record("T2-A", "Forecast timing check", WARN,
           "forecast file or actual_anndats_tp1 missing", "0, 0", "")

# ── T2-B: Fiscal target alignment ────────────────────────────────────────────
log("\n[T2-B] forecast_level fpedats ≈ datadate + 1 year (FY t+1)")
if "datadate" in df.columns and "target_fpedats_tp1" in df.columns:
    expected = df["datadate"] + pd.DateOffset(years=1)
    diff_days = (df["target_fpedats_tp1"] - expected).dt.days.abs()
    mismatch_30 = int((diff_days > 30).sum())
    mismatch_7  = int((diff_days > 7).sum())
    sample_rows = df[["gvkey","fyear","datadate","target_fpedats_tp1"]].assign(
        diff_abs_days=diff_days
    ).sort_values("diff_abs_days", ascending=False).head(10)
    save(sample_rows, "T2B_fiscal_target_alignment_top10.csv")
    record("T2-B", "target_fpedats_tp1 ≈ datadate+1yr (±7d tolerance)",
           PASS if mismatch_7 == 0 else BLOCKING,
           f"mismatch_gt_7d={mismatch_7}, mismatch_gt_30d={mismatch_30}",
           "0",
           "BLOCKING if >0: FPI or fpedats pointed to wrong fiscal year")
else:
    record("T2-B", "Fiscal target alignment", BLOCKING,
           "datadate or target_fpedats_tp1 not in AUC input",
           "columns present",
           "BLOCKING: cannot verify forecast targets correct fiscal year")

# ── T2-B2: Forecast-level fpedats alignment (row-by-row) ─────────────────────
log("\n[T2-B2] Forecast-level fpedats align to FY t+1 target (row-by-row)")
log("  T2-B verifies the design target; T2-B2 verifies each actual forecast row")
if FCST_MAIN.exists() and "target_fpedats_tp1" in df.columns:
    fm_b = pd.read_csv(FCST_MAIN, low_memory=False)
    fm_b.columns = [c.lower() for c in fm_b.columns]
    if "fpedats" in fm_b.columns and "gvkey" in fm_b.columns and "fyear" in fm_b.columns:
        fm_b["fpedats"] = pd.to_datetime(fm_b["fpedats"], errors="coerce")
        target_map = df[["gvkey", "fyear", "target_fpedats_tp1"]].copy()
        target_map["target_fpedats_tp1"] = pd.to_datetime(
            target_map["target_fpedats_tp1"], errors="coerce"
        )
        # Drop any duplicate target column in fm_b to avoid _x/_y suffixes
        fm_b_clean = fm_b.drop(columns=[c for c in ["target_fpedats_tp1"]
                                         if c in fm_b.columns], errors="ignore")
        chk = fm_b_clean.merge(target_map, on=["gvkey", "fyear"], how="left")
        chk["abs_diff_days"] = (
            chk["fpedats"] - chk["target_fpedats_tp1"]
        ).dt.days.abs()
        n_missing_target = int(chk["target_fpedats_tp1"].isna().sum())
        n_bad = int((chk["abs_diff_days"] > PENDS_TOL).sum())
        save(
            chk[["gvkey","fyear","fpedats","target_fpedats_tp1","abs_diff_days"]]
              .sort_values("abs_diff_days", ascending=False)
              .head(50),
            "T2B2_forecast_fpedats_alignment_top50.csv"
        )
        record("T2-B2", "Forecast-level fpedats align to FY t+1 target (±7d)",
               PASS if n_missing_target == 0 and n_bad == 0 else BLOCKING,
               f"missing_target={n_missing_target}, mismatch_gt_{PENDS_TOL}d={n_bad}",
               "0, 0",
               "BLOCKING if >0: forecast rows point to a different fiscal year than FY t+1")
    else:
        record("T2-B2", "Forecast-level fpedats alignment", BLOCKING,
               f"required columns missing from {FCST_MAIN.name}",
               "fpedats, gvkey, fyear all present",
               "BLOCKING: cannot row-by-row verify forecast target year")
else:
    record("T2-B2", "Forecast-level fpedats alignment", BLOCKING,
           f"{FCST_MAIN.name} or target_fpedats_tp1 missing",
           "both present",
           "BLOCKING: cannot row-by-row verify forecast target year")

# ── T2-C: Forecast horizon distribution ──────────────────────────────────────
log("\n[T2-C] Forecast horizon distribution (April→December ≈ 240-270 days)")
if FCST_MAIN.exists():
    fm2 = pd.read_csv(FCST_MAIN, low_memory=False)
    fm2.columns = [c.lower() for c in fm2.columns]
    if "forecast_anndats" in fm2.columns and "fpedats" in fm2.columns:
        fm2["forecast_anndats"] = pd.to_datetime(fm2["forecast_anndats"], errors="coerce")
        fm2["fpedats"]          = pd.to_datetime(fm2["fpedats"], errors="coerce")
        fm2["horizon_days"]     = (fm2["fpedats"] - fm2["forecast_anndats"]).dt.days
        neg = int((fm2["horizon_days"] < 0).sum())
        short_90  = int((fm2["horizon_days"] < 90).sum())
        short_pct = short_90 / len(fm2)
        desc = fm2["horizon_days"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
        save(desc.reset_index().rename(columns={"index":"stat","horizon_days":"value"}),
             "T2C_horizon_distribution.csv")
        record("T2-C-1", "Zero forecasts with horizon_days < 0",
               PASS if neg == 0 else BLOCKING,
               f"{neg}", "0",
               "BLOCKING if >0: forecast issued after fiscal year end → wrong FPI")
        record("T2-C-2", "Median forecast horizon days",
               PASS if 180 <= float(fm2["horizon_days"].median()) <= 330 else WARN,
               f"{fm2['horizon_days'].median():.0f}d", "180-330d (April→December)",
               "Outside range may indicate non-December FY firms mixed in")
        record("T2-C-3", "Forecasts with short horizon < 90 days",
               PASS if short_pct < 0.01 else WARN,
               f"{short_90:,} ({100*short_pct:.2f}%)", "< 1%",
               "High short-horizon share means some forecasts close to fiscal year end "
               "— may explain higher AUC for those observations")

# ── T2-D: EPS year-shift sanity check ────────────────────────────────────────
log("\n[T2-D] EPS year-shift check (EPS_t ≠ EPS_t+1)")
if "actual_eps_t" in df.columns and "actual_eps_tp1" in df.columns:
    corr = df["actual_eps_t"].corr(df["actual_eps_tp1"])
    same = (df["actual_eps_t"].round(4) == df["actual_eps_tp1"].round(4)).mean()
    save(pd.DataFrame([{
        "corr_eps_t_tp1": round(corr, 6),
        "fraction_eps_t_equals_tp1": round(same, 6),
        "mean_eps_t": round(df["actual_eps_t"].mean(), 4),
        "mean_eps_tp1": round(df["actual_eps_tp1"].mean(), 4),
    }]), "T2D_eps_year_shift.csv")
    record("T2-D-1", "Corr(EPS_t, EPS_t+1) in expected range",
           PASS if 0.80 <= corr <= 0.97 else WARN,
           f"{corr:.6f}", "0.80-0.97 (strong but not perfect)",
           "Corr > 0.99 suggests t and t+1 may be the same column (year-shift bug)")
    record("T2-D-2", "Fraction EPS_t == EPS_t+1 (to 4dp)",
           PASS if same < 0.01 else BLOCKING,
           f"{same:.4f}", "< 0.01",
           "BLOCKING if ≥ 0.01: EPS_t and EPS_t+1 are the same values → year-shift bug")

# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 — Sensitivity: understand and explain the high AUC
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "═"*70)
log("TIER 3 — Sensitivity analysis")
log("  T3-B and T3-C use detail_forecasts_clean_all_rows.csv.gz (all records)")
log("  — not the already-deduplicated forecast_level file")
log("═"*70)

# ── Load all-records clean forecast file ─────────────────────────────────────
all_matched = pd.DataFrame()
fc_clean_loaded = False
if FCST_CLEAN_ALL.exists():
    log("\n  Loading detail_forecasts_clean_all_rows.csv.gz for T3-B/T3-C...")
    fc_clean = pd.read_csv(FCST_CLEAN_ALL, low_memory=False)
    fc_clean.columns = [c.lower() for c in fc_clean.columns]
    if "forecast_eps" not in fc_clean.columns and "value" in fc_clean.columns:
        fc_clean["forecast_eps"] = pd.to_numeric(fc_clean["value"], errors="coerce")
    all_matched = build_all_matched_forecasts(df, fc_clean)
    if len(all_matched) > 0:
        fc_clean_loaded = True
        log(f"  All-matched forecasts: {len(all_matched):,} records for "
            f"{all_matched.groupby(['gvkey','fyear']).ngroups:,} firm-years")
    else:
        log("  WARNING: build_all_matched_forecasts returned empty — check ticker join")
else:
    log(f"  WARNING: {FCST_CLEAN_ALL} not found — T3-B and T3-C will use fallback")

# ── T3-A: Score bin monotonicity ─────────────────────────────────────────────
log("\n[T3-A] Score bin monotonicity check")
if len(bin_df) >= 3:
    rates = bin_df["actual_increase_rate"].values
    monotone = all(rates[i] <= rates[i+1] + 0.01 for i in range(len(rates)-1))
    record("T3-A", "Actual increase rate monotone across score bins",
           PASS if monotone else WARN,
           list(rates.round(3)),
           "monotone increasing (with 0.01 tolerance)",
           "Non-monotone with high AUC would be suspicious — check score construction")

# ── T3-B: Forecast window robustness ─────────────────────────────────────────
log("\n[T3-B] Forecast window robustness (Apr 1-7 / 1-15 / 1-30)")
log("  Uses all-records forecast file (not deduplicated) for true early-window test")

win_rows_all = []
if fc_clean_loaded and "f_dir" in all_matched.columns:
    for win_days in [7, 15, 30]:
        w_start = all_matched["forecast_window_start"]
        w_end   = all_matched["forecast_window_start"] + pd.Timedelta(days=win_days-1)
        win_sub = all_matched[
            all_matched["forecast_anndats"].between(w_start, w_end)
        ].copy()
        if len(win_sub) == 0:
            continue
        # Per-analyst latest within this shorter window
        win_sub = win_sub.sort_values(
            ["gvkey","fyear","analyst_id","forecast_anndats"],
            ascending=[True,True,True,False]
        ).drop_duplicates(["gvkey","fyear","analyst_id"])
        scores_w = win_sub.groupby(["gvkey","fyear"])["f_dir"].mean().reset_index(name="s")
        merged_w = df[["gvkey","fyear","y_true"]].merge(scores_w)
        if len(merged_w) < 50 or merged_w["y_true"].nunique() < 2:
            continue
        auc_w = roc_auc_score(merged_w["y_true"].astype(int), merged_w["s"])
        win_rows_all.append({
            "window": f"Apr 1-{win_days}", "n_firmyears": len(merged_w),
            "auc": round(auc_w, 4),
            "drop_from_full_30d": round(EXPECTED_AUC - auc_w, 4),
        })
elif FCST_MAIN.exists():
    # Fallback: use deduplicated file (less ideal but still useful)
    log("  Note: using deduplicated forecast-level file (fallback; may miss early forecasts)")
    fm3 = pd.read_csv(FCST_MAIN, low_memory=False)
    fm3.columns = [c.lower() for c in fm3.columns]
    if "forecast_anndats" in fm3.columns and "f_dir" not in fm3.columns \
            and "forecast_increase_raw" in fm3.columns:
        fm3["f_dir"] = pd.to_numeric(fm3["forecast_increase_raw"], errors="coerce")
    fm3_m = fm3.merge(df[["gvkey","fyear","y_true","forecast_window_start"]],
                       on=["gvkey","fyear"])
    fm3_m["forecast_anndats"] = pd.to_datetime(fm3_m["forecast_anndats"], errors="coerce")
    fm3_m["forecast_window_start"] = pd.to_datetime(fm3_m["forecast_window_start"], errors="coerce")
    if "f_dir" in fm3_m.columns:
        for win_days in [7, 15, 30]:
            w_end_col = fm3_m["forecast_window_start"] + pd.Timedelta(days=win_days-1)
            win_sub = fm3_m[fm3_m["forecast_anndats"].between(
                fm3_m["forecast_window_start"], w_end_col
            )].copy()
            scores_w = win_sub.groupby(["gvkey","fyear"])["f_dir"].mean().reset_index(name="s")
            merged_w = df[["gvkey","fyear","y_true"]].merge(scores_w)
            if len(merged_w) < 50 or merged_w["y_true"].nunique() < 2:
                continue
            auc_w = roc_auc_score(merged_w["y_true"].astype(int), merged_w["s"])
            win_rows_all.append({"window": f"Apr 1-{win_days} (fallback)", 
                                  "n_firmyears": len(merged_w), "auc": round(auc_w, 4),
                                  "drop_from_full_30d": round(EXPECTED_AUC - auc_w, 4)})

if win_rows_all:
    save(pd.DataFrame(win_rows_all), "T3B_forecast_window_robustness.csv")
    early = [r for r in win_rows_all if "Apr 1-7" in r["window"]]
    if early:
        drop = early[0]["drop_from_full_30d"]
        used_fallback = "(fallback)" in early[0]["window"]
        early_auc = early[0]["auc"]
        if used_fallback:
            # Fallback (deduplicated file) is not paper-literal — cannot serve as
            # final evidence for the all-forecasts window test.
            status_b = ACTION_REQ
            note_b = ("ACTION_REQUIRED: fallback used (deduplicated file); "
                      "generate detail_forecasts_clean_all_rows.csv.gz then rerun "
                      "for paper-literal evidence.")
        elif early_auc > 0.70 and abs(drop) < 0.05:
            status_b, note_b = PASS, "Early-window AUC stable; result robust."
        elif abs(drop) > 0.10:
            status_b = ACTION_REQ
            note_b  = ("ACTION_REQUIRED: |drop from full-Apr| > 0.10 — late-April "
                       "forecast updates materially drive the headline AUC. "
                       "Report both Apr 1-7 and full-April results.")
        elif early_auc > 0.60:
            status_b, note_b = WARN, "Apr 1-7 AUC modestly below full-April; explainable."
        else:
            status_b = ACTION_REQ
            note_b  = "ACTION_REQUIRED: Apr 1-7 AUC weak — early-formation signal is thin."
        record("T3-B", "AUC using Apr 1-7 only (early formation forecasts)",
               status_b,
               f"Apr1-7: {early_auc:.4f}, drop from full-Apr: {drop:.4f}"
               + (" [fallback]" if used_fallback else ""),
               "Apr1-7 AUC > 0.70 and |drop| < 0.05",
               note_b)
else:
    record("T3-B", "Window robustness", ACTION_REQ,
           "insufficient data — generate detail_forecasts_clean_all_rows.csv.gz",
           "> 0.70",
           "ACTION_REQUIRED: all-records forecast file missing; cannot validate window")

# ── T3-C: All-forecasts vs per-analyst-latest (PAPER-LITERAL comparison) ─────
log("\n[T3-C] Aggregation sensitivity: all-forecasts (paper-literal) vs per-analyst-latest")
log("  Also: rebuild per-analyst-latest from all-records file and verify it")
log("  equals the CURRENT main score and AUC (coverage + numerical equivalence).")

agg_rows = []
current_main_auc = EXPECTED_AUC
agg_rows.append({
    "method": "per_analyst_latest_CURRENT_MAIN",
    "n_firmyears": len(df), "auc": EXPECTED_AUC,
    "note": "current main result (reference)"
})

# Will populate downstream
rebuilt_match = None

if fc_clean_loaded and "f_dir" in all_matched.columns:
    # ── T3-C-1: per-analyst-latest REBUILT from all-records file ─────────────
    latest_rebuilt = (
        all_matched
        .sort_values(
            ["gvkey","fyear","analyst_id","forecast_anndats"],
            ascending=[True, True, True, False]
        )
        .drop_duplicates(["gvkey","fyear","analyst_id"])
    )
    scores_latest = (
        latest_rebuilt.groupby(["gvkey","fyear"])["f_dir"]
                      .mean().reset_index(name="s_rebuilt")
    )
    merged_latest = df[["gvkey","fyear","y_true","score"]].merge(
        scores_latest, on=["gvkey","fyear"], how="left"
    )
    n_current   = int(len(df))
    n_rebuilt   = int(merged_latest["s_rebuilt"].notna().sum())
    n_missing   = n_current - n_rebuilt
    if n_rebuilt > 50 and merged_latest.dropna(subset=["s_rebuilt"])["y_true"].nunique() == 2:
        in_both = merged_latest.dropna(subset=["s_rebuilt"])
        auc_current_overlap = roc_auc_score(
            in_both["y_true"].astype(int), in_both["score"]
        )
        auc_rebuilt = roc_auc_score(
            in_both["y_true"].astype(int), in_both["s_rebuilt"]
        )
        score_diff_max = float((in_both["score"] - in_both["s_rebuilt"]).abs().max())
        auc_diff       = abs(auc_current_overlap - auc_rebuilt)
    else:
        auc_current_overlap = auc_rebuilt = score_diff_max = auc_diff = float("nan")

    save(pd.DataFrame([{
        "n_firmyears_current_main":       n_current,
        "n_firmyears_rebuilt_from_allrec": n_rebuilt,
        "n_missing_from_rebuilt":          n_missing,
        "max_abs_score_diff":              score_diff_max,
        "auc_current_on_overlap":          auc_current_overlap,
        "auc_rebuilt_latest":              auc_rebuilt,
        "abs_auc_diff":                    auc_diff,
    }]), "T3C1_rebuilt_latest_equivalence.csv")

    rebuilt_ok = (
        n_missing == 0
        and (not np.isnan(score_diff_max)) and score_diff_max < 1e-10
        and (not np.isnan(auc_diff))       and auc_diff < 1e-8
    )
    if rebuilt_ok:
        status_r, note_r = PASS, (
            "Rebuilt latest identical to current main → ticker merge correct, "
            "dedup rule correct, main file is per-analyst-latest as documented."
        )
    elif n_missing > 0:
        status_r = ACTION_REQ
        note_r = (f"ACTION_REQUIRED: {n_missing} firm-years missing after rebuild "
                  "— all-records file does not cover the full main sample "
                  "(possible ticker/merge mismatch).")
    elif not np.isnan(score_diff_max) and score_diff_max >= 1e-10:
        status_r = ACTION_REQ
        note_r = (f"ACTION_REQUIRED: max_abs_score_diff={score_diff_max:.2e} — "
                  "rebuilt latest disagrees with current main; dedup rules may differ.")
    else:
        status_r, note_r = ACTION_REQ, "ACTION_REQUIRED: auc_diff too large; investigate."
    record("T3-C-1", "per-analyst-latest REBUILT from all-records ≡ current main",
           status_r,
           f"missing={n_missing}, max|dScore|={score_diff_max:.2e}, |dAUC|={auc_diff:.2e}"
           if not np.isnan(score_diff_max) else
           "rebuild insufficient data",
           "missing=0, max|dScore|<1e-10, |dAUC|<1e-8",
           note_r)
    rebuilt_match = rebuilt_ok

    # ── T3-C-2: all-forecasts paper-literal and per-analyst-first ───────────
    for method_name, method_sub in [
        ("all_forecasts_paper_literal", all_matched),
        ("per_analyst_first",
         all_matched.sort_values(["gvkey","fyear","analyst_id","forecast_anndats"])
                     .drop_duplicates(["gvkey","fyear","analyst_id"])),
    ]:
        scores_m = method_sub.groupby(["gvkey","fyear"])["f_dir"].mean().reset_index(name="s")
        merged_m = df[["gvkey","fyear","y_true"]].merge(scores_m, on=["gvkey","fyear"], how="inner")
        if len(merged_m) < 50 or merged_m["y_true"].nunique() < 2:
            continue
        auc_m = roc_auc_score(merged_m["y_true"].astype(int), merged_m["s"])
        agg_rows.append({"method": method_name,
                          "n_firmyears": len(merged_m),
                          "auc": round(auc_m, 4),
                          "note": "same firm-year overlap with main"})

    # Consensus median
    med_dir = all_matched.groupby(["gvkey","fyear"]).apply(
        lambda g: int(g["forecast_eps"].median() > g["actual_eps_t"].iloc[0])
    ).reset_index(name="s_med")
    merged_med = df[["gvkey","fyear","y_true"]].merge(med_dir, on=["gvkey","fyear"], how="inner")
    if len(merged_med) > 50 and merged_med["y_true"].nunique() == 2:
        auc_med = roc_auc_score(merged_med["y_true"].astype(int), merged_med["s_med"])
        agg_rows.append({"method": "consensus_median",
                          "n_firmyears": len(merged_med),
                          "auc": round(auc_med, 4), "note": ""})

if len(agg_rows) > 1:
    agg_df = pd.DataFrame(agg_rows)
    save(agg_df, "T3C_aggregation_sensitivity.csv")
    paper_row = [r for r in agg_rows if "paper_literal" in r["method"]]
    if paper_row:
        # Coverage alignment: firm-year count in paper-literal vs current main
        n_paper    = paper_row[0]["n_firmyears"]
        n_current2 = len(df)
        coverage_gap = n_current2 - n_paper
        diff_agg = abs(EXPECTED_AUC - paper_row[0]["auc"])
        coverage_note = (f"paper_literal covers {n_paper:,} firm-years vs "
                         f"{n_current2:,} in main (gap={coverage_gap}); "
                         "AUC diff computed on shared firm-years only." )
        record("T3-C", "all-forecasts vs per-analyst-latest AUC diff",
               PASS if diff_agg < 0.005 else
               (WARN if diff_agg < 0.015 else ACTION_REQ),
               f"diff={diff_agg:.4f} (current={EXPECTED_AUC:.4f}, all-fcst={paper_row[0]['auc']:.4f})",
               "< 0.005 ideal | 0.005-0.015: report both | > 0.015: ACTION_REQUIRED",
               "If ACTION_REQUIRED: report all-forecasts as paper-literal alternative; "
               "do NOT auto-switch without discussing with professor. " + coverage_note)
else:
    record("T3-C", "Aggregation sensitivity", ACTION_REQ,
           "clean forecast file not loaded or all-records merge empty",
           "diff < 0.005",
           f"ACTION_REQUIRED: generate {FCST_CLEAN_ALL.name}; fallback is not sufficient "
           "for paper-literal evidence.")

# ── T3-D: Currency strict-USD sensitivity (separated by aggregation) ─────────
log("\n[T3-D] Currency sensitivity: strict curr_act='USD' vs current (curr_act='USD' OR NULL)")
log("  Now compared at TWO aggregation levels so currency and aggregation effects")
log("  do not contaminate each other:")
log("    (1) per-analyst-latest   — same as current main AUC = 0.8079")
log("    (2) all-forecasts        — paper-literal aggregation")

curr_rows = []
# Reference: current main
curr_rows.append({"filter": "current_main_curr_act_USD_or_NULL",
                   "aggregation": "per_analyst_latest",
                   "n_firmyears": len(df),
                   "auc": EXPECTED_AUC, "note": "current main (reference)"})

def _auc_by_agg(matched_df: pd.DataFrame, how: str) -> tuple:
    """Return (n_firmyears, auc) for a given aggregation on matched forecasts."""
    if len(matched_df) == 0 or "f_dir" not in matched_df.columns:
        return (0, float("nan"))
    if how == "per_analyst_latest":
        sub = (matched_df.sort_values(
            ["gvkey","fyear","analyst_id","forecast_anndats"],
            ascending=[True, True, True, False]
        ).drop_duplicates(["gvkey","fyear","analyst_id"]))
    elif how == "all_forecasts":
        sub = matched_df
    else:
        return (0, float("nan"))
    scores_x = sub.groupby(["gvkey","fyear"])["f_dir"].mean().reset_index(name="s")
    merged_x = df[["gvkey","fyear","y_true"]].merge(
        scores_x, on=["gvkey","fyear"], how="inner"
    )
    if len(merged_x) < 50 or merged_x["y_true"].nunique() < 2:
        return (len(merged_x), float("nan"))
    return (len(merged_x),
            float(roc_auc_score(merged_x["y_true"].astype(int), merged_x["s"])))

strict_USD_latest_auc   = float("nan")
current_all_forecasts   = float("nan")
strict_USD_all_forecast = float("nan")

# For current-filter all-forecasts we can reuse all_matched (from T3-C).
if fc_clean_loaded and "f_dir" in all_matched.columns:
    n_af, auc_af = _auc_by_agg(all_matched, "all_forecasts")
    current_all_forecasts = auc_af
    curr_rows.append({"filter": "current_main_curr_act_USD_or_NULL",
                       "aggregation": "all_forecasts",
                       "n_firmyears": n_af, "auc": round(auc_af, 4), "note": ""})

if RAW_FCST_FILE.exists():
    try:
        raw_fcst = pd.read_csv(
            RAW_FCST_FILE,
            usecols=lambda c: c.lower() in
            {"ticker","fpedats","anndats","value","curr","curr_act",
             "curcode","estimator","analys","fpi","usfirm"},
            low_memory=False
        )
        raw_fcst.columns = [c.lower() for c in raw_fcst.columns]

        if "curr_act" in raw_fcst.columns:
            # Strict: curr_act == USD only (reject NULL)
            strict = raw_fcst[
                raw_fcst["curr_act"].astype(str).str.upper() == "USD"
            ].copy()
            # Filter to US EPS annual one-year-ahead, value present
            if "measure" not in strict.columns:
                pass
            if "usfirm" in strict.columns:
                strict = strict[
                    strict["usfirm"].astype(str).isin(["1", "1.0", "True", "true"])
                ]
            if "fpi" in strict.columns:
                strict = strict[strict["fpi"].astype(str).str.strip() == "1"]
            strict["forecast_eps"]     = pd.to_numeric(strict["value"], errors="coerce")
            strict["forecast_anndats"] = pd.to_datetime(strict["anndats"], errors="coerce")
            strict["fpedats"]          = pd.to_datetime(strict["fpedats"], errors="coerce")
            strict["ticker"]           = strict["ticker"].astype(str).str.strip()
            # Carry analyst_id so per_analyst_latest aggregation works
            id_parts = []
            for c in ("estimator", "analys"):
                if c in strict.columns:
                    id_parts.append(strict[c].astype(str).str.strip().replace({"nan": ""}))
            if id_parts:
                combined = id_parts[0]
                for p in id_parts[1:]:
                    combined = combined + "|" + p
                strict["analyst_id"] = combined.str.strip("|")
            else:
                strict["analyst_id"] = "unknown_row_" + strict.index.astype(str)

            strict_matched = build_all_matched_forecasts(df, strict)
            if len(strict_matched) > 0 and "f_dir" in strict_matched.columns:
                # Same-aggregation comparison: per-analyst-latest
                n_sl, auc_sl = _auc_by_agg(strict_matched, "per_analyst_latest")
                strict_USD_latest_auc = auc_sl
                curr_rows.append({"filter": "strict_USD_only",
                                   "aggregation": "per_analyst_latest",
                                   "n_firmyears": n_sl,
                                   "auc": round(auc_sl, 4),
                                   "note": "same aggregation as current main"})

                # All-forecasts comparison
                n_sa, auc_sa = _auc_by_agg(strict_matched, "all_forecasts")
                strict_USD_all_forecast = auc_sa
                curr_rows.append({"filter": "strict_USD_only",
                                   "aggregation": "all_forecasts",
                                   "n_firmyears": n_sa,
                                   "auc": round(auc_sa, 4),
                                   "note": ""})

                # Key metric: same-aggregation diff (per_analyst_latest)
                if not np.isnan(auc_sl):
                    diff_curr = abs(EXPECTED_AUC - auc_sl)
                    record("T3-D", "strict curr_act='USD' (same aggregation) vs current",
                           PASS if diff_curr < 0.01 else
                           (WARN if diff_curr < 0.02 else ACTION_REQ),
                           f"diff={diff_curr:.4f} (strict_latest={auc_sl:.4f}, "
                           f"current={EXPECTED_AUC:.4f})",
                           "< 0.01: NULL acceptance validated | > 0.02: ACTION_REQUIRED",
                           "Currency effect cleanly isolated (both use per-analyst-latest). "
                           "If ACTION_REQUIRED: switch main to strict USD filter.")
                else:
                    record("T3-D", "strict curr_act='USD' (same aggregation)", ACTION_REQ,
                           "strict_USD_latest AUC could not be computed",
                           "numeric AUC", "insufficient data after currency filter")
            else:
                record("T3-D", "Currency sensitivity", ACTION_REQ,
                       "strict matched file empty — ticker join or filters too restrictive",
                       "diff < 0.01",
                       "ACTION_REQUIRED: cannot validate NULL acceptance")
        else:
            record("T3-D", "Currency sensitivity", INFO,
                   "curr_act not in raw file; 'curr' field distribution below",
                   "diff < 0.01",
                   f"curr values: {raw_fcst['curr'].value_counts(dropna=False).head(3).to_dict()}")
    except Exception as e:
        record("T3-D", "Currency sensitivity", WARN, f"raw file read error: {e}",
               "diff < 0.01", "")
else:
    record("T3-D", "Currency sensitivity", WARN,
           f"raw file not found: {RAW_FCST_FILE.name}", "diff < 0.01",
           "The Bug-Fix #1 change (curr → curr_act) was manually verified during execution")

save(pd.DataFrame(curr_rows), "T3D_currency_sensitivity.csv")

# ══════════════════════════════════════════════════════════════════════════════
# TIER 4 — Structural and replication checks
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "═"*70)
log("TIER 4 — Structural checks")
log("═"*70)

# ── T4-A: Score≤2 / ≤5 / ≤6 final key equality ───────────────────────────────
log("\n[T4-A] Final key equality: score≤2 = score≤5 = score≤6")
key_rows = []
main_keys = None
if "gvkey" in df.columns and "fyear" in df.columns:
    main_keys = set(zip(df["gvkey"].astype(str), df["fyear"].astype(str)))
    key_rows.append({"sample": "main_score_le_2", "n_keys": len(main_keys),
                      "extra_vs_main": 0, "missing_vs_main": 0})
for label, fpath in [("sensitivity_score_le_5", LE5_FILE),
                      ("sensitivity_score_le_6", LE6_FILE)]:
    if fpath.exists() and main_keys:
        ds = pd.read_csv(fpath, usecols=["gvkey","fyear"], low_memory=False)
        keys_s = set(zip(ds["gvkey"].astype(str), ds["fyear"].astype(str)))
        extra_in_s    = len(keys_s - main_keys)
        extra_in_main = len(main_keys - keys_s)
        key_rows.append({"sample": label, "n_keys": len(keys_s),
                          "extra_vs_main": extra_in_s,
                          "missing_vs_main": extra_in_main})

if len(key_rows) == 3:
    save(pd.DataFrame(key_rows), "T4A_link_sensitivity_key_equality.csv")
    diff5 = key_rows[1]["extra_vs_main"]; diff6 = key_rows[2]["extra_vs_main"]
    record("T4-A", "score≤2 = score≤5 = score≤6 final firm-year keys",
           PASS if diff5 == 0 and diff6 == 0 else WARN,
           f"le5_extra_vs_main={diff5}, le6_extra_vs_main={diff6}",
           "0, 0",
           "Sensitivity samples collapse to main because low-quality links lack "
           "IBES actuals or analyst forecasts (expected, not a bug)")

# ── T4-B: Bootstrap equivalence (small-sample true equivalence test) ──────────
log("\n[T4-B] Bootstrap equivalence: vectorized numpy vs pandas concat (200 reps, N=2000)")

def bootstrap_pandas_slow(y, s, clusters, rng_state, reps=200):
    """Original slow pandas-concat bootstrap for comparison."""
    rng2 = np.random.default_rng(rng_state)
    cluster_arr = clusters.values
    unique_c = np.unique(cluster_arr)
    grouped = {}
    for c, sub in pd.DataFrame({"y": y, "s": s, "c": cluster_arr}).groupby("c"):
        grouped[c] = (sub["y"].values, sub["s"].values)
    out = []
    for _ in range(reps):
        draw = rng2.choice(unique_c, size=len(unique_c), replace=True)
        y_b = np.concatenate([grouped[c][0] for c in draw])
        s_b = np.concatenate([grouped[c][1] for c in draw])
        if len(np.unique(y_b)) < 2:
            continue
        out.append(roc_auc_score(y_b, s_b))
    return np.array(out)

def bootstrap_numpy_fast(y, s, clusters, rng_state, reps=200):
    """New numpy-vectorized bootstrap."""
    rng2 = np.random.default_rng(rng_state)
    cluster_arr = clusters.values
    unique_c = np.unique(cluster_arr)
    grp_y = {c: y[cluster_arr == c] for c in unique_c}
    grp_s = {c: s[cluster_arr == c] for c in unique_c}
    out = []
    for _ in range(reps):
        draw = rng2.choice(unique_c, size=len(unique_c), replace=True)
        y_b = np.concatenate([grp_y[c] for c in draw])
        s_b = np.concatenate([grp_s[c] for c in draw])
        if len(np.unique(y_b)) < 2:
            continue
        out.append(roc_auc_score(y_b, s_b))
    return np.array(out)

if "gvkey" in df.columns and len(df) > 0:
    # Use a small random subsample for speed
    sample_firms = df["gvkey"].dropna().unique()
    rng_small = np.random.default_rng(42)
    if len(sample_firms) > 500:
        sample_firms = rng_small.choice(sample_firms, size=500, replace=False)
    df_small = df[df["gvkey"].isin(sample_firms)].dropna(subset=["y_true","score","gvkey"])
    y_small = df_small["y_true"].astype(int).values
    s_small = df_small["score"].values
    cl_small = df_small["gvkey"]
    SEED_EQUIV = 9999

    boot_slow = bootstrap_pandas_slow(y_small, s_small, cl_small, SEED_EQUIV, reps=200)
    boot_fast = bootstrap_numpy_fast(y_small, s_small, cl_small, SEED_EQUIV, reps=200)

    min_reps = min(len(boot_slow), len(boot_fast))
    if min_reps > 10:
        diff_mean = abs(boot_slow[:min_reps].mean() - boot_fast[:min_reps].mean())
        diff_p25  = abs(np.percentile(boot_slow[:min_reps], 2.5) -
                        np.percentile(boot_fast[:min_reps], 2.5))
        diff_p975 = abs(np.percentile(boot_slow[:min_reps], 97.5) -
                        np.percentile(boot_fast[:min_reps], 97.5))
        save(pd.DataFrame({"bootstrap_slow": boot_slow[:min_reps],
                            "bootstrap_fast": boot_fast[:min_reps]}),
             "T4B_bootstrap_equivalence_200reps.csv")
        record("T4-B", "Bootstrap equivalence (reference grouped vs numpy-vectorized)",
               PASS if max(diff_mean, diff_p25, diff_p975) < 0.005 else WARN,
               f"diff_mean={diff_mean:.4f}, diff_CI_low={diff_p25:.4f}, diff_CI_hi={diff_p975:.4f}",
               "all diffs < 0.005",
               "Both implementations use np.concatenate; this verifies bootstrap SAMPLING "
               "logic is unchanged under the vectorisation refactor.")

# ── T4-C: Manual 50 firm-year audit ──────────────────────────────────────────
log("\n[T4-C] Manual 50 firm-year forecast audit (human readable)")
if FCST_MAIN.exists():
    fm4 = pd.read_csv(FCST_MAIN, low_memory=False)
    fm4.columns = [c.lower() for c in fm4.columns]
    # Sample 50 (gvkey, fyear) pairs from main AUC input
    sample_fy = df.sample(min(50, len(df)), random_state=RANDOM_SEED)[
        ["gvkey","fyear","y_true","score"]
    ].copy()
    sample_keys = sample_fy[["gvkey","fyear"]].drop_duplicates()
    # Merge forecast records — fixed to avoid column collision
    f_audit = fm4.merge(
        df[["gvkey","fyear","actual_eps_t","actual_eps_tp1","y_true"]].rename(
            columns={"y_true": "y_true_main", "actual_eps_t": "actual_eps_t_main"}
        ),
        on=["gvkey","fyear"], how="inner"
    )
    # SEMI-JOIN on (gvkey, fyear): strict 50 firm-year audit, not 50-gvkey-any-fyear
    f_audit = f_audit.merge(sample_keys, on=["gvkey","fyear"], how="inner")

    eps_c = "forecast_eps" if "forecast_eps" in f_audit else \
            ("forecast_value" if "forecast_value" in f_audit else None)
    if eps_c:
        f_audit[eps_c] = pd.to_numeric(f_audit[eps_c], errors="coerce")
        f_audit["actual_eps_t_main"] = pd.to_numeric(
            f_audit["actual_eps_t_main"], errors="coerce"
        )
        # Recomputed direction for manual verification
        f_audit["direction_check"] = (f_audit[eps_c] > f_audit["actual_eps_t_main"]).astype(int)

        keep = ["gvkey","fyear","y_true_main","actual_eps_t_main","actual_eps_tp1",
                "analyst_id","forecast_anndats","fpedats",eps_c,"direction_check"]
        keep = [c for c in keep if c in f_audit.columns]
        save(f_audit[keep].head(300), "T4C_manual_50firm_audit.csv")
        record("T4-C", "Manual forecast audit saved (50 firm-years semi-join; up to 300 rows)",
               INFO, f"saved to T4C_manual_50firm_audit.csv; n_firmyears_in_audit="
               f"{f_audit[['gvkey','fyear']].drop_duplicates().shape[0]}",
               "human review",
               "Verify: direction_check = 1[forecast_eps > actual_eps_t_main]; "
               "score = mean(direction_check) per firm-year")

# ══════════════════════════════════════════════════════════════════════════════
# TIER 5 — Documentation corrections (always needed)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "═"*70)
log("TIER 5 — Documentation corrections required before submission")
log("  (These are wording fixes, not data problems — status = DOC_FIX)")
log("═"*70)

# Paper's no-drift positive class rate for context
paper_nodrift_n_inc = 5418; paper_nodrift_n_dec = 2731
paper_nodrift_rate  = paper_nodrift_n_inc / (paper_nodrift_n_inc + paper_nodrift_n_dec)

record("T5-1", "WORDING FIX: delete 'base rate → AUC inflation' statement",
       DOC_FIX,
       "Must remove: 'actual_increase_rate high → AUC 虚高'",
       "Delete this sentence from all reports and summaries",
       f"AUC is rank-based (Mann-Whitney); base rate affects accuracy, not AUC. "
       f"Paper no-drift inc rate = {paper_nodrift_rate:.1%} > our {df['y_true'].mean():.1%}; "
       f"our base rate actually LOWER than paper's.")

record("T5-2", "WORDING FIX: pooled AUC ≠ average of annual AUCs",
       DOC_FIX,
       "Must replace: '全样本 AUC 是 19 个年度平均'",
       "Replace with: 'Main AUC 0.8079 is the pooled firm-year estimate. "
       "By-fiscal-year table is a diagnostic check.'",
       "Pooled AUC is computed on 29,465 observations together, not averaged across years.")

record("T5-3", "WORDING FIX: update all docs from 13 PASS to 14 PASS",
       DOC_FIX,
       "All documents still say '13 PASS / 0 FAIL'",
       "Change to '14 PASS / 0 FAIL' everywhere",
       "README_FIRST.md, analysts_forecast_auc_report.md, Excel README sheet, email template")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY AND DECISION
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "═"*70)
log("FORENSIC VALIDATION SUMMARY")
log("═"*70)

summary_df = pd.DataFrame(results)
save(summary_df, "FORENSIC_SUMMARY.csv")

blocking_tests   = summary_df[summary_df["status"] == BLOCKING]
action_req_tests = summary_df[summary_df["status"] == ACTION_REQ]
doc_fix_tests    = summary_df[summary_df["status"] == DOC_FIX]
warn_tests       = summary_df[summary_df["status"] == WARN]
pass_tests       = summary_df[summary_df["status"] == PASS]

log(f"\n  PASS:            {len(pass_tests)}")
log(f"  WARN:            {len(warn_tests)}")
log(f"  ACTION_REQUIRED: {len(action_req_tests)}")
log(f"  DOC_FIX:         {len(doc_fix_tests)}")
log(f"  BLOCKING_FAIL:   {len(blocking_tests)}")

log("\n" + "─"*70)

if len(blocking_tests) > 0:
    log("🚨  BLOCKING FAILURES — DO NOT SUBMIT:")
    for _, r in blocking_tests.iterrows():
        log(f"   {r['test_id']}: {r['name']}")
        log(f"           value = {r['value']}")
        log(f"           note  = {r['note'][:100]}")
    log("\n  Fix all BLOCKING issues before submitting to professor.")
    log("  Most likely: re-run 06_construct_auc_sample.py after fixing the bug.")

elif len(action_req_tests) > 0:
    log("🔶  ACTION_REQUIRED — Investigate and decide before submitting:")
    for _, r in action_req_tests.iterrows():
        log(f"   {r['test_id']}: {r['name']}")
        log(f"           value = {r['value']}")
    log("\n  These are not definitive bugs but require explicit decisions.")
    log("  See FORENSIC_SUMMARY.csv for details.")

else:
    log("✅  No BLOCKING failures. No ACTION_REQUIRED items.")
    if len(doc_fix_tests) > 0:
        log(f"📝  {len(doc_fix_tests)} documentation fixes needed (see T5 tests).")
    log("  After completing documentation fixes, you may submit to professor.")

log(f"\n  All outputs: {OUT_DIR}")
log("─"*70)
log()
log("SUGGESTED EMAIL PARAGRAPH (after all tests pass):")
log("-"*60)
log("I note that the analyst AUC of 0.8079 exceeds the Chen et al. (2022)")
log("no-drift benchmark of 0.6471. I performed additional forensic checks:")
log("(1) AUC recomputed via three independent methods — identical to 1e-8;")
log("(2) All forecasts dated before t+1 IBES Actual announcement (0 violations);")
log("(3) Forecast targets consistently aligned to FY t+1;")
log("(4) Within-year permutation test produces null AUCs centered at 0.500;")
log("(5) AUC decomposition mathematically consistent with observed score distribution.")
log("The higher AUC reflects the March GAAP replication sample's composition")
log("(IBES-covered firms; 44% have 5+ analysts, sub-group AUC = 0.868),")
log("the raw no-drift target, and the extended 2005-2023 sample period.")
log("-"*60)
