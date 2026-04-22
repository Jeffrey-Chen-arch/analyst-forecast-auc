"""
06_construct_auc_sample.py — Build the final AUC input samples.

Changes from v3:
  - Must Fix 1: Drop Compustat earnings columns (ni, ib, epspi, etc.) from the
    final AUC deliverables.  The GAAP sample carries these columns because they
    were used in the March sample selection filters, but they must not appear in
    the AUC output files.  Professor Dou's instruction is "do not use any earnings
    numbers from Compustat"; carrying ni in the final file even without using it
    makes the audit trail look unclean and would cause QA Check 11 to FAIL.
  - Must Fix 2: Enforce that the IBES–CRSP link table actually contains a valid
    'score' column.  Without score we cannot enforce the ≤ 2 / ≤ 5 / ≤ 6
    thresholds; proceeding silently would produce a sample that cannot honestly
    be labelled "main_score_le_2".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import *
from src.common import (
    setup_logging, ensure_dirs, read_csv_any, write_csv_gz, write_csv,
    merge_actual_eps_with_tolerance,
)


# ── Compustat earnings column removal ─────────────────────────────────────────
# These fields appear in gaap_sample.csv.gz because the March Table 1 sample
# construction used them as filter criteria.  They must not be present in the
# final AUC input files (professor's explicit requirement: no Compustat earnings).

COMPUSTAT_EARNINGS_COLS = {
    "ni", "ni_lead", "ni_lag",
    "ib", "ibc", "oiadp", "oibdp",
    "epspi", "epspx", "epsfi", "epsfx",
}


def drop_compustat_earnings_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any Compustat earnings columns that may have been carried along
    from the upstream GAAP sample.  Safe to call even if none are present.
    """
    drop_cols = [c for c in df.columns if c.lower() in COMPUSTAT_EARNINGS_COLS]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


# ── IBES–CRSP link: Route A only (scored, no ibtic) ──────────────────────────

def prepare_ibes_link(score_max: int) -> pd.DataFrame:
    link = read_csv_any(
        RAW_LINK_DIR / "ibes_crsp_link_raw.csv.gz",
        force_str_cols=["ticker", "ncusip", "permno"],
    )
    link.columns   = [c.lower() for c in link.columns]
    link["ticker"] = link["ticker"].astype(str).str.strip()
    link["permno"] = pd.to_numeric(link["permno"], errors="coerce")
    link["sdate"]  = pd.to_datetime(link.get("sdate", pd.NaT), errors="coerce") \
                      if "sdate" in link.columns else pd.Timestamp("1900-01-01")
    link["edate"]  = pd.to_datetime(link.get("edate", pd.NaT), errors="coerce").fillna(
                          pd.Timestamp("2099-12-31")) if "edate" in link.columns \
                     else pd.Timestamp("2099-12-31")

    # MUST FIX 2: Enforce that 'score' column exists and has non-missing values.
    # Without score we cannot label the sample "score ≤ X" and must stop.
    if "score" not in link.columns:
        raise RuntimeError(
            "IBES–CRSP link table is missing the 'score' (ICLINK quality) column.  "
            "The main sample requires WRDS ICLINK score ≤ 2.  "
            "Check that ibcrsphist was pulled correctly in step 03."
        )

    link["ibes_crsp_score"] = pd.to_numeric(link["score"], errors="coerce")

    if link["ibes_crsp_score"].notna().sum() == 0:
        raise RuntimeError(
            "IBES–CRSP link 'score' column exists but all values are missing.  "
            "Cannot apply score ≤ threshold to define the main sample."
        )

    link = link[link["ibes_crsp_score"].le(score_max)]
    return link[link["ticker"].notna() & link["permno"].notna()].copy()


def attach_ibes_ticker_route_a(sample: pd.DataFrame, score_max: int) -> pd.DataFrame:
    """
    Route A (primary): gvkey → permno (CCM Scheme B) → IBES ticker (ibcrsphist).
    Link is validated at forecast_window_start.
    'ibes_crsp_score' holds the WRDS ICLINK quality score (0 = best, 6 = worst).
    """
    link = prepare_ibes_link(score_max)
    s = sample.copy()
    s["permno"]               = pd.to_numeric(s["permno"], errors="coerce")
    s["forecast_window_start"] = pd.to_datetime(s["forecast_window_start"], errors="coerce")

    m     = s.merge(link, on="permno", how="left", suffixes=("", "_ibes"))
    valid = m[
        (m["sdate"].isna() | (m["sdate"] <= m["forecast_window_start"])) &
        (m["edate"].isna() | (m["edate"] >= m["forecast_window_start"]))
    ].copy()

    valid = valid.sort_values(
        ["gvkey", "fyear", "ibes_crsp_score", "sdate"],
        ascending=[True, True, True, False]
    )
    n_tix = (valid.groupby(["gvkey", "fyear"])["ticker"]
              .nunique().reset_index(name="n_valid_ibes_tickers"))
    pick  = valid.drop_duplicates(["gvkey", "fyear"], keep="first").copy()
    pick  = pick.merge(n_tix, on=["gvkey", "fyear"], how="left")
    pick["ibes_link_ambiguity_flag"] = pick["n_valid_ibes_tickers"] > 1
    pick["ibes_link_source"]         = "route_a_wrds_ibcrsphist"

    cols = [c for c in [
        "gvkey", "fyear", "ticker", "ncusip", "sdate", "edate",
        "ibes_crsp_score",
        "n_valid_ibes_tickers", "ibes_link_ambiguity_flag", "ibes_link_source",
    ] if c in pick.columns]
    out = s.merge(pick[cols], on=["gvkey", "fyear"], how="left")
    out["has_route_a_ibes_crsp_link"] = out["ticker"].notna()
    out["ibes_crsp_score_max_rule"]   = score_max
    return out


def attach_ibtic_appendix(sample: pd.DataFrame) -> pd.DataFrame:
    """Route B (appendix only): fills ticker for rows with no Route A match."""
    ibtic_path = INT_IBES_DIR / "comp_funda_ibtic.csv.gz"
    out = sample.copy()
    out["has_route_b_ibtic_fallback"] = False
    if not ibtic_path.exists():
        return out

    ibtic = read_csv_any(ibtic_path, force_str_cols=["gvkey", "ibtic"])
    ibtic.columns = [c.lower() for c in ibtic.columns]
    if "ibtic" not in ibtic.columns:
        return out

    ibtic = ibtic[["gvkey", "ibtic"]].rename(columns={"ibtic": "ticker_b"})
    out   = out.merge(ibtic, on="gvkey", how="left")
    no_route_a = out["ticker"].isna() & out["ticker_b"].notna()
    out.loc[no_route_a, "ticker"]                     = out.loc[no_route_a, "ticker_b"]
    out.loc[no_route_a, "has_route_b_ibtic_fallback"] = True
    out.loc[no_route_a, "ibes_link_source"]           = "route_b_ibtic_appendix"
    return out.drop(columns=["ticker_b"], errors="ignore")


# ── EPS matching with date tolerance ──────────────────────────────────────────

def attach_actual_eps(
    sample: pd.DataFrame,
    actuals: pd.DataFrame,
    target_col: str,
    out_eps_col: str,
    out_ann_col: str,
    require_known_by_col: str | None = None,
    strict_anndats: bool = True,
) -> pd.DataFrame:
    base = sample[["gvkey", "fyear", "ticker", target_col]].copy()
    if require_known_by_col and require_known_by_col in sample.columns:
        base[require_known_by_col] = sample[require_known_by_col].values

    result = merge_actual_eps_with_tolerance(
        base=base,
        actuals=actuals[["ticker", "fpedats", "actual_eps", "actual_anndats"]],
        target_date_col=target_col,
        tolerance_days=PENDS_MATCH_TOLERANCE_DAYS,
        require_known_by_col=require_known_by_col,
        strict_anndats=strict_anndats,
    )
    result = result.rename(columns={
        "actual_eps_out":     out_eps_col,
        "actual_anndats_out": out_ann_col,
    })
    return sample.merge(
        result[["gvkey", "fyear", out_eps_col, out_ann_col]],
        on=["gvkey", "fyear"], how="left"
    )


# ── Cumulative attrition ──────────────────────────────────────────────────────

def build_attrition_table(
    final: pd.DataFrame, steps: list[tuple], label: str
) -> pd.DataFrame:
    n_base = len(final)
    mask   = pd.Series(True, index=final.index)
    rows   = []
    for step_name, condition in steps:
        prev_n = int(mask.sum())
        if condition is not None:
            mask = mask & condition.reindex(final.index).fillna(False)
        sub   = final[mask]
        n_now = len(sub)
        rows.append({
            "sample_label":               label,
            "step":                       step_name,
            "n_obs":                      n_now,
            "n_firms":                    int(sub["gvkey"].nunique()),
            "min_fyear":                  int(sub["fyear"].min()) if n_now else None,
            "max_fyear":                  int(sub["fyear"].max()) if n_now else None,
            "dropped_from_previous":      prev_n - n_now,
            "pct_retained_from_previous": round(n_now / prev_n, 4) if prev_n else None,
            "pct_retained_from_base":     round(n_now / n_base, 4) if n_base else None,
        })
    return pd.DataFrame(rows)


# ── Build one AUC sample ──────────────────────────────────────────────────────

def build_one_auc_sample(
    score_max: int,
    label: str,
    use_ibtic_appendix: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    sample = read_csv_any(
        INT_LINK_DIR / "sample_with_crsp_permno_schemeB.csv.gz",
        force_str_cols=["gvkey", "permno"],
    )
    for c in ["datadate", "formation_date", "forecast_window_start",
              "forecast_window_end", "target_fpedats_t", "target_fpedats_tp1"]:
        sample[c] = pd.to_datetime(sample[c], errors="coerce")
    sample["gvkey"]  = sample["gvkey"].astype(str).str.zfill(6)
    sample["row_id"] = np.arange(len(sample))

    # Route A: WRDS IBES–CRSP scored link (mandatory for main + sensitivity)
    linked = attach_ibes_ticker_route_a(sample, score_max)

    # Route B: ibtic fallback (appendix only)
    if use_ibtic_appendix:
        linked = attach_ibtic_appendix(linked)
        linked["has_ticker_any"] = linked["ticker"].notna()
    else:
        linked["has_ticker_any"] = linked["has_route_a_ibes_crsp_link"]

    # Load actuals
    actuals = read_csv_any(
        INT_IBES_DIR / "actuals_clean_all_rows.csv.gz",
        force_str_cols=["ticker"],
    )
    actuals.columns = [c.lower() for c in actuals.columns]
    actuals["ticker"]         = actuals["ticker"].astype(str).str.strip()
    actuals["fpedats"]        = pd.to_datetime(actuals["fpedats"],        errors="coerce")
    actuals["actual_anndats"] = pd.to_datetime(
        actuals.get("actual_anndats", pd.NaT), errors="coerce"
    ) if "actual_anndats" in actuals.columns else pd.NaT
    actuals["actual_eps"] = pd.to_numeric(actuals["actual_eps"], errors="coerce")
    actuals = actuals[
        actuals["ticker"].notna() & actuals["fpedats"].notna() &
        actuals["actual_eps"].notna()
    ].copy()

    # Attach EPS_t (strict: anndats must be non-missing AND ≤ formation_date)
    a = attach_actual_eps(
        linked, actuals, "target_fpedats_t",
        "actual_eps_t", "actual_anndats_t",
        require_known_by_col="formation_date", strict_anndats=True,
    )
    # Attach EPS_{t+1} (no timing constraint on announcement)
    a = attach_actual_eps(
        a, actuals, "target_fpedats_tp1",
        "actual_eps_tp1", "actual_anndats_tp1",
        require_known_by_col=None, strict_anndats=False,
    )

    a["has_actual_t"]   = a["actual_eps_t"].notna()
    a["has_actual_tp1"] = a["actual_eps_tp1"].notna()
    a["actual_increase_raw"] = np.where(
        a["actual_eps_tp1"].gt(a["actual_eps_t"]), 1,
        np.where(a["actual_eps_tp1"].notna() & a["actual_eps_t"].notna(), 0, np.nan)
    )
    a["actual_tie_flag"] = (
        a["actual_eps_tp1"].eq(a["actual_eps_t"]) &
        a["actual_eps_tp1"].notna() & a["actual_eps_t"].notna()
    )

    # Load forecasts
    f = read_csv_any(
        INT_IBES_DIR / "detail_forecasts_clean_all_rows.csv.gz",
        force_str_cols=["ticker", "analyst_id"],
    )
    f.columns = [c.lower() for c in f.columns]
    f["ticker"]           = f["ticker"].astype(str).str.strip()
    f["fpedats"]          = pd.to_datetime(f["fpedats"],          errors="coerce")
    f["forecast_anndats"] = pd.to_datetime(f["forecast_anndats"], errors="coerce")
    f["forecast_revdats"] = pd.to_datetime(
        f.get("forecast_revdats", pd.NaT), errors="coerce"
    ) if "forecast_revdats" in f.columns else pd.NaT
    f["forecast_eps"] = pd.to_numeric(f["forecast_eps"], errors="coerce")

    base_for_f = a[a["has_ticker_any"] & a["has_actual_t"] & a["has_actual_tp1"]].copy()

    # Merge on ticker; filter by ±PENDS_MATCH_TOLERANCE_DAYS on fpedats AND window
    fm = base_for_f.merge(f, on="ticker", how="left", suffixes=("", "_f"))
    fm["forecast_fpedats_diff_days"] = (
        (fm["fpedats"] - fm["target_fpedats_tp1"]).dt.days.abs()
    )
    fm = fm[
        (fm["forecast_fpedats_diff_days"] <= PENDS_MATCH_TOLERANCE_DAYS) &
        fm["forecast_anndats"].between(fm["forecast_window_start"], fm["forecast_window_end"])
    ].copy()

    # One forecast per analyst per firm-year (latest revision in window)
    fm = fm.sort_values(
        ["row_id", "analyst_id", "forecast_anndats", "forecast_revdats"],
        ascending=[True, True, False, False]
    )
    fm_last = fm.drop_duplicates(["row_id", "analyst_id"], keep="first").copy()
    fm_last["forecast_increase_raw"] = np.where(
        fm_last["forecast_eps"].gt(fm_last["actual_eps_t"]), 1, 0
    )
    fm_last["forecast_tie_flag"] = fm_last["forecast_eps"].eq(fm_last["actual_eps_t"])

    score_agg = (
        fm_last.groupby("row_id").agg(
            n_forecasts         = ("forecast_eps",          "size"),
            n_unique_analysts   = ("analyst_id",            "nunique"),
            analyst_score       = ("forecast_increase_raw", "mean"),
            n_forecast_ties     = ("forecast_tie_flag",     "sum"),
            mean_forecast_eps   = ("forecast_eps",          "mean"),
            median_forecast_eps = ("forecast_eps",          "median"),
        ).reset_index()
    )
    median_dir = fm_last.groupby("row_id").agg(
        median_forecast_eps_agg=("forecast_eps", "median")
    ).reset_index()

    final = a.merge(score_agg, on="row_id", how="left")
    final = final.merge(median_dir, on="row_id", how="left")
    final["has_forecasts_in_window"] = final["n_forecasts"].fillna(0).gt(0)
    final["auc_sample_flag"] = (
        final["has_ticker_any"] &
        final["has_actual_t"]   &
        final["has_actual_tp1"] &
        final["has_forecasts_in_window"]
    )
    final["sample_label"] = label

    final["median_forecast_increase"] = np.where(
        final["median_forecast_eps_agg"].notna() & final["actual_eps_t"].notna(),
        (final["median_forecast_eps_agg"] > final["actual_eps_t"]).astype(float),
        np.nan
    )

    auc_input = final[final["auc_sample_flag"]].copy()
    auc_input["y_true"] = auc_input["actual_increase_raw"].astype(int)
    auc_input["score"]  = auc_input["analyst_score"].astype(float)

    # MUST FIX 1: Remove Compustat earnings columns from the final deliverables.
    # Professor's requirement is "do not use Compustat earnings"; having ni/ib/etc.
    # in the output file — even without using them — makes the audit trail look
    # unclean and causes QA Check 11 to FAIL.
    auc_input = drop_compustat_earnings_columns(auc_input)
    fm_last   = drop_compustat_earnings_columns(fm_last)

    # Attrition table
    has_ticker = (final["has_route_a_ibes_crsp_link"]
                  if not use_ibtic_appendix else final["has_ticker_any"])
    steps = [
        ("01_base_gaap_sample",                         None),
        ("02_has_CCM_SchemeB_permno_at_formation",       final["permno"].notna()),
        (f"03_has_WRDS_IBES_link_score_le_{score_max}", has_ticker),
        ("04_has_IBES_actual_EPS_t_known_by_formation", final["has_actual_t"]),
        ("05_has_IBES_actual_EPS_tp1",                  final["has_actual_tp1"]),
        ("06_has_analyst_forecasts_in_window",           final["has_forecasts_in_window"]),
        ("07_final_AUC_sample",                          final["auc_sample_flag"]),
    ]
    attrition = build_attrition_table(final, steps, label)

    return auc_input, fm_last, attrition


def main():
    logger = setup_logging(LOG_DIR / "06_construct_auc_sample.log")
    ensure_dirs([INT_MERGE_DIR, TABLE_DIR])

    # Main (score ≤ 2, no ibtic)
    main_auc, main_fcst, main_attr = build_one_auc_sample(
        MAIN_MAX_IBES_CRSP_SCORE, "main_score_le_2", use_ibtic_appendix=False
    )
    # Sensitivity 1 (score ≤ 5) and Sensitivity 2 (score ≤ 6)
    s5_auc, _, s5_attr = build_one_auc_sample(
        SENSITIVITY_MAX_IBES_CRSP_SCORE, "sensitivity_score_le_5", use_ibtic_appendix=False
    )
    s6_auc, _, s6_attr = build_one_auc_sample(
        SENSITIVITY_MAX_IBES_CRSP_SCORE2, "sensitivity_score_le_6", use_ibtic_appendix=False
    )

    all_attr = pd.concat([main_attr, s5_attr, s6_attr], ignore_index=True)

    # Appendix: ibtic coverage diagnostic
    if USE_IBTIC_FALLBACK_APPENDIX:
        app_auc, _, app_attr = build_one_auc_sample(
            MAIN_MAX_IBES_CRSP_SCORE, "appendix_score_le_2_plus_ibtic",
            use_ibtic_appendix=True
        )
        write_csv_gz(app_auc, INT_MERGE_DIR / "firm_year_auc_input_appendix_ibtic.csv.gz")
        all_attr = pd.concat([all_attr, app_attr], ignore_index=True)
        logger.info("Appendix (ibtic) sample: %d firm-years, %d firms",
                    len(app_auc), app_auc["gvkey"].nunique())

    write_csv_gz(main_auc,  INT_MERGE_DIR / "firm_year_auc_input_main_score_le_2.csv.gz")
    write_csv_gz(main_fcst, INT_MERGE_DIR / "forecast_level_directions_main_score_le_2.csv.gz")
    write_csv_gz(s5_auc,    INT_MERGE_DIR / "firm_year_auc_input_sensitivity_score_le_5.csv.gz")
    write_csv_gz(s6_auc,    INT_MERGE_DIR / "firm_year_auc_input_sensitivity_score_le_6.csv.gz")
    write_csv(main_attr, TABLE_DIR / "05_attrition_main_score_le_2.csv")
    write_csv(all_attr,  TABLE_DIR / "05_attrition_all_link_rules.csv")

    logger.info("Main AUC sample: %d firm-years, %d firms",
                len(main_auc), main_auc["gvkey"].nunique())

    # Verify no Compustat earnings in output
    comp_remaining = [c for c in main_auc.columns if c.lower() in COMPUSTAT_EARNINGS_COLS]
    if comp_remaining:
        logger.error("Compustat earnings columns still present: %s", comp_remaining)
    else:
        logger.info("Confirmed: no Compustat earnings columns in final AUC input.")

    if "ibes_crsp_score" in main_auc.columns:
        logger.info("IBES–CRSP score range in main sample: min=%.0f, max=%.0f",
                    main_auc["ibes_crsp_score"].min(),
                    main_auc["ibes_crsp_score"].max())


if __name__ == "__main__":
    main()
