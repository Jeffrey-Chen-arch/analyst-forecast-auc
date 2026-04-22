from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from src.common import (
    setup_logging, ensure_dirs, read_csv_any, write_csv,
    auc_fast, bootstrap_auc, summarize_auc,
)

RECENT_WINDOWS = {
    "full_sample":   (None, None),
    "2015_2023":     (2015, 2023),
    "2019_2023":     (2019, 2023),
    "fy2023_only":   (2023, 2023),
}


def roc_points(y_true, score) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[mask], s[mask]
    thresholds = np.r_[np.inf, np.sort(np.unique(s))[::-1], -np.inf]
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    pts = []
    for th in thresholds:
        pred = s >= th
        tpr  = float(((pred) & (y == 1)).sum() / pos) if pos else np.nan
        fpr  = float(((pred) & (y == 0)).sum() / neg) if neg else np.nan
        pts.append({"fpr": fpr, "tpr": tpr, "threshold": th})
    return pd.DataFrame(pts)


def compute_for_file(path, label):
    df = read_csv_any(path, force_str_cols=["gvkey"])
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").astype("Int64")
    df["score"]  = pd.to_numeric(df["score"],  errors="coerce")
    df = df.dropna(subset=["y_true", "score"]).copy()
    df["y_true"] = df["y_true"].astype(int)

    summary = summarize_auc(df, "y_true", "score")
    summary["sample_label"] = label

    boot = bootstrap_auc(
        df, "y_true", "score",
        cluster_col="gvkey" if BOOTSTRAP_CLUSTER_BY_FIRM else None,
        reps=BOOTSTRAP_REPS, seed=RANDOM_SEED,
    )
    summary.update({
        "bootstrap_reps":            BOOTSTRAP_REPS,
        "bootstrap_cluster_by_firm": BOOTSTRAP_CLUSTER_BY_FIRM,
        "auc_boot_mean":             float(boot["bootstrap_auc"].mean()),
        "auc_ci_p2_5":               float(boot["bootstrap_auc"].quantile(0.025)),
        "auc_ci_p97_5":              float(boot["bootstrap_auc"].quantile(0.975)),
        "p_value_auc_le_0p5":        float((boot["bootstrap_auc"] <= 0.5).mean()),
    })

    # By fiscal year
    by_year = []
    for fy, sub in df.groupby("fyear"):
        if len(sub) >= 10 and sub["y_true"].nunique() == 2:
            row = {"sample_label": label, "fyear": int(fy)}
            row.update(summarize_auc(sub, "y_true", "score"))
            by_year.append(row)

    # By analyst coverage buckets
    bins, blabels = [0, 1, 2, 4, 999999], ["1", "2", "3-4", "5+"]
    by_cov = []
    if "n_unique_analysts" in df.columns:
        df["coverage_bucket"] = pd.cut(
            df["n_unique_analysts"].fillna(0), bins=bins, labels=blabels, right=True
        )
        for b, sub in df.groupby("coverage_bucket", observed=False):
            if len(sub) > 0 and sub["y_true"].nunique() == 2:
                row = {"sample_label": label, "analyst_coverage_bucket": str(b)}
                row.update(summarize_auc(sub, "y_true", "score"))
                by_cov.append(row)

    # Actual-tie exclusion sensitivity
    if "actual_tie_flag" in df.columns:
        no_ties = df[~df["actual_tie_flag"].fillna(False)]
    else:
        no_ties = df.copy()
    tie_row = {"sample_label": label + "_exclude_actual_ties"}
    if len(no_ties) > 0 and no_ties["y_true"].nunique() == 2:
        tie_row.update(summarize_auc(no_ties, "y_true", "score"))

    # Consensus median sensitivity
    median_rows = []
    if "median_forecast_increase" in df.columns:
        df_med = df[df["median_forecast_increase"].notna()].copy()
        df_med["score_median"] = df_med["median_forecast_increase"].astype(float)
        if len(df_med) > 0 and df_med["y_true"].nunique() == 2:
            row = {"sample_label": label + "_consensus_median"}
            row.update(summarize_auc(df_med, "y_true", "score_median"))
            median_rows.append(row)

    # Recent-period subsamples
    recent_rows = []
    if "fyear" in df.columns:
        for window_label, (yr_start, yr_end) in RECENT_WINDOWS.items():
            if yr_start is None:
                sub = df.copy()
            else:
                sub = df[df["fyear"].between(yr_start, yr_end)]
            if len(sub) >= 10 and sub["y_true"].nunique() == 2:
                row = {
                    "sample_label": label,
                    "period":       window_label,
                    "fyear_start":  yr_start if yr_start else df["fyear"].min(),
                    "fyear_end":    yr_end   if yr_end   else df["fyear"].max(),
                }
                row.update(summarize_auc(sub, "y_true", "score"))
                recent_rows.append(row)

    roc = roc_points(df["y_true"], df["score"])
    roc["sample_label"] = label

    return (df,
            pd.DataFrame([summary]),
            pd.DataFrame(by_year),
            pd.DataFrame(by_cov),
            pd.DataFrame([tie_row]),
            pd.DataFrame(median_rows),
            pd.DataFrame(recent_rows),
            boot, roc)


def plot_roc(roc, label, auc_val, ci_lo, ci_hi, out_path, title_suffix=""):
    """
    FIX 8: Paper AUC reference is a text annotation, not a horizontal line.
    ROC y-axis = True Positive Rate, NOT AUC value. Drawing paper AUC (0.6471)
    as a horizontal line at y=0.6471 would misrepresent it as a TPR threshold.
    The paper benchmark is shown as a footnote-style text annotation instead.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc["fpr"], roc["tpr"], lw=2, color="#2166ac",
            label=f"Analysts (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], lw=1, ls="--", color="grey",
            label="Random guess (AUC = 0.50)")
    ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.1, color="#2166ac")

    ax.text(
        0.53, 0.07,
        f"Bootstrap 95% CI: [{100*ci_lo:.2f}%, {100*ci_hi:.2f}%]",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85),
    )
    # FIX 8: paper AUC as text annotation (NOT a horizontal line, which would
    # wrongly suggest that 0.6471 is a TPR benchmark on the ROC y-axis)
    ax.text(
        0.05, 0.03,
        f"Paper no-drift analyst AUC reference: "
        f"{100*PAPER_ANALYST_AUC_NO_DRIFT:.2f}%\n"
        f"(Chen et al. 2022, fn. 34; different sample)",
        transform=ax.transAxes, fontsize=7.5, color="#666666",
    )

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate",  fontsize=11)
    ax.set_ylabel("True Positive Rate",   fontsize=11)
    ax.set_title(
        f"ROC Curve: Analysts' Earnings Forecast Direction\n"
        f"(No Drift Adjustment{title_suffix})", fontsize=11
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    logger = setup_logging(LOG_DIR / "07_compute_auc.log")
    ensure_dirs([TABLE_DIR, FIG_DIR])

    files = [
        (INT_MERGE_DIR / "firm_year_auc_input_main_score_le_2.csv.gz",
         "main_score_le_2"),
        (INT_MERGE_DIR / "firm_year_auc_input_sensitivity_score_le_5.csv.gz",
         "sensitivity_score_le_5"),
        (INT_MERGE_DIR / "firm_year_auc_input_sensitivity_score_le_6.csv.gz",
         "sensitivity_score_le_6"),
    ]

    all_summ, all_years, all_cov, all_ties   = [], [], [], []
    all_median, all_recent, all_rocs = [], [], []

    for path, label in files:
        if not path.exists():
            logger.warning("Skipping missing: %s", path)
            continue

        (df, summ, by_year, by_cov, tie_s, median_s,
         recent_s, boot, roc) = compute_for_file(path, label)

        all_summ.append(summ);       all_years.append(by_year)
        all_cov.append(by_cov);      all_ties.append(tie_s)
        all_median.append(median_s); all_recent.append(recent_s)
        all_rocs.append((label, roc, summ))

        write_csv(boot, TABLE_DIR / f"06_bootstrap_dist_{label}.csv")
        write_csv(roc,  TABLE_DIR / f"06_roc_points_{label}.csv")
        logger.info(
            "[%s]  AUC=%.4f  CI=[%.4f, %.4f]  p≤0.5=%.4f  N=%d",
            label,
            float(summ["auc"].iloc[0]),
            float(summ["auc_ci_p2_5"].iloc[0]),
            float(summ["auc_ci_p97_5"].iloc[0]),
            float(summ["p_value_auc_le_0p5"].iloc[0]),
            int(summ["n_obs"].iloc[0]),
        )

    # Consolidated output tables
    write_csv(pd.concat(all_summ,   ignore_index=True),
              TABLE_DIR / "06_auc_main_and_sensitivity_summary.csv")
    write_csv(pd.concat(all_years,  ignore_index=True),
              TABLE_DIR / "06_auc_by_fyear.csv")
    write_csv(pd.concat(all_cov,    ignore_index=True),
              TABLE_DIR / "06_auc_by_analyst_coverage.csv")
    write_csv(pd.concat(all_ties,   ignore_index=True),
              TABLE_DIR / "06_auc_tie_sensitivity.csv")
    if any(len(m) > 0 for m in all_median):
        write_csv(pd.concat(all_median, ignore_index=True),
                  TABLE_DIR / "06_auc_consensus_median_sensitivity.csv")
    if any(len(r) > 0 for r in all_recent):
        write_csv(pd.concat(all_recent, ignore_index=True),
                  TABLE_DIR / "06_auc_recent_periods.csv")

    # ROC curves (one per sample)
    summ_all = pd.concat(all_summ, ignore_index=True)
    for label, roc, summ in all_rocs:
        row = summ_all[summ_all["sample_label"] == label]
        if len(row) == 0:
            continue
        suffix = "" if "main" in label else f" — {label}"
        fname  = f"roc_curve_{label}.png"
        plot_roc(
            roc, label,
            auc_val=float(row["auc"].iloc[0]),
            ci_lo=float(row["auc_ci_p2_5"].iloc[0]),
            ci_hi=float(row["auc_ci_p97_5"].iloc[0]),
            out_path=FIG_DIR / fname,
            title_suffix=suffix,
        )
        logger.info("Saved: %s", fname)

    logger.info("AUC computation complete.")


if __name__ == "__main__":
    main()
