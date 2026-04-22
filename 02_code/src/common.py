from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("auc_pipeline")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ── File I/O ──────────────────────────────────────────────────────────────────

def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def read_csv_any(path: Path, force_str_cols: list[str] | None = None, **kwargs) -> pd.DataFrame:
    """Load CSV (plain or .gz). Optionally force string dtype for ID columns
    to preserve leading zeros in tickers, CUSIPs, gvkeys, etc.

    Example:
        df = read_csv_any(path, force_str_cols=["ticker","cusip","gvkey"])
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    if force_str_cols:
        existing_dtype = kwargs.pop("dtype", {})
        dtype = {c: str for c in force_str_cols}
        dtype.update(existing_dtype)
        kwargs["dtype"] = dtype
    return pd.read_csv(path, low_memory=False, **kwargs)


def write_csv_gz(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, compression="gzip")


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


# ── Date helpers ──────────────────────────────────────────────────────────────

def next_calendar_month_window(dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (first_day, last_day) of the calendar month immediately after each date.

    Example: 2020-03-31 → (2020-04-01, 2020-04-30)
    """
    start = (dates + pd.offsets.MonthBegin(1)).dt.normalize()
    end   = start + pd.offsets.MonthEnd(0)
    return start, end


# ── IBES adjustment-basis helper ──────────────────────────────────────────────

def infer_ibes_basis(table_name: str) -> str:
    """Return 'adjusted' or 'unadjusted' from table name convention."""
    name = table_name.lower()
    if "actu" in name or "detu" in name:
        return "unadjusted"
    return "adjusted"


# ── Actual EPS matching with date tolerance ───────────────────────────────────

def merge_actual_eps_with_tolerance(
    base: pd.DataFrame,
    actuals: pd.DataFrame,
    target_date_col: str,
    tolerance_days: int,
    require_known_by_col: str | None = None,
    strict_anndats: bool = True,
) -> pd.DataFrame:
    """
    Match IBES actual EPS to each row in `base` by ticker + date proximity.

    actuals must have: ticker, fpedats, actual_eps, actual_anndats

    Rules:
      1. Same ticker
      2. |actuals.fpedats − base[target_date_col]| ≤ tolerance_days
      3. If require_known_by_col set AND strict_anndats=True:
           actual_anndats MUST be non-missing AND ≤ base[require_known_by_col]
         If strict_anndats=False:
           allow missing actual_anndats (fallback for sensitivity)
      4. Among multiple matches, pick the row with the latest actual_anndats
         (most recent announced value).

    Returns base with added columns: actual_eps_out, actual_anndats_out
    """
    b = base.copy()
    b["_rid"] = np.arange(len(b))
    b[target_date_col] = pd.to_datetime(b[target_date_col], errors="coerce")

    a = actuals[["ticker", "fpedats", "actual_eps", "actual_anndats"]].copy()
    a["fpedats"]        = pd.to_datetime(a["fpedats"],        errors="coerce")
    a["actual_anndats"] = pd.to_datetime(a["actual_anndats"], errors="coerce")
    a["actual_eps"]     = pd.to_numeric(a["actual_eps"],      errors="coerce")
    a = a[a["actual_eps"].notna() & a["fpedats"].notna()].copy()

    # Merge on ticker, then filter by date tolerance
    m = b[["_rid", "ticker", target_date_col]].merge(a, on="ticker", how="left")
    date_diff = (m["fpedats"] - m[target_date_col]).dt.days.abs()
    m = m[date_diff <= tolerance_days].copy()

    # Apply known-by constraint for EPS_t
    if require_known_by_col and require_known_by_col in b.columns:
        kb = b[["_rid", require_known_by_col]].copy()
        m  = m.merge(kb, on="_rid", how="left")
        if strict_anndats:
            # Main: anndats must be non-missing AND ≤ formation_date
            m = m[m["actual_anndats"].notna() &
                  (m["actual_anndats"] <= m[require_known_by_col])].copy()
        else:
            # Sensitivity: allow missing anndats if fpedats ≤ formation_date − 90d
            fpedats_ok = m["fpedats"] <= (m[require_known_by_col] - pd.Timedelta(days=90))
            anndats_ok = m["actual_anndats"].notna() & (m["actual_anndats"] <= m[require_known_by_col])
            m = m[anndats_ok | (m["actual_anndats"].isna() & fpedats_ok)].copy()

    if len(m) == 0:
        b["actual_eps_out"]    = np.nan
        b["actual_anndats_out"] = pd.NaT
        return b.drop(columns=["_rid"])

    # Pick latest actual_anndats per row
    m["_diff"] = (m["fpedats"] - m[target_date_col]).dt.days.abs()
    m = m.sort_values(["_rid", "_diff", "actual_anndats"], ascending=[True, True, False])
    best = m.drop_duplicates("_rid", keep="first")[["_rid", "actual_eps", "actual_anndats"]]

    out = b.merge(best.rename(columns={"actual_eps": "actual_eps_out",
                                        "actual_anndats": "actual_anndats_out"}),
                  on="_rid", how="left")
    return out.drop(columns=["_rid"])


# ── Validation ────────────────────────────────────────────────────────────────

def require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}\nHave: {list(df.columns)}")


def pick_existing_columns(df: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


# ── AUC ───────────────────────────────────────────────────────────────────────

def auc_fast(y_true: Sequence[int], score: Sequence[float]) -> float:
    """AUC via Mann–Whitney rank-sum. Returns NaN if either class absent."""
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[mask], s[mask]
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks        = pd.Series(s).rank(method="average").to_numpy()
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_auc(
    df: pd.DataFrame,
    y_col: str,
    score_col: str,
    cluster_col: str | None = None,
    reps: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Firm-clustered bootstrap AUC — numpy-vectorised, avoids pd.concat loop."""
    rng = np.random.default_rng(seed)

    if cluster_col is None:
        y = df[y_col].to_numpy(dtype=float)
        s = df[score_col].to_numpy(dtype=float)
        n = len(y)
        out = []
        for _ in range(reps):
            idx = rng.integers(0, n, size=n)
            out.append(auc_fast(y[idx], s[idx]))
    else:
        clusters = df[cluster_col].dropna().unique()
        # Pre-extract numpy arrays per cluster — avoids repeated DataFrame slicing
        grp_y = {}
        grp_s = {}
        for g, sub in df.groupby(cluster_col, sort=False):
            grp_y[g] = sub[y_col].to_numpy(dtype=float)
            grp_s[g] = sub[score_col].to_numpy(dtype=float)

        out = []
        for _ in range(reps):
            draw = rng.choice(clusters, size=len(clusters), replace=True)
            y_boot = np.concatenate([grp_y[g] for g in draw])
            s_boot = np.concatenate([grp_s[g] for g in draw])
            out.append(auc_fast(y_boot, s_boot))

    return pd.DataFrame({"bootstrap_auc": out})


def summarize_auc(df: pd.DataFrame, y_col: str, score_col: str) -> dict:
    """Standard AUC summary statistics for one sample."""
    auc      = auc_fast(df[y_col], df[score_col])
    y        = df[y_col]
    score    = df[score_col]
    pred_inc = score > 0.5
    return {
        "n_obs":                               int(len(df)),
        "n_firms":                             int(df["gvkey"].nunique()) if "gvkey" in df.columns else np.nan,
        "n_years":                             int(df["fyear"].nunique()) if "fyear" in df.columns else np.nan,
        "n_actual_increase":                   int((y == 1).sum()),
        "n_actual_decrease_or_equal":          int((y == 0).sum()),
        "actual_increase_rate":                float((y == 1).mean())   if len(df) else np.nan,
        "mean_score":                          float(score.mean())      if len(df) else np.nan,
        "median_score":                        float(score.median())    if len(df) else np.nan,
        "auc":                                 auc,
        "accuracy_cutoff_0p5":                 float((pred_inc.astype(int) == y).mean()) if len(df) else np.nan,
        "predicted_increase_rate_cutoff_0p5":  float(pred_inc.mean())  if len(df) else np.nan,
    }
