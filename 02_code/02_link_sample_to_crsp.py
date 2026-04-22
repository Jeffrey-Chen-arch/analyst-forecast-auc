from __future__ import annotations

import numpy as np
import pandas as pd

from config import *
from src.common import setup_logging, ensure_dirs, read_csv_any, write_csv_gz, write_csv, require_columns


def normalize_linkenddt(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    # WRDS convention often uses missing linkenddt for still-active links.
    return out.fillna(pd.Timestamp("2099-12-31"))


def main():
    logger = setup_logging(LOG_DIR / "02_link_sample_to_crsp.log")
    ensure_dirs([INT_LINK_DIR, TABLE_DIR])

    sample = read_csv_any(INT_SAMPLE_DIR / "base_gaap_sample_with_timing.csv.gz")
    ccm = read_csv_any(RAW_LINK_DIR / "ccm_links_full_from_march.csv.gz")
    require_columns(sample, ["gvkey", "fyear", "datadate", "formation_date"], "base sample")
    require_columns(ccm, ["gvkey", "permno", "liid", "linktype", "linkprim", "linkdt", "linkenddt"], "CCM links")

    sample["gvkey"] = sample["gvkey"].astype(str).str.zfill(6)
    sample["formation_date"] = pd.to_datetime(sample["formation_date"], errors="coerce")
    sample["datadate"] = pd.to_datetime(sample["datadate"], errors="coerce")

    ccm["gvkey"] = ccm["gvkey"].astype(str).str.zfill(6)
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"], errors="coerce")
    ccm["linkenddt"] = normalize_linkenddt(ccm["linkenddt"])
    ccm["linktype"] = ccm["linktype"].astype(str).str.upper()
    ccm["linkprim"] = ccm["linkprim"].astype(str).str.upper()

    # March Scheme B: LU/LC, P/C, valid at formation date.
    ccm_b = ccm[ccm["linktype"].isin(CCM_LINKTYPES) & ccm["linkprim"].isin(CCM_LINKPRIMS)].copy()

    merged = sample.merge(ccm_b, on="gvkey", how="left", suffixes=("", "_ccm"))
    valid = merged[(merged["linkdt"] <= merged["formation_date"]) & (merged["linkenddt"] >= merged["formation_date"])].copy()

    # Prefer primary over co-primary; if multiple same rank, prefer the link with latest linkdt.
    valid["linkprim_rank"] = np.where(valid["linkprim"].eq("P"), 0, 1)
    valid["linktype_rank"] = valid["linktype"].map({"LU": 0, "LC": 1}).fillna(9)
    valid = valid.sort_values(["gvkey", "fyear", "linkprim_rank", "linktype_rank", "linkdt"], ascending=[True, True, True, True, False])

    # Link ambiguity flag before picking one.
    amb = (valid.groupby(["gvkey", "fyear"])["permno"]
           .nunique(dropna=True).reset_index(name="n_valid_permnos_schemeB"))
    amb["ccm_ambiguity_flag"] = amb["n_valid_permnos_schemeB"] > 1

    picked = valid.drop_duplicates(["gvkey", "fyear"], keep="first").copy()
    picked = picked.merge(amb, on=["gvkey", "fyear"], how="left")

    # Add unmatched sample observations.
    out = sample.merge(
        picked[["gvkey", "fyear", "permno", "permco", "liid", "linktype", "linkprim", "linkdt", "linkenddt", "n_valid_permnos_schemeB", "ccm_ambiguity_flag"]],
        on=["gvkey", "fyear"], how="left"
    )
    out["has_ccm_schemeB_link_at_formation"] = out["permno"].notna()

    write_csv_gz(out, INT_LINK_DIR / "sample_with_crsp_permno_schemeB.csv.gz")

    summ = pd.DataFrame([
        {"step": "base_gaap_sample", "n_obs": len(sample), "n_firms": sample["gvkey"].nunique()},
        {"step": "has_CCM_SchemeB_permno_at_formation", "n_obs": int(out["has_ccm_schemeB_link_at_formation"].sum()),
         "n_firms": out.loc[out["has_ccm_schemeB_link_at_formation"], "gvkey"].nunique()},
        {"step": "ccm_ambiguity_flag", "n_obs": int(out["ccm_ambiguity_flag"].fillna(False).sum()),
         "n_firms": out.loc[out["ccm_ambiguity_flag"].fillna(False), "gvkey"].nunique()},
    ])
    write_csv(summ, TABLE_DIR / "01_ccm_schemeB_link_summary.csv")
    logger.info("Wrote CRSP-linked sample and summary.")


if __name__ == "__main__":
    main()
