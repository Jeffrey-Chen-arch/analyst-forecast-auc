from __future__ import annotations
import subprocess, sys
from pathlib import Path

# FIX 7: 09_quality_checks.py runs BEFORE 08_make_report.py so that the
# markdown report (and Excel workbook) contain the complete QA results.
SCRIPTS = [
    "00_setup_folders.py",
    "01_prepare_march_sample.py",
    "02_link_sample_to_crsp.py",
    "03_wrds_pull_ibes.py",          # ← requires WRDS internet connection
    "04_prepare_ibes_actuals.py",
    "05_prepare_analyst_forecasts.py",
    "06_construct_auc_sample.py",
    "07_compute_auc.py",
    "09_quality_checks.py",          # ← run QA before report generation
    "08_make_report.py",
    "10_make_excel_report.py",
]


def main():
    code_dir = Path(__file__).resolve().parent
    print("\n" + "="*65 + "\nAUC Pipeline: Analysts' Earnings Forecasts\n" + "="*65)
    for s in SCRIPTS:
        print(f"\n{'='*65}\nRunning: {s}\n{'='*65}")
        subprocess.run([sys.executable, str(code_dir / s)], check=True)
    print("\n" + "="*65)
    print("ALL STEPS COMPLETE.")
    print("Primary deliverables:")
    print("  05_output/report/AUC_results_tables.xlsx")
    print("  05_output/tables/06_auc_main_and_sensitivity_summary.csv")
    print("  05_output/tables/07_quality_checks.csv  ← all 13 checks must PASS")
    print("  05_output/figures/roc_curve_main_score_le_2.png")
    print("  05_output/report/analysts_forecast_auc_report.md")
    print("="*65)


if __name__ == "__main__":
    main()
