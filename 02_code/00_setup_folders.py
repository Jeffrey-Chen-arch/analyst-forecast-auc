from config import *
from src.common import ensure_dirs, setup_logging


def main():
    logger = setup_logging(LOG_DIR / "00_setup_folders.log")
    dirs = [
        INPUT_DIR,
        RAW_DIR, RAW_IBES_DIR, RAW_LINK_DIR,
        INTERMEDIATE_DIR, INT_SAMPLE_DIR, INT_LINK_DIR, INT_IBES_DIR, INT_MERGE_DIR,
        OUTPUT_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR,
        LOG_DIR,
    ]
    ensure_dirs(dirs)
    logger.info("Created project folders under %s", PROJECT_ROOT)


if __name__ == "__main__":
    main()
