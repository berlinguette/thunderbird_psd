import logging
import re
import time
from pathlib import Path

import pandas as pd
from logging_helpers.setup_logger import (
    cleanup_logger,
    message_debug,
    message_info,
    setup_logger,
)

from data_processing.cleaning.cleaner import clean_file
from data_processing.processing.processor import process_file
from data_processing.saving.io import dump_settings, save_parquet, save_report


def main(directory: Path):
    PARQ_PATH = directory / "raw_data/parquet/20220906-0005.parquet"
    EXP_TIMES_PATH = directory / "raw_data/exp_times.csv"
    PLOT_PATH = directory / "processed_data/plots"
    # PLOT_PATH = None
    DESTINATION = directory / "processed_data/cleaned_buffers"
    SETTINGS_PATH = DESTINATION / "settings.toml"
    REPORT_PATH = DESTINATION / "report.toml"

    uid = PARQ_PATH.name.split(".")[0]
    logger = logging.getLogger(f"Data-Pipeline-{uid}")
    setup_logger(logger, directory.parent.parent)
    t1 = time.perf_counter()

    df_clean, report, _ = clean_file(PARQ_PATH, directory, PLOT_PATH)

    message_info("Dumping settings", logger)
    dump_settings(
        directory / "settings.toml" if SETTINGS_PATH is None else SETTINGS_PATH
    )

    message_info("Saving processing report", logger)
    save_report(
        uid, report, directory / "report.toml" if REPORT_PATH is None else REPORT_PATH
    )

    message_info("Saving processed parquet file", logger)
    filename = re.sub("(\d+\-\d+)(.parquet)", r"\1_clean\2", PARQ_PATH.name)
    save_parquet(
        df_clean,
        filename,
        directory if DESTINATION is None else DESTINATION,
    )

    message_info("Computing values", logger)
    buffer_number = int(re.split("-(\d+)[_.]", PARQ_PATH.name)[1])
    elapsed_time = pd.read_csv(EXP_TIMES_PATH, index_col=0).at[
        buffer_number, "elapsed_s"
    ]

    fom, cps = process_file(df_clean, elapsed_time, ROOT_DIR, PLOT_PATH)
    print(f"FOM={fom}; CPS={cps}")  # TODO: Update this

    message_debug(f"Elapsed Time: {time.perf_counter() - t1}", logger, on_screen=False)
    cleanup_logger(logger)


if __name__ == "__main__":
    ROOT_DIR = Path("../sample_datasets/20220906_AmBe/")
    main(ROOT_DIR)
