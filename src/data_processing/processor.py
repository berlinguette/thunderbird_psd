import logging
import re
import time
from pathlib import Path
from typing import Optional

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


def process_file(
    parquet_path: Path,
    plot_destination: Optional[Path] = None,
    config_destination: Optional[Path] = None,
) -> None:
    ROOT_DIR = parquet_path.parent.parent.parent
    EXP_TIMES_PATH = ROOT_DIR / "raw_data/exp_times.csv"
    PARQ_DESTINATION = ROOT_DIR / "processed_data/cleaned_buffers"

    uid = parquet_path.name.split(".")[0]

    logger = logging.getLogger(f"Data-Pipeline-{uid}")
    setup_logger(logger, ROOT_DIR.parent.parent)
    t1 = time.perf_counter()

    df_clean, report, _ = clean_file(parquet_path, ROOT_DIR, plot_destination)

    message_info("Dumping settings", logger)
    dump_settings(
        PARQ_DESTINATION / "settings.toml"
        if config_destination is None
        else config_destination
    )

    message_info("Saving processing report", logger)
    save_report(
        uid,
        report,
        PARQ_DESTINATION / "report.toml"
        if config_destination is None
        else config_destination,
    )

    message_info("Saving processed parquet file", logger)
    filename = re.sub("(\d+\-\d+)(.parquet)", r"\1_clean\2", parquet_path.name)
    save_parquet(df_clean, filename, PARQ_DESTINATION)

    message_info("Computing values", logger)
    buffer_number = int(re.split("-(\d+)[_.]", parquet_path.name)[1])
    elapsed_time = pd.read_csv(EXP_TIMES_PATH, index_col=0).at[
        buffer_number, "elapsed_s"
    ]

    fom, cps = process_file(df_clean, elapsed_time, ROOT_DIR, uid, plot_destination)

    processing_time = time.perf_counter() - t1

    # TODO: Update this
    print(f"Elapsed Time: {processing_time:.3f} s || FOM={fom:.3f}; CPS={cps:.3f}")

    message_debug(f"Elapsed Time: {processing_time:.3f} s", logger, on_screen=False)
    cleanup_logger(logger)


if __name__ == "__main__":
    ROOT_DIR = Path("../sample_datasets/20220906_AmBe/")

    PARQ_PATH = ROOT_DIR / "raw_data/parquet/20220906-0005.parquet"

    PLOT_PATH = ROOT_DIR / "processed_data/plots"

    process_file(PARQ_PATH, PLOT_PATH, None)
