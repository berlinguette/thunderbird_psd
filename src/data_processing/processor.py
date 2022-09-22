import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from logging_helpers.setup_logger import cleanup_logger, message_info, setup_logger

from data_processing.cleaning.cleaner import clean_file
from data_processing.processing.processor import process
from data_processing.saving.io import dump_settings, save_parquet, save_report


def process_file(
    parquet_path: Path,
    plot_destination: Optional[Path] = None,
    report_destination: Optional[Path] = None,
) -> None:
    """Executes data processing pipeline and generates several pieces of data.
    - Report of data cleaning statistics
    - (TODO:) Report of data processing statistics
    - Settings used during processing
    - Plots:
        - Cleaned signals plot
        - Neutron/Gamma Classification
        - Figure of Merit (FOM)
        - tail vs total integrals
        - tail/total integrals (q) vs amplitude
    - A parquet file post data cleaning

    **NOTE:** This method assumes that `ROOT_DIR` is located in a directory that follows the parquet file
    is stored **3** folders into the experiment root; e.g. `./root/raw_data/parquet/my.parquet`

    Parameters
    ----------
    parquet_path : Path
        The path to the parquet file
    plot_destination : Optional[Path]
        The location where plots will be stored. If `plot_destination` is not specified, no plots
        will be generated
    report_destination : Optional[Path]
        The location where reports will be stored. If not specified, data will be stored in a `processed_data`
        folder inside of root

    Returns
    -------
    float:
        The figure of merit (FOM) value
    int:
        The number of neutrons
    float:
        The counts per second of neutrons
    """
    ROOT_DIR = parquet_path.parent.parent.parent
    EXP_TIMES_PATH = ROOT_DIR / "raw_data/exp_times.csv"
    PARQ_DESTINATION = ROOT_DIR / "processed_data/cleaned_buffers"

    uid = parquet_path.name.split(".")[0]

    logger = logging.getLogger(f"Data-Pipeline-{uid}")
    setup_logger(logger, ROOT_DIR.parent.parent / "processing.log")
    message_info(f"--- Starting processing for buffer {uid} ---", logger)
    t1 = time.perf_counter()

    df_clean, report, _ = clean_file(parquet_path, ROOT_DIR, plot_destination)

    message_info("Dumping settings", logger)
    dump_settings(
        PARQ_DESTINATION / "settings.toml"
        if report_destination is None
        else report_destination
    )

    message_info("Saving processing report", logger)
    save_report(
        uid,
        report,
        PARQ_DESTINATION / "report.toml"
        if report_destination is None
        else report_destination,
    )

    message_info("Saving processed parquet file", logger)
    filename = re.sub("(\d+\-\d+)(.parquet)", r"\1_clean\2", parquet_path.name)
    save_parquet(df_clean, filename, PARQ_DESTINATION)

    message_info("Computing values", logger)
    buffer_number = int(re.split("-(\d+)[_.]", parquet_path.name)[1])
    elapsed_time = pd.read_csv(EXP_TIMES_PATH, index_col=0).at[
        buffer_number, "elapsed_s"
    ]

    fom, counts = process(df_clean, ROOT_DIR, uid, plot_destination)
    cps = counts  / elapsed_time

    processing_time = time.perf_counter() - t1

    # TODO: Update this
    print(f"FOM={fom:.3f}; CPS={cps:.3f}")

    message_info(
        f"--- Completed processing buffer {uid} in {processing_time:.3f} s ---", logger
    )
    cleanup_logger(logger)

    return fom, counts, cps

def process_directory(
    directory: Path,
    n_start: Optional[int] = 1,
    n_end: Optional[int] = None,
    plot_destination: Optional[Path] = None,
    config_destination: Optional[Path] = None,
) -> None:
    """Executes data processing pipeline on raw parquets in a directory.
    See `process_file()` for output information.

    Parameters
    ----------
    directory : Path
        The path to the directory that contains the parquet files
    n_start : Optional[int]
        The starting index of parquets to use; default set to 1 since `exp_times.csv`
        does not correctly record elapsed time
    n_end : Optional[int]
        The number of parquets to process. if not specified, the entire directory
        will be processed starting from `n_start` to the second last parquet.
        Last parquet is also removed as `exp_times.csv` does not correctly record elapsed time
    plot_destination : Optional[Path]
        The location where plots will be stored. If `plot_destination` is not specified, no plots
        will be generated
    report_destination : Optional[Path]
        The location where reports will be stored. If not specified, data will be stored in a `processed_data`
        folder inside of root
    """
    PARQ_PATHS = [f for f in directory.iterdir()]
    n_end = n_end + 1 if n_end is not None else -1

    for PARQ_PATH in PARQ_PATHS[n_start:n_end]:
        cps, counts, fom = process_file(PARQ_PATH, plot_destination, config_destination)


if __name__ == "__main__":
    ROOT_DIR = Path("../sample_datasets/20220906_AmBe/")

    PARQ_PATH = ROOT_DIR / "raw_data/parquet/20220906-0005.parquet"

    PLOT_PATH = ROOT_DIR / "processed_data/plots"

    # process_file(PARQ_PATH, PLOT_PATH, None)

    process_directory(
        ROOT_DIR / "raw_data/parquet", n_end=2, plot_destination=PLOT_PATH
    )
