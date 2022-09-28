import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from logging_helpers.setup_logger import cleanup_logger, message_info, setup_logger

from data_processing.cleaning.cleaner import clean_file
from data_processing.processing.processor import process
from data_processing.saving.io import (
    dump_settings,
    save_cleaning_report,
    save_parquet,
    save_results,
    create_folder
)


def process_file(
    parquet_path: Path,
    plot_destination: Optional[Path] = None,
    report_destination: Optional[Path] = None,
) -> None:
    """Executes data processing pipeline and generates several pieces of data.
    - Report of data cleaning statistics
    - Report of data processing statistics
    - Settings used during processing
    - Plots:
        - Cleaned signals plot
        - Neutron/Gamma Classification
        - Figure of Merit (FOM)
        - tail vs total integrals
        - tail/total integrals (q) vs amplitude
    - A parquet file post data cleaning

    Parameters
    ----------
    parquet_path : Path
        The path to the parquet file. This method assumes that parquets are stored in a directory structure
        with at least **3** levels: e.g. `./root/raw_data/parquet/my.parquet`
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
    setup_logger(logger, PARQ_DESTINATION.parent / "processing.log")
    message_info(f"--- Starting processing for buffer {uid} ---", logger)
    t1 = time.perf_counter()

    df_clean, report, _ = clean_file(parquet_path, ROOT_DIR, plot_destination)

    message_info("Dumping settings", logger)
    dump_settings(
        PARQ_DESTINATION / "cleaning_settings.toml"
        if report_destination is None
        else report_destination
    )

    message_info("Saving processing report", logger)
    save_cleaning_report(
        uid,
        report,
        PARQ_DESTINATION / "cleaning_report.toml"
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
    cps = counts / elapsed_time

    processing_time = time.perf_counter() - t1

    message_info(
        f"Processed {uid} showing: Count={counts} neutrons | CPS={cps:.3f} neutrons/sec | FOM={fom:.3f}", logger)

    message_info(
        f"--- Completed processing buffer {uid} in {processing_time:.3f} s ---", logger
    )

    return fom, counts, cps


def process_directory(
    directory: Path,
    n_start: Optional[int] = 1,
    n_end: Optional[int] = None,
    plot_destination: Optional[Path] = None,
    config_destination: Optional[Path] = None,
) -> None:
    """Executes data processing pipeline on raw parquets in a directory.
    See `process_file()` for output information. An output `.csv` is created/updated
    in a `processed_data` folder two directories up. This output contains the
    counts, counts per second, and figure of merit (FOM) value for each buffer that
    was processed.

    Parameters
    ----------
    directory : Path
        The path to the directory that contains the parquet files. `directory` is assumed to
        have a structure with at least **3** levels: e.g. `./root/raw_data/parquet/my.parquet`
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
    logger = logging.getLogger("Directory-Processor")
    setup_logger(logger, directory.parent.parent.parent / "processor.log")

    PARQ_PATHS = [f for f in directory.iterdir()]
    n_end = n_end + 1 if n_end is not None else -1


    create_folder(directory.parent.parent / "processed_data/cleaned_buffers")

    if plot_destination is not None:
        create_folder(plot_destination)

    message_info("Intiating directory processing...", logger)
    column_labels = ["neutron_count", "count_rate [s^-1]", "FOM"]
    results = pd.DataFrame(columns=column_labels)
    for PARQ_PATH in PARQ_PATHS[n_start:n_end]:
        message_info(f"Processing {PARQ_PATH.name}")
        fom, counts, cps = process_file(
            PARQ_PATH, plot_destination, config_destination)
        df = pd.DataFrame(
            [(int(counts), round(cps, 3), round(fom, 3))],
            columns=column_labels,
            index=[PARQ_PATH.name.split(".")[0]],
        )
        results = pd.concat([results, df])

    save_results(results, directory.parent.parent /
                 "processed_data/report.csv")

    message_info("Completed directory processing!", logger)
    cleanup_logger(logger)


if __name__ == "__main__":

    BASE_PATH = Path("C:/Users/Neutron Computer/Documents/Data/20220922_AmBedistancetests/")
    exlcusions = ["AmBe_0cm_1525V", "AmBe_30cm_1525V"]
    for ROOT_DIR in BASE_PATH.iterdir():

        if not ROOT_DIR.is_dir() or ROOT_DIR.name not in ["AmBe_20cm_1525V"]:
            continue

        
        PLOT_PATH = ROOT_DIR / "processed_data/plots"


        # 0 and 30 cm bad...
        process_directory(
            ROOT_DIR / "raw_data/parquet", plot_destination=PLOT_PATH
        )

    # ROOT_DIR = Path("C:/Users/Neutron Computer/Documents/Data/20220922_AmBedistancetests/AmBe_100cm_1525V")
    # PARQ_PATH = ROOT_DIR / "raw_data/parquet/20220906-0005.parquet"
    # process_file(PARQ_PATH, PLOT_PATH, None)

