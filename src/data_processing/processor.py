import re
import time
from io.io import dump_settings, save_parquet, save_report
from pathlib import Path

import pandas as pd

from cleaning.cleaner import clean_file
from processing.processor import process_file

if __name__ == "__main__":
    ROOT_DIR = Path("../sample_datasets/20220906_AmBe/")
    PARQ_PATH = ROOT_DIR / "raw_data/parquet/20220906-0005.parquet"
    EXP_TIMES_PATH = ROOT_DIR / "raw_data/exp_times.csv"
    # PLOT_PATH = ROOT_DIR / "processed_data/plots"
    PLOT_PATH = None
    DESTINATION = ROOT_DIR / "processed_data/cleaned_buffers"
    SETTINGS_PATH = DESTINATION / "settings.toml"
    REPORT_PATH = DESTINATION / "report.toml"

    t1 = time.perf_counter()

    df_clean, report, _ = clean_file(PARQ_PATH, ROOT_DIR, PLOT_PATH)

    dump_settings(
        ROOT_DIR / "settings.toml" if SETTINGS_PATH is None else SETTINGS_PATH
    )

    uid = PARQ_PATH.name.split(".")[0]

    save_report(
        uid, report, ROOT_DIR / "report.toml" if REPORT_PATH is None else REPORT_PATH
    )

    filename = re.sub("(\d+\-\d+)(.parquet)", r"\1_clean\2", PARQ_PATH.name)
    save_parquet(
        df_clean,
        filename,
        ROOT_DIR if DESTINATION is None else DESTINATION,
    )

    buffer_number = int(re.split("-(\d+)[_.]", PARQ_PATH.name)[1])
    elapsed_time = pd.read_csv(EXP_TIMES_PATH, index_col=0).at[
        buffer_number, "elapsed_s"
    ]

    fom, cps = process_file(
        df_clean, elapsed_time, PARQ_PATH.name.split(".")[0], PLOT_PATH
    )
    print(f"FOM={fom}; CPS={cps}")

    print(f"Compelted in: {time.perf_counter() - t1}")
