import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_processing.data_cleaning import (
    filter_incomplete_triggers,
    filter_low_snr,
    filter_multipeaks,
    rms,
    subtract_rms,
)
from data_processing.plotting import plot_signal
from data_processing.processing_configs import (
    MIN_SNR,
    PK_FIND_HEIGHT,
    PK_FIND_PROMINENCE,
    RMS_CUTOFF,
    TRIGGER_THRESHOLD,
)
from data_processing.utils import (
    dump_settings,
    generate_report,
    load_exp_info,
    save_parquet,
    save_report,
)


def clean_file(
    filepath: Path,
    root_dir: Path,
    plot_path: Optional[Path] = None,
):
    df = pd.read_parquet(filepath)
    df.columns = df.columns.astype("int16")
    num_initial_signals = df.shape[0]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    num_missing_signals = num_initial_signals - df.shape[0]

    df = df.T

    dc_offset = df.iloc[0:50].mean()
    df_offset = -1 * (df - dc_offset)

    baseline_rms = df_offset.apply(lambda x: rms(x, RMS_CUTOFF))
    df_offset = df_offset.apply(lambda x: subtract_rms(x, baseline_rms))

    df_single_peaks, props = filter_multipeaks(
        df_offset, height=PK_FIND_HEIGHT, prominence=PK_FIND_PROMINENCE
    )
    num_multipeaks = df_offset.shape[0] - df_single_peaks.shape[0]

    df_complete_triggers = filter_incomplete_triggers(
        df_single_peaks, TRIGGER_THRESHOLD
    )
    num_incomplete = df_single_peaks.shape[0] - df_complete_triggers.shape[0]

    peak_heights = props.apply(lambda x: x["peak_heights"][0])
    df_high_snr = filter_low_snr(
        df_complete_triggers, peak_heights, baseline_rms, MIN_SNR
    )
    num_low_snr = df_complete_triggers.shape[0] - df_high_snr.shape[0]

    if plot_path is not None:
        exp_info_path = root_dir / "exp_info.toml"
        exp_info = load_exp_info(exp_info_path)
        sample_interval = exp_info["picoscope"]["sample_interval"]
        fig, _ = plot_signal(df_complete_triggers, sample_interval)
        fig.savefig(plot_path / "Cleaned Signals.png")

    report = generate_report(
        num_initial_signals,
        num_missing_signals,
        num_multipeaks,
        num_incomplete,
        num_low_snr,
        df_high_snr.shape[0],
    )

    return df_high_snr, report, props


if __name__ == "__main__":
    ROOT_DIR = Path("../sample_datasets/20220906_AmBe/")
    PARQ_PATH = ROOT_DIR / "raw_data/parquet/20220906-0005.parquet"

    DESTINATION = Path("../sample_datasets/AmBe_port_test")
    SETTINGS_PATH = DESTINATION / "settings.toml"
    REPORT_PATH = DESTINATION / "report.toml"

    t1 = time.perf_counter()

    df_clean, report, props = clean_file(
        PARQ_PATH,
        ROOT_DIR,
        destination=DESTINATION,
    )

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

    print(f"Compelted in: {time.perf_counter() - t1}")
