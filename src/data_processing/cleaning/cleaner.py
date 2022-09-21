import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from data_processing.cleaning.cleaning_configs import *
from data_processing.cleaning.data_cleaning import (filter_incomplete_triggers,
                                                    filter_low_snr,
                                                    filter_multipeaks, rms,
                                                    subtract_rms)
from data_processing.reporting.plotting import plot_signal
from data_processing.reporting.reporting import generate_report, save_plot
from data_processing.saving.io import load_exp_info
from logging_helpers.setup_logger import (cleanup_logger, message_debug,
                                          message_info, setup_logger)


def clean_file(
    filepath: Path,
    root_dir: Path,
    plot_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    file_name = filepath.name.split(".")[0]
    logger = logging.getLogger(f"Data-Cleaning-{file_name}")
    setup_logger(logger, root_dir.parent.parent)
    t1 = time.perf_counter()

    message_info("Reading parquet file", logger)
    df = pd.read_parquet(filepath)
    df.columns = df.columns.astype("int16")
    num_initial_signals = df.shape[0]

    message_info("Dropping invalid data points", logger)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    num_missing_signals = num_initial_signals - df.shape[0]

    df = df.T

    message_info("Removing baseline offset", logger)
    dc_offset = df.iloc[0:50].mean()
    df_offset = -1 * (df - dc_offset)
    baseline_rms = df_offset.apply(lambda x: rms(x, RMS_CUTOFF))
    df_offset = df_offset.apply(lambda x: subtract_rms(x, baseline_rms))

    message_info("Filtering multipeaks", logger)
    df_single_peaks, props = filter_multipeaks(
        df_offset, height=PK_FIND_HEIGHT, prominence=PK_FIND_PROMINENCE
    )
    num_multipeaks = df_offset.shape[0] - df_single_peaks.shape[0]

    df_complete_triggers = filter_incomplete_triggers(
        df_single_peaks, TRIGGER_THRESHOLD
    )
    num_incomplete = df_single_peaks.shape[0] - df_complete_triggers.shape[0]

    message_info("Filtering low signal-to-noise", logger)
    peak_heights = props.apply(lambda x: x["peak_heights"][0])
    df_high_snr = filter_low_snr(
        df_complete_triggers, peak_heights, baseline_rms, MIN_SNR
    )
    num_low_snr = df_complete_triggers.shape[0] - df_high_snr.shape[0]

    if plot_path is not None:
        message_info("Plotting clean signals", logger)
        exp_info_path = root_dir / "exp_info.toml"
        exp_info = load_exp_info(exp_info_path)
        sample_interval = exp_info["picoscope"]["sample_interval"]
        fig, _ = plot_signal(df_complete_triggers, sample_interval)
        save_plot(plot_path / file_name, fig, f"{file_name}-cleaned_signals.png")

    message_info("Generating cleaning report", logger)
    report = generate_report(
        num_initial_signals,
        num_missing_signals,
        num_multipeaks,
        num_incomplete,
        num_low_snr,
        df_high_snr.shape[0],
    )

    message_debug(f"Elapsed Time: {time.perf_counter() - t1}", logger, on_screen=False)
    return df_high_snr, report, props
