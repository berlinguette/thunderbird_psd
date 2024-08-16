import logging
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_processing.cleaning.cleaning_configs import *
from data_processing.cleaning.data_cleaning import (
    filter_incomplete_triggers,
    filter_low_snr,
    filter_multipeaks,
    rms,
    subtract_rms,
)
# from data_processing.reporting.plotting import plot_signal
from data_processing.reporting.reporting import generate_report, save_plot
from data_processing.saving.io import load_exp_info
from logging_helpers.setup_logger import Messenger, setup_logger


def clean_file(
    filepath: Path,
    root_dir: Path,
    plot_destination: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """[DEPRECATED]Cleans the given parquet file using the following pipeline:
    1. Removing invalid entries
    2. Recentering mean DC offset
    3. Filtering multipeak signals
    4. Filtering low signal-to-noise

    Parameters
    ----------
    filepath : Path
        The path to the parquet file
    root_dir : Path
        The path to the root directory of the parquet file
    plot_path : Optional[Path]
        The location of where plots will be stored; if `plot_destination` is not specified,
        no plots will be generated

    Returns
    -------
    DataFrame:
        Contains the dataframe of cleaned signals
    Dict:
        Contains the cleaning report
    DataFrame:
        Contains peak information generated during multipeak filtering
    """
    file_name = filepath.name.split(".")[0]
    logger = logging.getLogger(f"Data-Cleaning-{file_name}")
    messenger = Messenger(logger)
    setup_logger(logger, root_dir.parent.parent / "processing.log")
    t1 = time.perf_counter()

    messenger.info("Reading parquet file")
    df = pd.read_parquet(filepath)
    df.columns = df.columns.astype("int16")
    num_initial_signals = df.shape[0]

    messenger.info("Dropping invalid data points")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    num_missing_signals = num_initial_signals - df.shape[0]

    df = df.T

    messenger.info("Removing baseline offset")
    dc_offset = df.iloc[0:50].mean()
    df_offset = -1 * (df - dc_offset)
    baseline_rms = df_offset.apply(lambda x: rms(x, RMS_CUTOFF))
    df_offset = df_offset.apply(lambda x: subtract_rms(x, baseline_rms))

    messenger.info("Filtering multipeaks")
    df_single_peaks, props = filter_multipeaks(
        df_offset, height=PK_FIND_HEIGHT, prominence=PK_FIND_PROMINENCE
    )
    num_multipeaks = df_offset.shape[0] - df_single_peaks.shape[0]

    df_complete_triggers = filter_incomplete_triggers(
        df_single_peaks, TRIGGER_THRESHOLD
    )
    num_incomplete = df_single_peaks.shape[0] - df_complete_triggers.shape[0]

    messenger.info("Filtering low signal-to-noise")
    peak_heights = props.apply(lambda x: x["peak_heights"][0])
    df_high_snr = filter_low_snr(
        df_complete_triggers, peak_heights, baseline_rms, MIN_SNR
    )
    num_low_snr = df_complete_triggers.shape[0] - df_high_snr.shape[0]

    if plot_destination is not None:
        messenger.info("Plotting clean signals")
        try:
            exp_info_path = root_dir / "exp_info.toml"
            exp_info = load_exp_info(exp_info_path)
            sample_interval = exp_info["picoscope"]["sample_interval"]
        except FileNotFoundError:
            sample_interval = 2  # Assume it's 2 ns

        fig, _ = plot_signal(df_complete_triggers, sample_interval)
        save_plot(plot_destination / file_name, fig, f"{file_name}-cleaned_signals.png")
        plt.close(fig)

    messenger.info("Generating cleaning report")
    report = generate_report(
        num_initial_signals,
        num_missing_signals,
        num_multipeaks,
        num_incomplete,
        num_low_snr,
        df_high_snr.shape[0],
    )

    messenger.debug(f"Elapsed Time: {time.perf_counter() - t1:.3f} s")
    return df_high_snr, report, props
