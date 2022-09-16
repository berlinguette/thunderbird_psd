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
from data_processing.figure_of_merit import FOM, fit_fom, n_sigma_classifier
from data_processing.peak_finding import get_bases
from data_processing.plot_configs import QUOTIENT_LOWER_LIM, QUOTIENT_UPPER_LIM
from data_processing.plotting import (
    plot_bounded_scatter,
    plot_classification_with_grouping,
    plot_fom,
    plot_signal,
)
from data_processing.processing_configs import (
    CLASSIFIER_WINDOW_N,
    CUTOFF_VOLTAGE,
    FINE_DC_OFFSET,
    MAX_VOLTAGE,
    MIN_SNR,
    N_BINS,
    PEAK_OFFSET,
    PK_FIND_HEIGHT,
    PK_FIND_PROMINENCE,
    RMS_CUTOFF,
    TAIL_ONSET,
    TRIGGER_THRESHOLD,
)
from data_processing.psd import generate_psd
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
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
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
        fig.savefig(plot_path / f"{filepath.name.split('.')[0]}-cleaned_signals.png")

    report = generate_report(
        num_initial_signals,
        num_missing_signals,
        num_multipeaks,
        num_incomplete,
        num_low_snr,
        df_high_snr.shape[0],
    )

    return df_high_snr, report, props


def process_file(
    df: pd.DataFrame,
    elapsed_time: float,
    buffer_id: Optional[str] = None,
    plot_path: Optional[Path] = None,
):
    df = df - FINE_DC_OFFSET

    output = df.apply(
        lambda series: get_bases(series, series.idxmax(), PEAK_OFFSET, TAIL_ONSET)
    )
    left_bases, right_bases = output.iloc[0, :], output.iloc[1, :]

    psd_report = generate_psd(df, left_bases, right_bases, df.max(), CUTOFF_VOLTAGE)

    counts, bins = np.histogram(psd_report.loc["tail / total"], N_BINS)
    params, _ = fit_fom(counts, bins)
    neutrons, gammas, f_gate, f_gamma = n_sigma_classifier(
        psd_report, params[3:], CLASSIFIER_WINDOW_N, N_BINS
    )

    if plot_path is not None:
        tail_vs_total, _ = plot_bounded_scatter(
            psd_report.loc["total integral"],
            psd_report.loc["tail integral"],
            "total integral (a.u.)",
            "tail integral (a.u.)",
        )
        tail_vs_total.savefig(plot_path / f"{buffer_id}-tail_vs_total.png")

        q_vs_amplitude, _ = plot_bounded_scatter(
            psd_report.loc["amplitude"],
            psd_report.loc["tail / total"],
            "pulse amplitude ($V$)",
            "tail / total (a.u.)",
            (0, MAX_VOLTAGE),
            (QUOTIENT_LOWER_LIM, QUOTIENT_UPPER_LIM),
        )
        q_vs_amplitude.savefig(plot_path / f"{buffer_id}-q_vs_amplitude.png")

        fom, _, = plot_fom(
            psd_report.loc["tail / total"], params=params, unimodal=False, n_bins=N_BINS
        )
        fom.savefig(plot_path / f"{buffer_id}-fom.png")

        classification, _ = plot_classification_with_grouping(
            neutrons, gammas, f_gamma, f_gate, psd_report.loc["amplitude"].max()
        )
        classification.savefig(plot_path / f"{buffer_id}-classification.png")

    return FOM(*params[0:2], *params[3:-1]), neutrons.shape[1] / elapsed_time


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
