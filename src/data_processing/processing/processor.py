from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from reporting.plot_configs import QUOTIENT_LOWER_LIM, QUOTIENT_UPPER_LIM
from reporting.plotting import (
    plot_bounded_scatter,
    plot_classification_with_grouping,
    plot_fom,
)
from reporting.reporting import save_plot

from processing.figure_of_merit import FOM, fit_fom, n_sigma_classifier
from processing.peak_finding import get_bases
from processing.processing_configs import *
from processing.psd import generate_psd


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
        save_plot(
            plot_path / buffer_id, tail_vs_total, f"{buffer_id}-tail_vs_total.png"
        )

        q_vs_amplitude, _ = plot_bounded_scatter(
            psd_report.loc["amplitude"],
            psd_report.loc["tail / total"],
            "pulse amplitude ($V$)",
            "tail / total (a.u.)",
            (0, MAX_VOLTAGE),
            (QUOTIENT_LOWER_LIM, QUOTIENT_UPPER_LIM),
        )
        save_plot(
            plot_path / buffer_id, q_vs_amplitude, f"{buffer_id}-q_vs_amplitude.png"
        )

        fom, _, = plot_fom(
            psd_report.loc["tail / total"], params=params, unimodal=False, n_bins=N_BINS
        )
        save_plot(plot_path / buffer_id, fom, f"{buffer_id}-fom.png")

        classification, _ = plot_classification_with_grouping(
            neutrons, gammas, f_gamma, f_gate, psd_report.loc["amplitude"].max()
        )
        save_plot(
            plot_path / buffer_id, classification, f"{buffer_id}-classification.png"
        )

    return FOM(*params[0:2], *params[3:-1]), neutrons.shape[1] / elapsed_time
