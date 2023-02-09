import logging
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_processing.processing.figure_of_merit import FOM, fit_fom, n_sigma_classifier, n_sigma_classifier
from data_processing.processing.peak_finding import get_bases
from data_processing.processing.processing_configs import *
from data_processing.processing.psd import generate_psd
from data_processing.reporting.plot_configs import (
    QUOTIENT_LOWER_LIM,
    QUOTIENT_UPPER_LIM,
)
from data_processing.reporting.plotting import (
    plot_bounded_scatter,
    plot_classification_with_grouping,
    plot_fom,
    plot_histo
)
from data_processing.reporting.reporting import save_plot
from logging_helpers.setup_logger import (
    Messenger,
    setup_logger,
)

N_SIGMA = 3

def process(
    df: pd.DataFrame,
    root_dir: Path,
    buffer_id: Optional[str] = None,
    plot_path: Optional[Path] = None,
) -> tuple[float, int]:
    """Processes a given dataframe containg signals using the following pipeline:
    1. Adjusting DC offset
    2. Computing PSD Metrics
    3. Classification of PSD

    Parameters
    ----------
    df : DataFrame
        The dataframe that contains signals to be processed. Assumes no invalid signals
    root_dir: Path
        The path to the root of the experiment
    buffer_id: Optional[str]
        The id of the buffer used as prefix for file outputs
    plot_path : Optional[Path]
        The location of where plots will be stored; if `plot_destination` is not specified,
        no plots will be generated

    Returns
    -------
    float:
        The figure of merit (FOM) value
    int:
        The number of neutrons
    """
    logger = logging.getLogger(f"Data-Processing-{buffer_id}")
    messenger = Messenger(logger)
    setup_logger(logger, root_dir.parent.parent / "processing.log")
    t1 = time.perf_counter()

    messenger.info("Removing fine DC offset")
    df = df - FINE_DC_OFFSET

    messenger.info("Getting bases")
    output = df.apply(
        lambda series: get_bases(series, series.idxmax(), PEAK_OFFSET, TAIL_ONSET)
    )
    left_bases, right_bases = output.iloc[0, :], output.iloc[1, :]

    messenger.info("Computing PSD metrics")
    psd_report = generate_psd(df, left_bases, right_bases, df.max(), CUTOFF_VOLTAGE)

    messenger.info("Fitting Figure of Merit")
    counts, bins = np.histogram(psd_report.loc["tail / total"], N_BINS)
    params, _ = fit_fom(counts, bins)

    # Reorder params of instances when neutron counts > gamma counts
    if params[0] < params[3]:
        params = list(params)
        params[:3], params[3:] = params[3:], params[:3]
        params = tuple(params)

    print(psd_report)
    
    neutrons, gammas = n_sigma_classifier(
        psd_report, params[3:], CLASSIFIER_WINDOW_N
    )

    if plot_path is not None:
        messenger.info("Plotting processed data")
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

        classification, _ =  plot_classification_with_grouping(
            neutrons, 
            gammas, 
            params[3:], 
            psd_report.loc["amplitude"].max(), 
            CLASSIFIER_WINDOW_N
        )

        save_plot(
            plot_path / buffer_id, classification, f"{buffer_id}-classification.png"
        )


    messenger.debug(f"Elapsed Time: {time.perf_counter() - t1:.3f} s")
    return FOM(*params[0:2], *params[3:-1]), neutrons.shape[1]
