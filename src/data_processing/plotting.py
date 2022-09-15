import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processing.figure_of_merit import fit_fom, gaussian, FOM

from data_processing.plot_configs import *


def plot_signal(
    data: pd.DataFrame, sample_interval: float = 2
) -> tuple[plt.Figure, plt.Axes]:
    """Plots signals"""

    if data is None:
        return None

    fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))
    ax.plot(np.arange(0, data.shape[0] * sample_interval, sample_interval), data)

    ax.grid(visible=True)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)

    ax.set_xlabel("time (ns)", fontsize=FONT_SIZE)
    ax.set_ylabel("amplitude ($V$)", fontsize=FONT_SIZE)

    fig.tight_layout()

    return fig, ax


def plot_fom(
    psd: list, unimodal: bool = False, n_bins: int = 100, guesses: tuple = None
) -> tuple[plt.Figure, plt.Axes]:
    """Returns the Figure of Merit and fitting data for the biomdal gaussians"""
    counts, bins = np.histogram(psd, n_bins)

    params, cov = fit_fom(counts, bins, n_bins=n_bins, guesses=guesses)

    fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))

    ax.plot(bins[:-1], counts, "--", label="Original PSD")

    if not unimodal:
        ax.plot(
            bins[:-1], gaussian(bins[:-1], *params[0:3]), label="Neutron", c="orange"
        )

    ax.plot(bins[:-1], gaussian(bins[:-1], *params[3:]), label="$\gamma$", c="indigo")

    fom_val = (
        "N/A" if unimodal else f"{FOM(params[0], params[1], params[3], params[4]):.3f}"
    )
    ax.set_xlim(QUOTIENT_LOWER_LIM, QUOTIENT_UPPER_LIM)
    ax.set_title(f"FoM: {fom_val}")
    ax.legend()

    ax.set_ylabel("counts", fontsize=FONT_SIZE)
    ax.set_xlabel("tail/total (a.u.)", fontsize=FONT_SIZE)

    params_dict = {
        "mu1": params[0],
        "sigma1": params[1],
        "A1": params[2],
        "mu2": params[3],
        "sigma2": params[4],
        "A2": params[5],
    }

    fig.tight_layout()

    return fig, ax, params_dict, cov


def plot_scatter(
    x: list, y: list, xlabel: str, ylabel: str
) -> tuple[plt.Figure, plt.Axes]:
    """Returns a generic scatter plot"""
    fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_X))

    ax.scatter(x, y, marker=SCATTER_MARKER_DOT, s=SCATTER_MARKER_SIZE_SMALL)

    ax.set_ylim(QUOTIENT_LOWER_LIM, QUOTIENT_UPPER_LIM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()

    return fig, ax


def plot_classification(
    neutrons: pd.DataFrame,
    gammas: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))

    ax.scatter(
        neutrons.loc["amplitude"],
        neutrons.loc["tail / total"],
        marker="s",
        s=SCATTER_MARKER_SIZE_LARGE,
        color="k",
        label="Neutron",
    )

    ax.scatter(
        gammas.loc["amplitude"],
        gammas.loc["tail / total"],
        marker=".",
        s=SCATTER_MARKER_SIZE_LARGE,
        color="indigo",
        label="$\gamma$",
    )

    ax.set_xlabel("pulse amplitude ($V$)", fontsize=FONT_SIZE)
    ax.set_ylabel("tail / total (a.u.)", fontsize=FONT_SIZE)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(-0, 0.5)
    ax.legend()

    fig.tight_layout()

    return fig, ax
