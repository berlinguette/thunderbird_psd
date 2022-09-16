import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import interpolate

from data_processing.figure_of_merit import FOM, gaussian
from data_processing.plot_configs import *
from data_processing.processing_configs import CLASSIFIER_WINDOW_N, CUTOFF_VOLTAGE


def plot_signal(data: pd.DataFrame, sample_interval: float = 2) -> tuple[Figure, Axes]:
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
    psd: list, params: tuple, unimodal: bool = False, n_bins: int = 100
) -> tuple[Figure, Axes]:
    """Returns the Figure of Merit and fitting data for the biomdal gaussians"""
    counts, bins = np.histogram(psd, n_bins)

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

    fig.tight_layout()

    return fig, ax


def plot_scatter(
    x: list,
    y: list,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_X))
    ax.scatter(x, y, marker=SCATTER_MARKER_DOT, s=SCATTER_MARKER_SIZE_SMALL)
    fig.tight_layout()
    return fig, ax


def plot_bounded_scatter(
    x: list,
    y: list,
    xlabel: str,
    ylabel: str,
    xbounds: tuple = None,
    ybounds: tuple = None,
) -> tuple[Figure, Axes]:
    """Returns a generic scatter plot"""
    fig, ax = plot_scatter(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(xbounds)
    ax.set_ylim(ybounds)
    return fig, ax


def plot_classification(
    neutrons: pd.DataFrame,
    gammas: pd.DataFrame,
) -> tuple[Figure, Axes]:
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


def plot_classification_with_grouping(
    neutrons: pd.DataFrame,
    gammas: pd.DataFrame,
    f_gamma: interpolate.interp1d,
    f_gate: interpolate.interp1d,
    max_voltage: float,
):
    fig, ax = plot_classification(neutrons, gammas)
    voltage_space = np.linspace(0, max_voltage, CLASSIFICATION_PLOT_RES)
    ax.plot(f_gate(voltage_space), voltage_space, "r--")

    ax.plot(f_gamma(voltage_space), voltage_space, "r--")

    ax.text(1.73, 0.48, s=f"n_neutron = {neutrons.shape[1]}")

    ax.text(1.73, 0.46, s=f"n_gamma = {gammas.shape[1]}")

    ax.fill_betweenx(
        voltage_space,
        f_gamma(voltage_space),
        f_gate(voltage_space),
        alpha=0.1,
        color="orange",
    )
    ax.vlines(CUTOFF_VOLTAGE, 0, 1, color="k", linestyles="--", linewidth=1.2)
    ax.set_title(
        f"Neutron Classification at n={CLASSIFIER_WINDOW_N} and cutoff = {CUTOFF_VOLTAGE:.3f}V"
    )
    fig.tight_layout()
    return fig, ax
