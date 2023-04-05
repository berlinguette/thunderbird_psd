import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_processing.processing.figure_of_merit import FOM, gaussian
from data_processing.processing.processing_configs import (
    CLASSIFIER_WINDOW_N,
    CUTOFF_VOLTAGE,
    DEFAULT_LOWER_ENERGY_BOUND)
from data_processing.processing.dataframe_manipulation import generate_neutron_signals
from data_processing.reporting.plot_configs import *
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import interpolate
from typing import Callable
from data_processing.dataframe_validation import DataframeColumn, get_df_col
from math import floor, log10


def plot_tail_vs_total(
    df: pd.DataFrame,
    experiment_display_name: str
) -> tuple[Figure, Axes]:
    y_resolution = _get_histogram_y_resolution()
    max_energy = get_df_col(df, DataframeColumn.CALIB_ENERGY).max()
    dataset_size = df.shape[0]

    fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))

    cmap = mpl.colormaps['gnuplot']  # type: ignore
    energy_col = get_df_col(df, DataframeColumn.ENERGY)
    short_col = get_df_col(df, DataframeColumn.ENERGYSHORT)

    ax.hist2d(
        energy_col,
        energy_col - short_col,
        bins=(HISTOGRAM_RES, y_resolution),
        norm=mpl.colors.LogNorm(),
        range=[[0, max_energy + .05], [0, 0.50]],
        cmap=cmap
    )

    fig.suptitle(
        f"Tail vs Total - {experiment_display_name}", fontsize=SUPTITLE_FONT_SIZE)
    ax.set_title(f"Event count = {dataset_size:,d}", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Tail", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel("Total", fontsize=AXIS_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONT_SIZE)
    ax.tick_params(axis='both', which='minor', labelsize=AXIS_TICK_FONT_SIZE)

    return fig, ax


def plot_psd_histogram(
    df: pd.DataFrame,
    colormap_name: str = 'gnuplot',
    colorbar: bool = False,
    **kwargs
) -> tuple[Figure, Axes]:
    y_resolution = _get_histogram_y_resolution()
    
    fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))

    energy_col = get_df_col(df, DataframeColumn.CALIB_ENERGY)
    psd_col = get_df_col(df, DataframeColumn.PSD)
    
    min_energy, max_energy = _get_range_with_margins(
        (energy_col.min(), energy_col.max()))
    min_psd, max_psd = _get_range_with_margins(
        psd_col.min(), psd_col.max())
    
    cmap = mpl.colormaps[colormap_name]  # type: ignore

    _, _, _, image = ax.hist2d(
        energy_col,
        psd_col,
        bins=(HISTOGRAM_RES, y_resolution),
        # range=[[0, max_energy * 1.01], [0, 0.50]],
        range=[[min_energy, max_energy], [min_psd, max_psd]],
        cmap=cmap,
        **kwargs
    )
    
    if colorbar:
        fig.colorbar(image, ax=ax)

    ax.set_ylim(min_psd, max_psd)
    ax.set_xlim(min_energy, max_energy)

    ax.set_xlabel("Energy (MeVee)", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel("PSD", fontsize=AXIS_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONT_SIZE)
    ax.tick_params(axis='both', which='minor', labelsize=AXIS_TICK_FONT_SIZE)

    return fig, ax


def add_fit_window_to_plot(
    axes: Axes,
    neutron_lb_fit: Callable,
    neutron_ub_fit: Callable,
    upper_energy_bound: float,
    lower_energy_bound: float = DEFAULT_LOWER_ENERGY_BOUND,
) -> Axes:
    energy_space = np.linspace(0, upper_energy_bound + 0.5, 200)
    axes.plot(energy_space, neutron_lb_fit(energy_space), 'r--')
    axes.plot(energy_space, neutron_ub_fit(energy_space), 'r--')
    # ax.vlines(all_slice_xs[0],
    #           neutron_lb_fit(all_slice_xs[0]),
    #           neutron_ub_fit(all_slice_xs[0]),
    #           'r', ls='--')
    axes.vlines(lower_energy_bound,
                neutron_lb_fit(lower_energy_bound),
                neutron_ub_fit(lower_energy_bound),
                'r', ls="--")  # type: ignore
    return axes


def plot_classification(
    df: pd.DataFrame,
    neutron_lb_fit: Callable,
    neutron_ub_fit: Callable,
    experiment_display_name: str,
    lower_energy_bound: float = DEFAULT_LOWER_ENERGY_BOUND
) -> tuple[Figure, Axes]:
    count_limit = 5

    # y_resolution = _get_histogram_y_resolution()
    max_energy = get_df_col(df, DataframeColumn.CALIB_ENERGY).max()

    # fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))

    class_col = get_df_col(df, DataframeColumn.NEUTRON_CLASS)
    # energy_col = get_df_col(df, DataframeColumn.CALIB_ENERGY)
    # psd_col = get_df_col(df, DataframeColumn.PSD)

    g_vs_n = class_col.map({True: 1, False: -1})
    # cmap = mpl.colormaps['RdBu_r']  # type: ignore

    # ax.hist2d(
    #     energy_col,
    #     psd_col,
    #     weights=g_vs_n,
    #     bins=(HISTOGRAM_RES, y_resolution),
    #     range=[[0, max_energy + .05], [0, 0.50]],
    #     cmap=cmap,
    #     vmin=-count_limit,
    #     vmax=count_limit,
    # )

    fig, ax = plot_psd_histogram(
        df,
        colormap_name='RdBu_r',
        weights=g_vs_n,
        vmin=-count_limit,
        vmax=count_limit
    )

    # energy_space = np.linspace(0, max_energy + 0.5, 200)
    # ax.plot(energy_space, neutron_lb_fit(energy_space), 'r--')
    # ax.plot(energy_space, neutron_ub_fit(energy_space), 'r--')
    # # ax.vlines(all_slice_xs[0],
    # #           neutron_lb_fit(all_slice_xs[0]),
    # #           neutron_ub_fit(all_slice_xs[0]),
    # #           'r', ls='--')
    # ax.vlines(lower_energy_bound,
    #           neutron_lb_fit(lower_energy_bound),
    #           neutron_ub_fit(lower_energy_bound),
    #           'r', ls="--") # type: ignore
    ax = add_fit_window_to_plot(
        ax,
        neutron_lb_fit,
        neutron_ub_fit,
        max_energy
    )

    # ax.set_ylim(0, 0.5)
    # ax.set_xlim(0, max_energy + .05)
    n_neutrons = df[df["NASA"]].shape[0]
    fig.suptitle(
        f"Neutron Classification: {experiment_display_name}",
        fontsize=SUPTITLE_FONT_SIZE)
    ax.set_title(f"Neutron count = {n_neutrons}", fontsize=TITLE_FONT_SIZE)
    # ax.set_xlabel("Energy (MeVee)", fontsize=AXIS_FONT_SIZE)
    # ax.set_ylabel("PSD", fontsize=AXIS_FONT_SIZE)
    # ax.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONT_SIZE)
    # ax.tick_params(axis='both', which='minor', labelsize=AXIS_TICK_FONT_SIZE)
    event_colors = [mpl.patches.Patch(facecolor=cmap(1.)),  # type: ignore
                    mpl.patches.Patch(facecolor=cmap(0.))]  # type: ignore
    ax.legend(event_colors, ["Neutrons", "Gamma"])

    return fig, ax


# def plot_neutron_traces(
#     neutron_events_df: pd.DataFrame,
# ) -> tuple[Figure, Axes]:
#     neutron_signals_df = generate_neutron_signals(neutron_events_df)

#     series_names = []

#     fig, ax = plt.subplots(figsize=(FIG_DIM_X,FIG_DIM_Y))

#     for index, row in neutron_events_df.iterrows():
#         n_psd = row['tail / total']
#         n_eng = row['CALIB_ENERGY']
#         n_time = row['EVENT_TIME']
#         n_ps_remain = row['EVENT_PS']
#         series_name = (f"Time {n_time} + {n_ps_remain} ps, ",
#                        f"PSD {n_psd:.4f}, E {n_eng:.4f} MeVee")
#         series_names.append(series_name)
#         ax.plot(0-neutron_signals_df[index]) # type: ignore

#     x_start, x_end = ax.get_xlim()
#     ax.xaxis.set_ticks(np.arange(x_start, x_end, 10)) # type: ignore
#     ax.legend(series_names)

#     return fig, ax

# def plot_signal(data: pd.DataFrame, sample_interval: float = 2) -> tuple[Figure, Axes]:
#     """Plots signals"""

#     if data is None:
#         return None

#     fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))
#     ax.plot(np.arange(0, data.shape[0] *
#             sample_interval, sample_interval), data)

#     ax.grid(visible=True)
#     ax.tick_params(axis="both", labelsize=FONT_SIZE)

#     ax.set_xlabel("time (ns)", fontsize=FONT_SIZE)
#     ax.set_ylabel("amplitude ($V$)", fontsize=FONT_SIZE)

#     fig.tight_layout()

#     return fig, ax


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

    ax.plot(bins[:-1], gaussian(bins[:-1], *params[3:]),
            label="$\gamma$", c="indigo")

    fom_val = (
        "N/A" if unimodal else f"{FOM(params[0], params[1], params[3], params[4]):.3f}"
    )
    ax.set_xlim(QUOTIENT_LOWER_LIM, QUOTIENT_UPPER_LIM)
    ax.set_title(f"FoM: {fom_val}", fontsize=TITLE_FONT_SIZE)
    ax.legend()

    ax.set_ylabel("counts", fontsize=AXIS_FONT_SIZE)
    ax.set_xlabel("tail/total (a.u.)", fontsize=AXIS_FONT_SIZE)

    fig.tight_layout()

    return fig, ax


def plot_scatter(
    x: list,
    y: list,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_X))
    ax.scatter(x, y,
               marker=SCATTER_MARKER_DOT,  # type: ignore
               s=SCATTER_MARKER_SIZE_SMALL)
    fig.tight_layout()
    return fig, ax


# def plot_bounded_scatter(
#     x: list,
#     y: list,
#     xlabel: str,
#     ylabel: str,
#     xbounds: tuple = None,
#     ybounds: tuple = None,
# ) -> tuple[Figure, Axes]:
#     """Returns a generic scatter plot"""
#     fig, ax = plot_scatter(x, y)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     ax.set_xlim(xbounds)
#     ax.set_ylim(ybounds)
#     return fig, ax


# def plot_classification(
#     neutrons: pd.DataFrame,
#     gammas: pd.DataFrame,
# ) -> tuple[Figure, Axes]:
#     fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))

#     ax.scatter(
#         neutrons.loc["amplitude"],
#         neutrons.loc["tail / total"],
#         marker="s",
#         s=SCATTER_MARKER_SIZE_LARGE,
#         color="k",
#         label="Neutron",
#     )

#     ax.scatter(
#         gammas.loc["amplitude"],
#         gammas.loc["tail / total"],
#         marker=".",
#         s=SCATTER_MARKER_SIZE_LARGE,
#         color="indigo",
#         label="$\gamma$",
#     )

#     ax.set_xlabel("pulse amplitude ($V$)", fontsize=FONT_SIZE)
#     ax.set_ylabel("tail / total (a.u.)", fontsize=FONT_SIZE)
#     ax.set_xlim(0, 2.5)
#     ax.set_ylim(-0, 0.5)
#     ax.legend()

#     fig.tight_layout()

#     return fig, ax


# def plot_classification_with_grouping_OLD(
#     neutrons: pd.DataFrame,
#     gammas: pd.DataFrame,
#     f_gamma: interpolate.interp1d,
#     f_gate: interpolate.interp1d,
#     max_voltage: float,
# ):
#     fig, ax = plot_classification(neutrons, gammas)
#     voltage_space = np.linspace(0, max_voltage, CLASSIFICATION_PLOT_RES)
#     ax.plot(f_gate(voltage_space), voltage_space, "r--")

#     ax.plot(f_gamma(voltage_space), voltage_space, "r--")

#     ax.text(1.73, 0.48, s=f"n_neutron = {neutrons.shape[1]}")

#     ax.text(1.73, 0.46, s=f"n_gamma = {gammas.shape[1]}")

#     ax.fill_betweenx(
#         voltage_space,
#         f_gamma(voltage_space),
#         f_gate(voltage_space),
#         alpha=0.1,
#         color="orange",
#     )
#     ax.vlines(CUTOFF_VOLTAGE, 0, 1, color="k", linestyles="--", linewidth=1.2)
#     ax.set_title(
#         f"Neutron Classification at n={CLASSIFIER_WINDOW_N} and cutoff = {CUTOFF_VOLTAGE:.3f}V"
#     )
#     fig.tight_layout()
#     return fig, ax

# def plot_classification_with_grouping(
#     neutrons: pd.DataFrame,
#     gammas: pd.DataFrame,
#     params: tuple,
#     max_voltage: float,
#     cutoff_voltage: float,
#     n: int
# ) -> tuple:
#     fig, ax = plot_classification(neutrons, gammas)
#     voltage_space = np.linspace(0, max_voltage, CLASSIFICATION_PLOT_RES)

#     f_gamma = gaussian(voltage_space, *params)
#     mu, sigma, A = params
#     f_gate = gaussian(voltage_space, mu + n * sigma, sigma, A)

#     ax.plot(f_gamma[f_gamma.argmax():], voltage_space[f_gamma.argmax():], "r--")
#     ax.plot(f_gate[f_gate.argmax():], voltage_space[f_gate.argmax():], "r--")

#     ax.text(1.73, 0.48, s=f"n_neutron = {neutrons.shape[1]}")

#     ax.text(1.73, 0.46, s=f"n_gamma = {gammas.shape[1]}")

#     filt = f_gamma[f_gamma.argmax():] < f_gate[f_gamma.argmax():]
#     ax.fill_betweenx(
#         voltage_space[f_gamma.argmax():],
#         f_gamma[f_gamma.argmax():],
#         f_gate[f_gamma.argmax():],
#         where=filt,
#         alpha=0.1,
#         color="orange",
#     )
#     ax.vlines(cutoff_voltage, 0, 1, color="k", linestyles="--", linewidth=1.2)
#     ax.set_title(
#         f"Neutron Classification at n={n} and cutoff = {cutoff_voltage:.3f}V"
#     )
#     fig.tight_layout()

#     return fig, ax

def _get_histogram_y_resolution(
    x_resolution: int = HISTOGRAM_RES,
    plot_width: int = FIG_DIM_X,
    plot_height: int = FIG_DIM_Y
) -> int:
    return x_resolution*plot_width//plot_height


def _get_range_with_margins(
    value: tuple[float, float], 
    margin_multiplier: float = 0.01
) -> tuple[float, float]:
    start = min(value)
    end = max(value)
    width = end - start
    if width == 0:
        margin = start * margin_multiplier 
    else:
        margin = width * margin_multiplier
    return (start - margin, end + margin)
