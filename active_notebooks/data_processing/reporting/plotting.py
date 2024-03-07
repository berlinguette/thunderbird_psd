from math import ceil
from typing import Any, Callable, Literal, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_processing.dataframe_validation import DataframeColumn, get_df_col
from data_processing.processing.figure_of_merit import FOM, gaussian
from data_processing.processing.processing_configs import DEFAULT_LOWER_ENERGY_BOUND
from data_processing.reporting.plot_configs import *
from matplotlib.axes import Axes
from matplotlib.figure import Figure

Kwargs = dict[str, Any]
GraphData = dict[Literal["x"] | Literal["y"], pd.Series]
GraphingFunction = Callable[[Figure, Axes, GraphData, Kwargs], Axes]
AxesMatrix = list[list[Axes]]


def plot_single(
    graphing_function: GraphingFunction,
    data: GraphData,
    figsize: tuple[float, float] = (FIG_DIM_X, FIG_DIM_Y),
    supertitle: str | None = None,
    supertitle_font_size: float = SUPTITLE_FONT_SIZE,
    plot_kwargs: Kwargs | None = None,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=figsize, **kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}
    ax = graphing_function(fig, ax, data, plot_kwargs)

    if supertitle is not None:
        fig.suptitle(supertitle, fontsize=supertitle_font_size)

    return fig, ax


def plot_many(
    graphing_function: GraphingFunction,
    data: list[GraphData],
    figsize: tuple[float, float] = (FIG_DIM_X, FIG_DIM_Y),
    max_cols: int = SUBPLOTS_MAX_COLS,
    supertitle: str | None = None,
    supertitle_font_size: float = SUPTITLE_FONT_SIZE,
    plot_kwargs: list[Kwargs] | None = None,
    **kwargs,
):
    n_plots = len(data)
    n_cols = min(max_cols, n_plots)
    n_rows = ceil(n_plots / n_cols)

    subplots_result: tuple[Figure, AxesMatrix] = plt.subplots(
        figsize=figsize, ncols=n_cols, nrows=n_rows, **kwargs
    )
    fig, axs = subplots_result

    if plot_kwargs is None:
        plot_kwargs = [{} for subplot_data in data]
    for i, subplot_data in enumerate(data):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row][col]
        subplot_kwargs = plot_kwargs[i]

        ax = graphing_function(fig, ax, subplot_data, subplot_kwargs)
        axs[row][col] = ax

    if supertitle is not None:
        fig.suptitle(supertitle, fontsize=supertitle_font_size)

    return fig, axs


# Graphing Functions
def graph_tail_vs_total(
    fig: Figure, ax: Axes, data: GraphData, graph_kwargs: Kwargs
) -> Axes:
    x_resolution = graph_kwargs.get("x_resolution", HISTOGRAM_RES)
    y_resolution = graph_kwargs.get("y_resolution", HISTOGRAM_RES)
    max_energy = graph_kwargs.get("max_energy", 5)
    cmap = graph_kwargs.get("cmap", mpl.colormaps["gnuplot"])  # type: ignore
    title_font_size = graph_kwargs.get("title_font_size", TITLE_FONT_SIZE)
    axis_font_size = graph_kwargs.get("axis_font_size", AXIS_FONT_SIZE)
    axis_tick_font_size = graph_kwargs.get("axis_tick_font_size", AXIS_TICK_FONT_SIZE)

    data_x = data["x"]
    data_y = data["y"]
    dataset_size = data_x.shape[0]

    ax.hist2d(
        data_x,
        data_y,
        bins=(x_resolution, y_resolution),
        norm=mpl.colors.LogNorm(),
        range=[[0, max_energy + 0.05], [0, 0.50]],
        cmap=cmap,
    )

    ax.set_title(f"Event count = {dataset_size:,d}", fontsize=title_font_size)
    ax.set_xlabel("Tail", fontsize=axis_font_size)
    ax.set_ylabel("Total", fontsize=axis_font_size)
    ax.tick_params(axis="both", which="major", labelsize=axis_tick_font_size)
    ax.tick_params(axis="both", which="minor", labelsize=axis_tick_font_size)

    return ax


def graph_psd_histogram(
    fig: Figure, ax: Axes, data: GraphData, graph_kwargs: Kwargs
) -> Axes:
    x_resolution = graph_kwargs.get("x_resolution", HISTOGRAM_RES)
    y_resolution = graph_kwargs.get("y_resolution", HISTOGRAM_RES)
    energy_start_zero = graph_kwargs.get("energy_start_zero", False)
    cmap = graph_kwargs.get("cmap", mpl.colormaps["gnuplot"])  # type: ignore
    colorbar = graph_kwargs.get("colorbar", False)
    axis_font_size = graph_kwargs.get("axis_font_size", AXIS_FONT_SIZE)
    axis_tick_font_size = graph_kwargs.get("axis_tick_font_size", AXIS_TICK_FONT_SIZE)
    cmin = graph_kwargs.get("cmin", 0)

    data_x = data["x"]
    data_y = data["y"]

    min_energy, max_energy = _get_range_with_margins((data_x.min(), data_x.max()))
    if energy_start_zero:
        min_energy = 0
    min_psd, max_psd = _get_range_with_margins((data_y.min(), data_y.max()))

    _, _, _, image = ax.hist2d(
        data_x,
        data_y,
        bins=(x_resolution, y_resolution),
        range=[[min_energy, max_energy], [min_psd, max_psd]],
        cmap=cmap,
        cmin=cmin
    )

    if colorbar:
        fig.colorbar(image, ax=ax)

    ax.set_ylim(min_psd, max_psd)
    ax.set_xlim(min_energy, max_energy)
    ax.set_xlabel("Energy (MeVee)", fontsize=axis_font_size)
    ax.set_ylabel("PSD", fontsize=axis_font_size)
    ax.tick_params(axis="both", which="major", labelsize=axis_tick_font_size)
    ax.tick_params(axis="both", which="minor", labelsize=axis_tick_font_size)

    return ax


# All In One Plot Functions
def plot_tail_vs_total(
    df: pd.DataFrame, experiment_display_name: str, **kwargs
) -> tuple[Figure, Axes]:
    figsize = _popget(kwargs, "figsize", (FIG_DIM_X, FIG_DIM_Y))
    x_resolution = _popget(kwargs, "x_resolution", HISTOGRAM_RES)
    # cmap = _popget(kwargs, "cmap", mpl.colormaps["gnuplot"])  # type: ignore

    y_resolution = _get_histogram_y_resolution(
        x_resolution=x_resolution, plot_width=figsize[0], plot_height=figsize[1]
    )
    max_energy = get_df_col(df, DataframeColumn.CALIB_ENERGY).max()
    plot_kwargs = {
        "x_resolution": x_resolution,
        "y_resolution": y_resolution,
        "max_energy": max_energy,
        **kwargs,
    }

    energy_col = get_df_col(df, DataframeColumn.ENERGY)
    short_col = get_df_col(df, DataframeColumn.ENERGYSHORT)
    data: GraphData = {"x": energy_col, "y": energy_col - short_col}

    supertitle = f"Tail vs Total - {experiment_display_name}"

    fig, ax = plot_single(
        graph_tail_vs_total,
        data,
        figsize=figsize,
        supertitle=supertitle,
        supertitle_font_size=SUPTITLE_FONT_SIZE,
        plot_kwargs=plot_kwargs,
    )

    return fig, ax


def plot_psd_histogram(
    df: pd.DataFrame,
    colormap_name: str = "gnuplot",
    colorbar: bool = False,
    energy_start_zero: bool = False,
    **kwargs,
) -> tuple[Figure, Axes]:
    figsize = _popget(kwargs, "figsize", (FIG_DIM_X, FIG_DIM_Y))
    x_resolution = _popget(kwargs, "x_resolution", HISTOGRAM_RES)
    y_resolution = _get_histogram_y_resolution(
        x_resolution=x_resolution, plot_width=figsize[0], plot_height=figsize[1]
    )
    cmap = mpl.colormaps[colormap_name]  # type: ignore
    plot_kwargs = {
        "x_resolution": x_resolution,
        "y_resolution": y_resolution,
        "energy_start_zero": energy_start_zero,
        "cmap": cmap,
        "colorbar": colorbar,
        **kwargs,
    }

    energy_col = get_df_col(df, DataframeColumn.CALIB_ENERGY)
    psd_col = get_df_col(df, DataframeColumn.PSD)
    data: GraphData = {"x": energy_col, "y": psd_col}

    fig, ax = plot_single(
        graph_psd_histogram, data, figsize=figsize, plot_kwargs=plot_kwargs
    )

    return fig, ax


def add_fit_window_to_plot(
    axes: Axes,
    neutron_lb_fit: Callable,
    neutron_ub_fit: Callable,
    upper_energy_bound: float,
    lower_energy_bound: float = DEFAULT_LOWER_ENERGY_BOUND,
) -> Axes:
    energy_space = np.linspace(0, upper_energy_bound + 0.5, 200)
    axes.plot(energy_space, neutron_lb_fit(energy_space), "r--")
    axes.plot(energy_space, neutron_ub_fit(energy_space), "r--")
    # ax.vlines(all_slice_xs[0],
    #           neutron_lb_fit(all_slice_xs[0]),
    #           neutron_ub_fit(all_slice_xs[0]),
    #           'r', ls='--')
    axes.vlines(
        lower_energy_bound,
        neutron_lb_fit(lower_energy_bound),
        neutron_ub_fit(lower_energy_bound),
        "r",  # type: ignore
        ls="--",
    )  # type: ignore
    return axes


def plot_classification(
    df: pd.DataFrame,
    neutron_lb_fit: Callable,
    neutron_ub_fit: Callable,
    experiment_display_name: str,
    count_limit: int = 5,
    **kwargs,
) -> tuple[Figure, Axes]:
    # y_resolution = _get_histogram_y_resolution()
    max_energy = get_df_col(df, DataframeColumn.CALIB_ENERGY).max()

    # fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))

    class_col = get_df_col(df, DataframeColumn.NEUTRON_CLASS)
    # energy_col = get_df_col(df, DataframeColumn.CALIB_ENERGY)
    # psd_col = get_df_col(df, DataframeColumn.PSD)

    g_vs_n = class_col.map({True: 1, False: -1})
    cmap = mpl.colormaps["RdBu_r"]  # type: ignore

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
        colormap_name="RdBu_r",
        weights=g_vs_n,
        vmin=-count_limit,
        vmax=count_limit,
        energy_start_zero=True,
        **kwargs,
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
    ax = add_fit_window_to_plot(ax, neutron_lb_fit, neutron_ub_fit, max_energy)

    # ax.set_ylim(0, 0.5)
    # ax.set_xlim(0, max_energy + .05)
    n_neutrons = df[df["NASA"]].shape[0]
    fig.suptitle(
        f"Neutron Classification: {experiment_display_name}",
        fontsize=SUPTITLE_FONT_SIZE,
    )
    ax.set_title(f"Neutron count = {n_neutrons}", fontsize=TITLE_FONT_SIZE)
    # ax.set_xlabel("Energy (MeVee)", fontsize=AXIS_FONT_SIZE)
    # ax.set_ylabel("PSD", fontsize=AXIS_FONT_SIZE)
    # ax.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONT_SIZE)
    # ax.tick_params(axis='both', which='minor', labelsize=AXIS_TICK_FONT_SIZE)
    event_colors = [
        mpl.patches.Patch(facecolor=cmap(1.0)),  # type: ignore
        mpl.patches.Patch(facecolor=cmap(0.0)),  # type: ignore
    ]  # type: ignore
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

    ax.plot(bins[:-1], gaussian(bins[:-1], *params[3:]), label="$\gamma$", c="indigo")

    fom_val = (
        "N/A" if unimodal else f"{FOM(params[0], params[1], params[3], params[4]):.3f}"
    )
    ax.set_xlim(QUOTIENT_LOWER_LIM, QUOTIENT_UPPER_LIM)
    ax.set_title(f"FoM: {fom_val}", fontsize=TITLE_FONT_SIZE)
    ax.legend()

    ax.set_ylabel("counts", fontsize=AXIS_FONT_SIZE)
    ax.set_xlabel("tail/total (a.u.)", fontsize=AXIS_FONT_SIZE)

    fig.tight_layout()

    return fig, axplot


def plot_scatter(
    x: list | pd.Series, y: list | pd.Series, **kwargs
) -> tuple[Figure, Axes]:
    figsize = kwargs.get("figsize", (FIG_DIM_X, FIG_DIM_Y))
    marker = kwargs.get("marker", SCATTER_MARKER_DOT)
    marker_size = kwargs.get("s", SCATTER_MARKER_SIZE_SMALL)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, marker=marker, s=marker_size)  # type: ignore
    fig.tight_layout()
    return fig, ax


def plot_bounded_scatter(
    x: list | pd.Series,
    y: list | pd.Series,
    xlabel: str,
    ylabel: str,
    xbounds: tuple[float, float] | None = None,
    ybounds: tuple[float, float] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Returns a generic scatter plot"""
    fig, ax = plot_scatter(x, y, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xbounds is None:
        xbounds = _get_range_with_margins((min(x), max(x)))
    if ybounds is None:
        ybounds = _get_range_with_margins((min(y), max(y)))
    ax.set_xlim(xbounds)
    ax.set_ylim(*ybounds)

    return fig, ax


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
    plot_height: int = FIG_DIM_Y,
) -> int:
    return x_resolution * plot_width // plot_height


def _get_range_with_margins(
    value: tuple[float, float], margin_multiplier: float = 0.01
) -> tuple[float, float]:
    start = min(value)
    end = max(value)
    width = end - start
    if width == 0:
        margin = start * margin_multiplier
    else:
        margin = width * margin_multiplier
    return (start - margin, end + margin)


K = TypeVar("K")
V = TypeVar("V")


def _popget(dictionary: dict[K, V], key: K, default: V) -> V:
    # acts like dict.get, but pops key out of dict if exists
    try:
        return dictionary.pop(key)
    except KeyError:
        return default
