from math import ceil

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Colormap, Normalize, LogNorm
from matplotlib.collections import QuadMesh
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from data_processing.dataframe_validation import DetectorDataframeColumn, get_df_col, EnergyColumn
from data_processing.processing.figure_of_merit import FOM, gaussian
from data_processing.reporting.plot_configs import *
from data_processing.types import (
    AxesMatrix,
    DictKey,
    DictValue,
    GraphData,
    GraphingFunction,
    GraphingFunction2,
    StrAnyDict,
    WindowBorders,
    BorderSettings,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_single(
    graphing_function: GraphingFunction,
    data: GraphData,
    figsize: tuple[float, float] = (FIG_DIM_X, FIG_DIM_Y),
    supertitle: str | None = None,
    supertitle_font_size: float = SUPTITLE_FONT_SIZE,
    plot_kwargs: StrAnyDict | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize, **kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}
    ax = graphing_function(fig, ax, data, plot_kwargs)

    if supertitle is not None:
        fig.suptitle(supertitle, fontsize=supertitle_font_size)

    return fig, ax


def plot_single2(
    graphing_function: GraphingFunction2,
    data: GraphData,
    figsize: tuple[float, float] = (FIG_DIM_X, FIG_DIM_Y),
    plot_kwargs: StrAnyDict | None = None,
    **kwargs,
) -> tuple[Figure, Axes, StrAnyDict]:
    fig, ax = plt.subplots(figsize=figsize, **kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}
    ax, return_data = graphing_function(ax, data, plot_kwargs)

    return fig, ax, return_data


def plot_many(
    graphing_function: GraphingFunction,
    data: list[GraphData],
    figsize: tuple[float, float] = (FIG_DIM_X, FIG_DIM_Y),
    max_cols: int = SUBPLOTS_MAX_COLS,
    supertitle: str | None = None,
    supertitle_font_size: float = SUPTITLE_FONT_SIZE,
    plot_kwargs: list[StrAnyDict] | None = None,
    **kwargs,
) -> tuple[Figure, AxesMatrix]:
    n_plots = len(data)
    n_cols = min(max_cols, n_plots)
    n_rows = ceil(n_plots / n_cols)

    subplots_result: tuple[Figure, AxesMatrix] = plt.subplots(
        figsize=figsize, ncols=n_cols, nrows=n_rows, **kwargs
    )
    fig, axs = subplots_result

    if plot_kwargs is None:
        plot_kwargs = [{} for _ in data]
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


def plot_many2(
    graphing_function: GraphingFunction2,
    data: list[GraphData],
    figsize: tuple[float, float] = (FIG_DIM_X, FIG_DIM_Y),
    max_cols: int = SUBPLOTS_MAX_COLS,
    plot_kwargs: list[StrAnyDict] | None = None,
    **kwargs,
) -> tuple[Figure, AxesMatrix, list[StrAnyDict]]:
    n_plots = len(data)
    n_cols = min(max_cols, n_plots)
    n_rows = ceil(n_plots / n_cols)

    subplots_result: tuple[Figure, AxesMatrix] = plt.subplots(
        figsize=figsize, ncols=n_cols, nrows=n_rows, **kwargs
    )
    fig, axs = subplots_result

    return_data_list: list[StrAnyDict] = []
    if plot_kwargs is None:
        plot_kwargs = [{} for _ in data]
    for i, subplot_data in enumerate(data):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row][col]
        subplot_kwargs = plot_kwargs[i]

        ax, return_data = graphing_function(ax, subplot_data, subplot_kwargs)
        axs[row][col] = ax
        return_data_list.append(return_data)

    return fig, axs, return_data_list


# Graphing Functions
def graph_tail_vs_total(
    fig: Figure, ax: Axes, data: GraphData, graph_kwargs: StrAnyDict
) -> Axes:
    ax, _ = graph_tail_vs_total2(ax, data, graph_kwargs)
    return ax


def graph_tail_vs_total2(
    ax: Axes, data: GraphData, graph_kwargs: StrAnyDict
) -> tuple[Axes, StrAnyDict]:
    x_resolution = graph_kwargs.get("x_resolution", HISTOGRAM_RES)
    y_resolution = graph_kwargs.get("y_resolution", HISTOGRAM_RES)
    max_energy = graph_kwargs.get("max_energy", 5)
    cmap = graph_kwargs.get("cmap", colormaps["gnuplot"])  # type: ignore
    title_font_size = graph_kwargs.get("title_font_size", TITLE_FONT_SIZE)
    axis_font_size = graph_kwargs.get("axis_font_size", AXIS_FONT_SIZE)
    axis_tick_font_size = graph_kwargs.get("axis_tick_font_size", AXIS_TICK_FONT_SIZE)

    data_x = data["x"]
    data_y = data["y"]
    dataset_size = data_x.shape[0]

    H, xedges, yedges, image = ax.hist2d(
        data_x,
        data_y,
        bins=(x_resolution, y_resolution),
        norm=LogNorm(),
        range=[[0, max_energy + 0.05], [0, 0.50]],
        cmap=cmap,
    )

    ax.set_title(f"Event count = {dataset_size:,d}", fontsize=title_font_size)
    ax.set_xlabel("Tail", fontsize=axis_font_size)
    ax.set_ylabel("Total", fontsize=axis_font_size)
    ax.tick_params(axis="both", which="major", labelsize=axis_tick_font_size)
    ax.tick_params(axis="both", which="minor", labelsize=axis_tick_font_size)

    return ax, {"H": H, "xedges": xedges, "yedges": yedges, "image": image}


def graph_psd_histogram(
    fig: Figure, ax: Axes, data: GraphData, graph_kwargs: StrAnyDict
) -> Axes:
    x_resolution = graph_kwargs.get("x_resolution", HISTOGRAM_RES)
    y_resolution = graph_kwargs.get("y_resolution", HISTOGRAM_RES)
    energy_start_zero = graph_kwargs.get("energy_start_zero", False)
    cmap = graph_kwargs.get("cmap", colormaps["gnuplot"])  # type: ignore
    colorbar = graph_kwargs.get("colorbar", False)
    axis_font_size = graph_kwargs.get("axis_font_size", AXIS_FONT_SIZE)
    axis_tick_font_size = graph_kwargs.get("axis_tick_font_size", AXIS_TICK_FONT_SIZE)
    cmin = graph_kwargs.get("cmin")
    weights = graph_kwargs.get("weights", None)
    vmin = graph_kwargs.get("vmin")
    vmax = graph_kwargs.get("vmax")

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
        cmin=cmin,
        weights=weights,
        vmin=vmin,
        vmax=vmax,
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


def graph_psd_histogram2(
    ax: Axes, data: GraphData, graph_kwargs: StrAnyDict
) -> tuple[Axes, StrAnyDict]:
    x_resolution = graph_kwargs.get("x_resolution", HISTOGRAM_RES)
    y_resolution = graph_kwargs.get("y_resolution", HISTOGRAM_RES)
    energy_start_zero = graph_kwargs.get("energy_start_zero", False)
    cmap = graph_kwargs.get("cmap", colormaps["gnuplot"])  # type: ignore
    axis_font_size = graph_kwargs.get("axis_font_size", AXIS_FONT_SIZE)
    axis_tick_font_size = graph_kwargs.get("axis_tick_font_size", AXIS_TICK_FONT_SIZE)
    cmin = graph_kwargs.get("cmin")
    weights = graph_kwargs.get("weights", None)
    vmin = graph_kwargs.get("vmin")
    vmax = graph_kwargs.get("vmax")

    data_x = data["x"]
    data_y = data["y"]

    min_energy, max_energy = _get_range_with_margins((data_x.min(), data_x.max()))
    if energy_start_zero:
        min_energy = 0
    min_psd, max_psd = _get_range_with_margins((data_y.min(), data_y.max()))

    H, xedges, yedges, image = ax.hist2d(
        data_x,
        data_y,
        bins=(x_resolution, y_resolution),
        range=[[min_energy, max_energy], [min_psd, max_psd]],
        cmap=cmap,
        cmin=cmin,
        weights=weights,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_ylim(min_psd, max_psd)
    ax.set_xlim(min_energy, max_energy)
    ax.set_xlabel("Energy (MeVee)", fontsize=axis_font_size)
    ax.set_ylabel("PSD", fontsize=axis_font_size)
    ax.tick_params(axis="both", which="major", labelsize=axis_tick_font_size)
    ax.tick_params(axis="both", which="minor", labelsize=axis_tick_font_size)

    return ax, {"H": H, "xedges": xedges, "yedges": yedges, "image": image}


# Figure/Axes Modification Functions
def add_supertitle_to_figure(
    fig: Figure, supertitle: str, font_size: float = SUPTITLE_FONT_SIZE
) -> Figure:
    fig.suptitle(supertitle, fontsize=font_size)
    return fig


def add_colorbar_to_axes(
    fig: Figure, ax: Axes, image: QuadMesh, **kwargs
) -> Figure:
    fig.colorbar(image, ax=ax, **kwargs)
    return fig


def add_fit_window_to_plot(
    axes: Axes,
    borders: WindowBorders,
    graph_x_limits: tuple[float, float],
    graph_y_limits: tuple[float, float],
    line_color_code = "r",
    line_style = "--"
) -> Axes:
    plot_style = f"{line_color_code}{line_style}"
    left_border = borders.left
    right_border = borders.right
    bottom_border_fn = borders.bottom
    top_border_fn = borders.top

    min_energy = left_border if left_border is not None else graph_x_limits[0]
    max_energy = right_border if right_border is not None else graph_x_limits[1]

    energy_space = np.linspace(min_energy, max_energy, 200)
    if bottom_border_fn is not None:
        axes.plot(energy_space, bottom_border_fn(energy_space), plot_style)
    if top_border_fn is not None:
        axes.plot(energy_space, top_border_fn(energy_space), plot_style)
    # ax.vlines(all_slice_xs[0],
    #           neutron_lb_fit(all_slice_xs[0]),
    #           neutron_ub_fit(all_slice_xs[0]),
    #           'r', ls='--')
    if left_border is not None:
        line_bottom = (
            bottom_border_fn(left_border)
            if bottom_border_fn is not None
            else graph_y_limits[0]
        )
        line_top = (
            top_border_fn(left_border)
            if top_border_fn is not None
            else graph_y_limits[1]
        )
        axes.vlines(left_border, line_bottom, line_top, line_color_code, ls=line_style)  # type: ignore
    if right_border is not None:
        line_bottom = (
            bottom_border_fn(right_border)
            if bottom_border_fn is not None
            else graph_y_limits[0]
        )
        line_top = (
            top_border_fn(right_border)
            if top_border_fn is not None
            else graph_y_limits[1]
        )
        axes.vlines(right_border, line_bottom, line_top, line_color_code, ls=line_style)  # type: ignore
    return axes


# All In One Plot Functions
def plot_tail_vs_total(
    df: pd.DataFrame, experiment_display_name: str, **kwargs
) -> tuple[Figure, Axes]:
    figsize = _popget(kwargs, "figsize", (FIG_DIM_X, FIG_DIM_Y))
    x_resolution = _popget(kwargs, "x_resolution", HISTOGRAM_RES)

    y_resolution = _get_histogram_y_resolution(
        x_resolution=x_resolution, plot_width=figsize[0], plot_height=figsize[1]
    )
    max_energy = get_df_col(df, DetectorDataframeColumn.ENERGY).max()
    plot_kwargs = {
        "x_resolution": x_resolution,
        "y_resolution": y_resolution,
        "max_energy": max_energy,
        **kwargs,
    }

    energy_col = get_df_col(df, DetectorDataframeColumn.ENERGY)
    short_col = get_df_col(df, DetectorDataframeColumn.ENERGYSHORT)
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
    energy_column: EnergyColumn = DetectorDataframeColumn.CALIB_ENERGY,
    cmap: str | Colormap = "gnuplot",
    colorbar: bool = False,
    energy_start_zero: bool = False,
    **kwargs,
) -> tuple[Figure, Axes]:
    figsize = _popget(kwargs, "figsize", (FIG_DIM_X, FIG_DIM_Y))
    x_resolution = _popget(kwargs, "x_resolution", HISTOGRAM_RES)
    y_resolution = _get_histogram_y_resolution(
        x_resolution=x_resolution, plot_width=figsize[0], plot_height=figsize[1]
    )
    if isinstance(cmap, str):
        cmap = colormaps[cmap]
    plot_kwargs = {
        "x_resolution": x_resolution,
        "y_resolution": y_resolution,
        "energy_start_zero": energy_start_zero,
        "cmap": cmap,
        "colorbar": colorbar,
        **kwargs,
    }

    energy_col = get_df_col(df, energy_column)
    psd_col = get_df_col(df, DetectorDataframeColumn.PSD)
    data: GraphData = {"x": energy_col, "y": psd_col}

    fig, ax = plot_single(
        graph_psd_histogram, data, figsize=figsize, plot_kwargs=plot_kwargs
    )

    return fig, ax


def plot_psd_histogram2(
    df: pd.DataFrame,
    energy_column: EnergyColumn = DetectorDataframeColumn.CALIB_ENERGY,
    cmap: str | Colormap = "gnuplot",
    colorbar: bool = False,
    energy_start_zero: bool = False,
    **kwargs,
) -> tuple[Figure, Axes, StrAnyDict]:
    figsize = _popget(kwargs, "figsize", (FIG_DIM_X, FIG_DIM_Y))
    x_resolution = _popget(kwargs, "x_resolution", HISTOGRAM_RES)
    y_resolution = _get_histogram_y_resolution(
        x_resolution=x_resolution, plot_width=figsize[0], plot_height=figsize[1]
    )
    if isinstance(cmap, str):
        cmap = colormaps[cmap]
    plot_kwargs = {
        "x_resolution": x_resolution,
        "y_resolution": y_resolution,
        "energy_start_zero": energy_start_zero,
        "cmap": cmap,
        "colorbar": colorbar,
        **kwargs,
    }

    energy_col = get_df_col(df, energy_column)
    psd_col = get_df_col(df, DetectorDataframeColumn.PSD)
    data: GraphData = {"x": energy_col, "y": psd_col}

    fig, ax, return_data = plot_single2(
        graph_psd_histogram2, data, figsize=figsize, plot_kwargs=plot_kwargs
    )
    
    if colorbar:
        image = return_data.get("image")
        if image is not None and isinstance(image, QuadMesh):
            fig.colorbar(image, ax=ax)
        else:
            raise ValueError("No image found for colorbar addition.")

    return fig, ax, return_data


def plot_classification(
    df: pd.DataFrame,
    borders: WindowBorders,
    experiment_display_name: str,
    class_col_name: DetectorDataframeColumn,
    energy_col_name: EnergyColumn,
    cmap: str | Colormap = "RdBu_r",
    count_limit: int = 5,
    legend: bool = True,
    titles: bool = True,
    **kwargs,
) -> tuple[Figure, Axes]:
    fit_window_kwargs: StrAnyDict = {}
    kwargs, fit_window_kwargs = _kwarg_transfer(
        kwargs, fit_window_kwargs, "line_color_code"
    )
    kwargs, fit_window_kwargs = _kwarg_transfer(
        kwargs, fit_window_kwargs, "line_style"
    )
    # y_resolution = _get_histogram_y_resolution()
    # max_energy = get_df_col(df, DataframeColumn.CALIB_ENERGY).max()

    # fig, ax = plt.subplots(figsize=(FIG_DIM_X, FIG_DIM_Y))

    class_col = get_df_col(df, class_col_name)
    # energy_col = get_df_col(df, DataframeColumn.CALIB_ENERGY)
    # psd_col = get_df_col(df, DataframeColumn.PSD)

    g_vs_n = class_col.map({True: 1, False: -1})
    if isinstance(cmap, str):
        cmap = colormaps[cmap]

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
        energy_column=energy_col_name,
        cmap=cmap,
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
    ax = add_fit_window_to_plot(ax, borders, ax.get_xlim(), ax.get_ylim(), **fit_window_kwargs)

    # ax.set_ylim(0, 0.5)
    # ax.set_xlim(0, max_energy + .05)
    if titles:
        n_neutrons = df[df[class_col_name.value]].shape[0]
        fig.suptitle(
            f"Neutron Classification: {experiment_display_name}",
            fontsize=SUPTITLE_FONT_SIZE,
        )
        ax.set_title(f"Neutron count = {n_neutrons}", fontsize=TITLE_FONT_SIZE)
    # ax.set_xlabel("Energy (MeVee)", fontsize=AXIS_FONT_SIZE)
    # ax.set_ylabel("PSD", fontsize=AXIS_FONT_SIZE)
    # ax.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONT_SIZE)
    # ax.tick_params(axis='both', which='minor', labelsize=AXIS_TICK_FONT_SIZE)
    if legend:
        event_colors = [
            Patch(facecolor=cmap(1.0)),  # type: ignore
            Patch(facecolor=cmap(0.0)),  # type: ignore
        ]  # type: ignore
        ax.legend(event_colors, ["Neutrons", "Gamma"])

    return fig, ax


def plot_multiple_classification(
    df: pd.DataFrame,
    border_settings: list[BorderSettings],
    energy_col_name: EnergyColumn,
    cmap: str | Colormap = "Greys",
    normalizer: Normalize | None = None,
    count_limit: int = 5,
    legend: bool = True,
    **kwargs,
):
    figsize: tuple[int, int] = _popget(kwargs, "figsize", (FIG_DIM_X, FIG_DIM_Y))
    x_resolution: int = _popget(kwargs, "x_resolution", HISTOGRAM_RES)
    energy_start_zero: bool = kwargs.get("energy_start_zero", False)
    axis_font_size: int = kwargs.get("axis_font_size", AXIS_FONT_SIZE)
    axis_tick_font_size: int = kwargs.get("axis_tick_font_size", AXIS_TICK_FONT_SIZE)
    cmin: int = kwargs.get("cmin", 0)
    vmin: int = kwargs.get("vmin", 0)
    norm: Normalize = normalizer if normalizer is not None else Normalize(vmin=vmin, vmax=count_limit)
    random_cmaps: list[str] = [
        "Purples", "Blues", "Greens", "Oranges", "Reds"
    ]
    random_colors: list[str] = [
        "m", "b", "g", "y", "r"
    ]
    
    if not isinstance(x_resolution, int) or x_resolution <= 0:
        raise ValueError("x_resolution must be a positive integer")
    if not isinstance(energy_start_zero, bool):
        raise ValueError("energy_start_zero must be a boolean value")
    if not isinstance(axis_font_size, int) or axis_font_size <= 0:
        raise ValueError("axis_font_size must be a positive integer")
    if not isinstance(axis_tick_font_size, int) or axis_tick_font_size <= 0:
        raise ValueError("axis_tick_font_size must be a positive integer")
    if not isinstance(cmin, int) or cmin < 0:
        raise ValueError("cmin must be a non-negative integer")
    if not isinstance(vmin, int) or vmin < 0:
        raise ValueError("vmin must be a non-negative integer")
    
    y_resolution = _get_histogram_y_resolution(
        x_resolution=x_resolution, plot_width=figsize[0], plot_height=figsize[1]
    )
    if isinstance(cmap, str):
        base_cmap: Colormap = colormaps[cmap]
    else:
        base_cmap = cmap
        
    x_data = df[energy_col_name.value]
    y_data = df[DetectorDataframeColumn.PSD.value]
    
    x_min, x_max = _get_range_with_margins((x_data.min(), x_data.max()))
    if energy_start_zero:
        x_min = 0
    y_min, y_max = _get_range_with_margins((y_data.min(), y_data.max()))
    
    H, xedges, yedges = np.histogram2d(
        x_data,
        y_data,
        bins=(x_resolution, y_resolution),
        range=[[x_min, x_max], [y_min, y_max]],
        
    )
    
    h_norm = norm(H)
    base_colors = base_cmap(h_norm)
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    X, Y = np.ravel(Xc), np.ravel(Yc)
    
    border_counts: list[int] = []
    border_color_patches: list[Patch] = []
    border_labels: list[str] = []
    for i, bs in enumerate(border_settings):
        borders, border_label, border_config = bs
        _cmap: str | Colormap = border_config.get("cmap", random_cmaps[i % len(random_cmaps)])
        
        if isinstance(_cmap, str):
            border_cmap: Colormap = colormaps[_cmap]
        elif isinstance(_cmap, Colormap):
            border_cmap = _cmap
        else:
            raise ValueError("Borders cmap must be a string or Colormap instance")
        
        mask = _get_borders_mask(X, Y, borders, Xc.shape)
        border_colors = border_cmap(h_norm)
        base_colors[mask] = border_colors[mask]  # type: ignore
        
        mask_counts = H[mask].sum()
        
        border_counts.append(int(mask_counts))
        border_color_patches.append(Patch(facecolor=border_cmap(0.6)))
        border_labels.append(border_label)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(
        Xc, Yc, base_colors, shading="auto"
    )
    
    for i, bs in enumerate(border_settings):
        borders, _, border_config = bs
        line_color_code = border_config.get("line_color_code", random_colors[i % len(random_colors)])
        line_style = border_config.get("line_style", "-")
        
        ax = add_fit_window_to_plot(
            ax, borders, ax.get_xlim(), ax.get_ylim(),
            line_color_code=line_color_code,
            line_style=line_style
        )
        
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Energy (MeVee)", fontsize=axis_font_size)
    ax.set_ylabel("PSD", fontsize=axis_font_size)
    ax.tick_params(axis="both", which="major", labelsize=axis_tick_font_size)
    ax.tick_params(axis="both", which="minor", labelsize=axis_tick_font_size)
        
    if legend:
        ax.legend(border_color_patches, border_labels)
        
    return fig, ax, border_counts
                


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

    return fig, ax


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


def _popget(
    dictionary: dict[DictKey, DictValue], key: DictKey, default: DictValue
) -> DictValue:
    # acts like dict.get, but pops key out of dict if exists
    try:
        return dictionary.pop(key)
    except KeyError:
        return default
    
def _kwarg_transfer(src_kwargs: StrAnyDict, dest_kwargs: StrAnyDict, key: str):
    if key in src_kwargs:
        dest_kwargs[key] = src_kwargs.pop(key)
    return src_kwargs, dest_kwargs


def _get_borders_mask(X: np.ndarray, Y: np.ndarray, borders: WindowBorders, shape: tuple[int, int]) -> np.ndarray:
    left_border = borders.left
    right_border = borders.right
    bottom_border_fn = borders.bottom
    top_border_fn = borders.top

    mask = np.ones(X.shape, dtype=bool)

    if left_border is not None:
        mask &= X >= left_border
    if right_border is not None:
        mask &= X <= right_border
    if bottom_border_fn is not None:
        mask &= Y >= bottom_border_fn(X)
    if top_border_fn is not None:
        mask &= Y <= top_border_fn(X)
        
    mask = mask.reshape(shape)
    return mask
