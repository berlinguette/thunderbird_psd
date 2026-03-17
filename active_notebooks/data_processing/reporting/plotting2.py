import pandas as pd
from data_processing.reporting.plotter import SinglePlot
from data_processing.reporting.plot_config_fns import get_tail_vs_total_config2, get_psd_histogram_config
from data_processing.reporting.plot_configs import *
from data_processing.reporting.plot_data_functions import get_tail_vs_total_data, get_psd_histogram_data
from data_processing.reporting.plot_functions import graph_tail_vs_total, graph_psd_histogram
from data_processing.reporting.figure_update_fns import make_supertitle_adder
from data_processing.reporting.axes_update_fns import make_label_setter, make_limit_setter, make_ticksize_setter
# from data_processing.dataframe_validation import get_df_col, DetectorDataframeColumn

def plot_tail_vs_total(df: pd.DataFrame, experiment_display_name: str, **kwargs) -> tuple[Figure, Axes]:
    # plot_kwargs, figsize = get_tail_vs_total_config(df, **kwargs)
    config = get_tail_vs_total_config2(df, **kwargs)
    data = get_tail_vs_total_data(df)
    supertitle = f"Tail vs Total - {experiment_display_name}"
    add_supertitle = make_supertitle_adder(supertitle, font_size=SUPTITLE_FONT_SIZE)

    plotter = (
        SinglePlot(config.figsize)
        .plot_data(graph_tail_vs_total, data, config)
        .update_figure(add_supertitle)
    )
    
    return plotter.plot_objects


def plot_psd_histogram(
    df: pd.DataFrame,
    energy_column: EnergyColumn = DetectorDataframeColumn.CALIB_ENERGY,
    cmap: str | Colormap = "gnuplot",
    colorbar: bool = False,
    energy_start_zero: bool = False,
    **kwargs,
) -> tuple[Figure, Axes, StrAnyDict]:
    data = get_psd_histogram_data(df, energy_column)
    plot_kwargs, figsize = get_psd_histogram_config(data, cmap, **kwargs)
    plot_range = plot_kwargs["range"]
    x_range, y_range = plot_range
    set_limits = make_limit_setter(x_range, y_range)
    axis_font_size = plot_kwargs["axis_font_size"]
    set_axes_titles = make_label_setter()
    axis_tick_font_size = plot_kwargs["axis_tick_font_size"]
    axis_fontsize = make_ticksize_setter(axis_font_size)

    plotter = (
        SinglePlot(figsize)
        .plot_data(graph_psd_histogram, data, plot_kwargs)
        .update_axes()

    )
