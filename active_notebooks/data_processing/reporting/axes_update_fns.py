from data_processing.reporting.plot_configs import *
from data_processing.types import AxesUpdateFunction, WindowBorders
from matplotlib.axes import Axes
import numpy as np


def make_title_adder(
        title: str, font_size: float = TITLE_FONT_SIZE
) -> AxesUpdateFunction:
    def add_title_to_axes(ax: Axes) -> Axes:
        ax.set_title(title, fontsize=font_size)
        return ax
    
    return add_title_to_axes


def make_fit_window_adder(
        borders: WindowBorders,
        graph_x_limits: tuple[float, float],
        graph_y_limits: tuple[float, float],
        line_color_code = "r",
        line_style = "--"
) -> AxesUpdateFunction:
    def add_fit_window_to_plot(axes: Axes) -> Axes:
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
    
    return add_fit_window_to_plot


def make_limit_setter(
    xlim: tuple[float | None, float | None] | None,
    ylim: tuple[float | None, float | None] | None,
) -> AxesUpdateFunction:
    def set_plot_limits(axes: Axes) -> Axes:
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        return axes
    
    return set_plot_limits


def make_label_setter(
    xlabel: str, ylabel: str, fontsize: int = AXIS_FONT_SIZE
) -> AxesUpdateFunction:
    def set_plot_axes_labels(axes: Axes) -> Axes:
        axes.set_xlabel(xlabel, fontsize=fontsize)
        axes.set_ylabel(ylabel, fontsize=fontsize)
        return axes
    
    return set_plot_axes_labels


def make_ticksize_setter(
    tick_font_size: int = AXIS_TICK_FONT_SIZE
) -> AxesUpdateFunction:
    def set_tick_font_size(axes: Axes) -> Axes:
        axes.tick_params(axis="both", which="major", labelsize=tick_font_size)
        axes.tick_params(axis="both", which="minor", labelsize=tick_font_size)
        return axes
    
    return set_tick_font_size
