from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from data_processing.types import GraphData, StrAnyDict
from data_processing.reporting.plot_configs import *
from data_processing.reporting.plot_config_models import TailVsTotalConfig, PsdHistogramConfig

# Callable[[Axes, GraphData, StrAnyDict], tuple[Axes, StrAnyDict]]

def graph_tail_vs_total(
    ax: Axes,
    data: GraphData,
    config: TailVsTotalConfig
) -> tuple[Axes, StrAnyDict]:
    # x_resolution = graph_kwargs.get("x_resolution", HISTOGRAM_RES)
    # y_resolution = graph_kwargs.get("y_resolution", HISTOGRAM_RES)
    # max_energy = graph_kwargs.get("max_energy", 5)
    range = [[0, config.max_energy + 0.05], [0, 0.50]]
    # cmap = graph_kwargs.get("cmap", colormaps["gnuplot"])  # type: ignore

    data_x = data["x"]
    data_y = data["y"]
    dataset_size = data_x.shape[0]

    H, xedges, yedges, image = ax.hist2d(
        data_x,
        data_y,
        bins=config.bins,
        norm=LogNorm(),
        range=range,
        cmap=config.cmap,
    )

    return_data = {
        "H": H,
        "xedges": xedges,
        "yedges": yedges,
        "image": image,
        "dataset_size": dataset_size
    }
    return ax, return_data


def graph_psd_histogram(
    ax: Axes,
    data: GraphData,
    # graph_kwargs: StrAnyDict
    config: PsdHistogramConfig
) -> tuple[Axes, StrAnyDict]:
    # x_resolution = graph_kwargs.get("x_resolution", HISTOGRAM_RES)
    # y_resolution = graph_kwargs.get("y_resolution", HISTOGRAM_RES)
    # energy_start_zero = graph_kwargs.get("energy_start_zero", False)
    # cmap = graph_kwargs.get("cmap", colormaps["gnuplot"])  # type: ignore
    # cmin = graph_kwargs.get("cmin")
    # weights = graph_kwargs.get("weights", None)
    # vmin = graph_kwargs.get("vmin")
    # vmax = graph_kwargs.get("vmax")

    data_x = data["x"]
    data_y = data["y"]
    dataset_size = data_x.shape[0]

    min_energy, max_energy = _get_range_with_margins((data_x.min(), data_x.max()))
    # if energy_start_zero:
    if config.energy_start_zero:
        min_energy = 0
    min_psd, max_psd = _get_range_with_margins((data_y.min(), data_y.max()))

    H, xedges, yedges, image = ax.hist2d(
        data_x,
        data_y,
        # bins=(x_resolution, y_resolution),
        bins=config.bins,
        range=[[min_energy, max_energy], [min_psd, max_psd]],
        cmap=config.cmap,
        cmin=config.cmin,
        weights=config.weights,
        vmin=config.vmin,
        vmax=config.vmax,
    )

    return_data = {
        "H": H,
        "xedges": xedges,
        "yedges": yedges,
        "image": image,
        "dataset_size": dataset_size
    }
    return ax, return_data


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