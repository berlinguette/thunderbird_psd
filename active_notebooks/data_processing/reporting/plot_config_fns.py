import pandas as pd
from data_processing.types import DictKey, DictValue, DictKV, StrAnyDict
from data_processing.reporting.plot_configs import *
from data_processing.dataframe_validation import DetectorDataframeColumn, get_df_col
from data_processing.reporting.plot_config_models import TailVsTotalConfig


def get_tail_vs_total_config(df: pd.DataFrame, **kwargs) -> tuple[StrAnyDict, tuple[float, float]]:
    figsize = popget(kwargs, "figsize", (FIG_DIM_X, FIG_DIM_Y))
    x_resolution = popget(kwargs, "x_resolution", HISTOGRAM_RES)
    fig_width, fig_height = figsize
    y_resolution = get_histogram_y_resolution(x_resolution, fig_width, fig_height)
    max_energy = get_df_col(df, DetectorDataframeColumn.ENERGY).max()
    plot_kwargs = {
        "x_resolution": x_resolution,
        "y_resolution": y_resolution,
        "max_energy": max_energy,
        **kwargs
    }
    return plot_kwargs, figsize


def get_tail_vs_total_config2(df: pd.DataFrame, **kwargs) -> TailVsTotalConfig:
    width, height = popget(kwargs, "figsize", (FIG_DIM_X, FIG_DIM_Y))
    kwargs["width"] = width
    kwargs["height"] = height
    max_energy = get_df_col(df, DetectorDataframeColumn.ENERGY).max()
    kwargs["max_energy"] = max_energy
    return TailVsTotalConfig.model_validate(kwargs)


def get_psd_histogram_config():
    pass  # STUB


def popget(dictionary: DictKV, key: DictKey, default: DictValue) -> DictValue:
    try:
        return dictionary.pop(key)
    except KeyError:
        return default


def kwarg_transfer(src_kwargs: DictKV, dest_kwargs: DictKV, key: DictKey) -> tuple[DictKV, DictKV]:
    if key in src_kwargs:
        dest_kwargs[key] = src_kwargs.pop(key)
    return src_kwargs, dest_kwargs


def get_histogram_y_resolution(
    x_resolution: int = HISTOGRAM_RES,
    plot_width: int = FIG_DIM_X,
    plot_height: int = FIG_DIM_Y,
) -> int:
    return x_resolution * plot_width // plot_height
