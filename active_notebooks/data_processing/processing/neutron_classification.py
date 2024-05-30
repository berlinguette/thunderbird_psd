from typing import Callable, NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from data_processing.dataframe_validation import DataframeColumn
from data_processing.processing.processing_configs import \
    DEFAULT_LOWER_ENERGY_BOUND
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from data_processing.dataframe_validation import get_df_col


V = TypeVar('V', float, pd.Series[float], NDArray)
WindowBorderFunction = Callable[[V], V]
class WindowBorders(NamedTuple):
    left: float | None
    right: float | None
    bottom: WindowBorderFunction | None
    top: WindowBorderFunction | None


def generate_nasa_neutron_window(
    slice_fit_df: pd.DataFrame,
    slice_xs: np.ndarray,
    window_offset: float = 0.2,
    sigma: float = 5,
    lower_energy_bound: float = 0.1966
) -> WindowBorders:
    # Define lower and upper bounds (neutron_lb_fit, neutron_ub_fit)
    neutron_lb = savgol_filter(
        slice_fit_df["mu1"] + sigma * slice_fit_df["sigma1"], 
        window_length=21, 
        polyorder=3)  # reduce noise
    neutron_lb_fit: WindowBorderFunction = interp1d(
        slice_xs, 
        neutron_lb, 
        fill_value=(neutron_lb[0], neutron_lb[-1]),  # type: ignore
        bounds_error=False)  # now it's a function!

    def neutron_ub_fit(x: pd.Series[float]) -> pd.Series[float]:
        # upper bound is just a fixed PSD offset from lower bound
        # as observed in NASA paper graphs
        return neutron_lb_fit(x) + window_offset
    
    return WindowBorders(left=lower_energy_bound, bottom=neutron_lb_fit, top=neutron_ub_fit, right=None)


def classify(
    psd_report: pd.DataFrame, 
    borders: WindowBorders,
    label: DataframeColumn = DataframeColumn.NEUTRON_CLASS,
    # window_adj_offset: int = 0
) -> pd.DataFrame:
    """Classify signals as neutron or non-neutron for a given count window
    Classification creates a new column of boolean values, where True indicates a neutron classified signal.
    
    Parameters
    ----------
    psd_report: DataFrame
        DataFrame containing signal PSD data. 
        It must have the "tail / total" (for PSD value) and "CALIB_ENERGY" (for calibrated energy) columns.
    borders: WindowBorders
        The borders for the neutron window
    label: str
        Column label to use for signal classification results
        
    Returns
    -------
    psd_report: DataFrame
        Original DataFrame with new column for neutron classification
    """
    energy_col = get_df_col(psd_report, DataframeColumn.CALIB_ENERGY)
    psd_col = get_df_col(psd_report, DataframeColumn.PSD)
    
    bottom_border_fn = borders.bottom
    top_border_fn = borders.top
    bottom_border = bottom_border_fn(energy_col.astype(float)) if bottom_border_fn is not None else None
    top_border = top_border_fn(energy_col.astype(float)) if top_border_fn is not None else None
    within_psd_bounds = _is_within_bounds(psd_col, bottom_border, top_border)
    
    left_border = borders.left
    right_border = borders.right
    within_energy_bounds = _is_within_bounds(energy_col, left_border, right_border)
    
    psd_report[label.value] = within_psd_bounds & within_energy_bounds
    
    return psd_report

def _is_within_bounds(
    value_col: pd.Series, 
    lower_bound: float|pd.Series|None, 
    upper_bound: float|pd.Series|None
) -> pd.Series[bool]:
    if lower_bound is not None:
        if upper_bound is not None:
            within_bounds = value_col.between(lower_bound, upper_bound)
        else:
            within_bounds = value_col >= lower_bound
    else:
        if upper_bound is not None:
            within_bounds = value_col <= upper_bound
        else:
            within_bounds = value_col == value_col
    return within_bounds
