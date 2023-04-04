from typing import Callable

import numpy as np
import pandas as pd
from data_processing.dataframe_validation import DataframeColumn
from data_processing.processing.processing_configs import \
    DEFAULT_LOWER_ENERGY_BOUND
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from data_processing.dataframe_validation import get_df_col

def generate_nasa_neutron_window(
    slice_fit_df: pd.DataFrame,
    slice_xs: np.ndarray,
    window_offset: float = 0.2,
    sigma: float = 5
) -> tuple[Callable[[float], float], 
           Callable[[float], float]]:
    # Define lower and upper bounds (neutron_lb_fit, neutron_ub_fit)
    neutron_lb = savgol_filter(
        slice_fit_df["mu1"] + sigma * slice_fit_df["sigma1"], 
        window_length=21, 
        polyorder=3)  # reduce noise
    neutron_lb_fit: Callable[[float], float] = interp1d(
        slice_xs, 
        neutron_lb, 
        fill_value=(neutron_lb[0], neutron_lb[-1]),  # type: ignore
        bounds_error=False)  # now it's a function!

    def neutron_ub_fit(x: float) -> float:
        # upper bound is just a fixed PSD offset from lower bound
        # as observed in NASA paper graphs
        return neutron_lb_fit(x) + window_offset
    
    return neutron_lb_fit, neutron_ub_fit


def classify(
    psd_report: pd.DataFrame, 
    lb_fit_fn: Callable[[np.ndarray], np.ndarray], 
    ub_fit_fn: Callable[[np.ndarray], np.ndarray], 
    label: DataframeColumn = DataframeColumn.NEUTRON_CLASS,
    le_cutoff: float = DEFAULT_LOWER_ENERGY_BOUND,
    window_adj_offset: int = 0
) -> pd.DataFrame:
    """Classify signals as neutron or non-neutron for a given count window
    Classification creates a new column of boolean values, where True indicates a neutron classified signal.
    
    Parameters
    ----------
    psd_report: DataFrame
        DataFrame containing signal PSD data. 
        It must have the "tail / total" (for PSD value) and "CALIB_ENERGY" (for calibrated energy) columns.
    lb_fit_fn: Callable[[ndarray], ndarray]
        A function describing the window's lower bounds
    ub_fit_fn: Callable[[ndarray], ndarray]
        A function describing the window's upper bounds
    label: str
        Column label to use for signal classification results
    le_cutoff: float, default L0 (previously determined lower energy cutoff)
        Lower energy cutoff (AKA the left boundary of the neutron window)
        
    Returns
    -------
    psd_report: DataFrame
        Original DataFrame with new column for neutron classification
    """
    energy_col = get_df_col(psd_report, DataframeColumn.CALIB_ENERGY)
    psd_col = get_df_col(psd_report, DataframeColumn.PSD)
    lb_fits = lb_fit_fn(energy_col.astype(float)) # type: ignore
    ub_fits = ub_fit_fn(energy_col.astype(float)) # type: ignore
    within_psd_bounds = psd_col.between(
        lb_fits + window_adj_offset, ub_fits
    )
    within_eng_bounds = energy_col >= le_cutoff
    psd_report[label.value] = within_psd_bounds & within_eng_bounds
    
    return psd_report