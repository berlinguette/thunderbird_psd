import numpy as np
import pandas as pd
from data_processing.types import WindowBorders, WindowBorderFunction
from data_processing.dataframe_validation import DataframeColumn, get_df_col
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.signal import savgol_filter


def generate_nasa_neutron_window(
    slice_fit_df: pd.DataFrame,
    window_offset: float = 0.2,
    sigma: float = 5,
    lower_energy_bound: float = 0.1966,
) -> WindowBorders:
    # Define lower and upper bounds (neutron_lb_fit, neutron_ub_fit)
    neutron_lb = savgol_filter(
        slice_fit_df["mu1"] + sigma * slice_fit_df["sigma1"],
        window_length=21,
        polyorder=3,
    )  # reduce noise
    slice_xs = _get_energy_midpoints(slice_fit_df)
    neutron_lb_fit: WindowBorderFunction = interp1d(
        slice_xs,
        neutron_lb,
        fill_value=(neutron_lb[0], neutron_lb[-1]),  # type: ignore
        bounds_error=False,
    )  # now it's a function!

    def neutron_ub_fit(x: pd.Series[float]) -> pd.Series[float]:
        # upper bound is just a fixed PSD offset from lower bound
        # as observed in NASA paper graphs
        return neutron_lb_fit(x) + window_offset

    return WindowBorders(
        left=lower_energy_bound, bottom=neutron_lb_fit, top=neutron_ub_fit, right=None
    )


def generate_new_neutron_window(
    slice_fit_df: pd.DataFrame,
    sigma: float = 3,
    fom_energy_range: tuple[float, float] = (0.10, 0.35),
) -> WindowBorders:
    energy_bin_left_edges = slice_fit_df['slice_energy_min']
    energy_bin_midpoints = _get_energy_midpoints(slice_fit_df)
    left_border = _generate_new_window_left_border(
        slice_fit_df, energy_bin_midpoints, fom_energy_range=fom_energy_range
    )
    right_border = 0.688  # keVee, Compton edge + detector resolution
    bottom_border = _generate_new_window_bottom_border(
        slice_fit_df, energy_bin_left_edges, sigma
    )
    top_border = _generate_new_window_top_border(
        slice_fit_df, energy_bin_left_edges, sigma
    )
    return WindowBorders(
        left=left_border, right=right_border, bottom=bottom_border, top=top_border
    )


def _generate_new_window_left_border(
    slice_fit_df: pd.DataFrame,
    energy_bin_midpoints: pd.Series,
    fom_energy_range: tuple[float, float]
) -> float:
    # create interp: x = energy_bin_midpoints, y = fom
    energy_fom_curve = interp1d(energy_bin_midpoints, slice_fit_df["fom"] - 1.27)
    # use root_scalar to find E where FOM = 1.27 and return
    root_results = root_scalar(energy_fom_curve, bracket=fom_energy_range)
    if root_results.converged:
        return root_results.root
    else:
        raise ValueError(
            f"Left border could not be found after {root_results.iterations}: "
            + f"{root_results.flag}"
        )


def _generate_new_window_bottom_border(
    slice_fit_df: pd.DataFrame, energy_bin_left_edges: pd.Series, sigma: float
) -> WindowBorderFunction:
    border_psd_values = slice_fit_df["mu2"] - sigma * slice_fit_df["sigma2"]
    fill_values = (border_psd_values.iloc[0], border_psd_values.iloc[-1])
    border: WindowBorderFunction = interp1d(
        energy_bin_left_edges,
        border_psd_values,
        kind="previous",
        fill_value=fill_values,  # type: ignore
        bounds_error=False,
    )
    return border


def _generate_new_window_top_border(
    slice_fit_df: pd.DataFrame, energy_bin_left_edges: pd.Series, sigma: float
) -> WindowBorderFunction:
    border_psd_values = slice_fit_df["mu2"] + sigma * slice_fit_df["sigma2"]
    fill_values = (border_psd_values.iloc[0], border_psd_values.iloc[-1])
    border: WindowBorderFunction = interp1d(
        energy_bin_left_edges,
        border_psd_values,
        kind="previous",
        fill_value=fill_values,  # type: ignore
        bounds_error=False,
    )
    return border


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
    bottom_border = (
        bottom_border_fn(energy_col.astype(float))
        if bottom_border_fn is not None
        else None
    )
    top_border = (
        top_border_fn(energy_col.astype(float)) if top_border_fn is not None else None
    )
    within_psd_bounds = _is_within_bounds(psd_col, bottom_border, top_border)

    left_border = borders.left
    right_border = borders.right
    within_energy_bounds = _is_within_bounds(energy_col, left_border, right_border)

    psd_report[label.value] = within_psd_bounds & within_energy_bounds

    return psd_report


def _is_within_bounds(
    value_col: pd.Series,
    lower_bound: float | pd.Series | None,
    upper_bound: float | pd.Series | None,
) -> pd.Series:
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


def _get_energy_midpoints(df: pd.DataFrame) -> pd.Series:
    slice_energy_min = df['slice_energy_min']
    slice_energy_max = df['slice_energy_max']
    intervals = pd.Series(pd.arrays.IntervalArray.from_arrays(slice_energy_min, slice_energy_max), index=df.index)
    midpoints = pd.Series(intervals.array.mid, index=df.index)  # type: ignore
    return midpoints
