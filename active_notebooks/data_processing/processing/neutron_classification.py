"""This module is responsible for neutron classification."""
# from typing import Any

import pandas as pd
from data_processing.dataframe_validation import (
    DetectorDataframeColumn,
    EnergyColumn,
    get_df_col,
)
from data_processing.types import WindowBorders
# from scipy.interpolate import interp1d
# from scipy.optimize import root_scalar
# from scipy.signal import savgol_filter


def classify(
    psd_report: pd.DataFrame,
    energy_column: EnergyColumn,
    borders: WindowBorders,
    label: DetectorDataframeColumn = DetectorDataframeColumn.NEUTRON_CLASS
) -> pd.DataFrame:
    """Classify signals as neutron or non-neutron for a given "window"
    Classification creates a new column of boolean values, where True indicates a neutron classified signal.
    Signals with PSD/energy values within the window are classified as neutrons.

    :param psd_report: Dataframe containing signal PSD (in "tail / total" column) and energy data
    :type psd_report: pd.DataFrame
    :param energy_column: Dataframe column containing pulse energy values
    :type energy_column: EnergyColumn
    :param borders: Neutron window borders
    :type borders: WindowBorders
    :param label: Label of neutron classification column, defaults to DetectorDataframeColumn.NEUTRON_CLASS
    :type label: DetectorDataframeColumn, optional
    :return: Original DataFrame with new column for neutron classification
    :rtype: pd.DataFrame
    """
    energy_col = get_df_col(psd_report, energy_column)
    psd_col = get_df_col(psd_report, DetectorDataframeColumn.PSD)

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
