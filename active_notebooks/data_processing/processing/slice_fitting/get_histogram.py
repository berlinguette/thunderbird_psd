import numpy as np
import pandas as pd
from data_processing.dataframe_validation import (
    DetectorDataframeColumn,
    EnergyColumn,
    get_df_col,
)


def get_psd_energy_histogram(
    df: pd.DataFrame,
    energy_column: EnergyColumn,
    energy_width: float = 0.0150,
    psd_bin_count: int = 100,
    psd_min: float = 0.0,
    psd_max: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gets PSD/energy 2D histogram for a set of neutron detector energy.

    :param df: Dataframe of all neutron detector events
    :type df: pd.DataFrame
    :param energy_column: Name of energy column in events dataframe
    :type energy_column: EnergyColumn
    :param energy_width: Desired energy bin width, defaults to 0.0150
    :type energy_width: float, optional
    :param psd_bin_count: Desired number of PSD bins, defaults to 100
    :type psd_bin_count: int, optional
    :param psd_min: Minimum PSD value used in histogram, defaults to 0.0
    :type psd_min: float, optional
    :param psd_max: Maximum PSD value used in histogram, defaults to 0.5
    :type psd_max: float, optional
    :return: PSD/energy 2D histogram, energy bins, PSD bins
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    x = get_df_col(df, energy_column)
    y = get_df_col(df, DetectorDataframeColumn.PSD)

    within_psd = y.between(psd_min, psd_max)
    y = y[within_psd == True].copy()
    x = x[within_psd == True].copy()

    x_bins: np.ndarray = np.linspace(0, x.max(), int(x.max() / energy_width) + 1)
    print(f"Energy width = {x_bins[1]-x_bins[0]} MeVee")
    y_bins: np.ndarray = np.linspace(psd_min, psd_max, psd_bin_count + 1)

    Z, xe, ye = np.histogram2d(x, y, bins=[x_bins, y_bins])
    return Z, xe, ye
