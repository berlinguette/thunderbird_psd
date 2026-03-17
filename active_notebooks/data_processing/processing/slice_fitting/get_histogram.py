import numpy as np
import pandas as pd
import polars as pl
from data_processing.dataframe_validation import (
    DetectorDataframeColumn,
    EnergyColumn,
    get_df_col,
    get_lf_col_expr
)


def get_psd_energy_histogram(
    df: pd.DataFrame,
    energy_column: EnergyColumn,
    energy_width: float = 0.0150,
    energy_bins: np.ndarray | None = None,
    psd_bin_count: int = 100,
    psd_min: float = 0.0,
    psd_max: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = get_df_col(df, energy_column)
    y = get_df_col(df, DetectorDataframeColumn.PSD)

    within_psd = y.between(psd_min, psd_max)
    y = y[within_psd == True].copy()
    x = x[within_psd == True].copy()

    if energy_bins is not None:
        x_bins = energy_bins
    else:
        x_bins: np.ndarray = np.linspace(0, x.max(), int(x.max() / energy_width) + 1)
    print(f"Energy width = {x_bins[1]-x_bins[0]} MeVee")
    y_bins: np.ndarray = np.linspace(psd_min, psd_max, psd_bin_count + 1)

    Z, xe, ye = np.histogram2d(x, y, bins=[x_bins, y_bins])
    return Z, xe, ye


def get_psd_energy_histogram_polars(
    lf: pl.LazyFrame,
    energy_column: EnergyColumn,
    energy_width: float = 0.0150,
    energy_bins: np.ndarray | None = None,
    psd_bin_count: int = 100,
    psd_min: float = 0.0,
    psd_max: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_col = get_lf_col_expr(energy_column)
    y_col = get_lf_col_expr(DetectorDataframeColumn.PSD)

    within_psd = y_col.is_between(psd_min, psd_max)
    within_df = (
        lf.filter(within_psd)
        .with_columns(x_col.alias("x"), y_col.alias("y"))
        .select("x", "y")
        .collect()
    )
    x = within_df.get_column("x").to_pandas()
    y = within_df.get_column("y").to_pandas()

    if energy_bins is not None:
        x_bins = energy_bins
    else:
        x_bins: np.ndarray = np.linspace(0, x.max(), int(x.max() / energy_width) + 1)
    y_bins: np.ndarray = np.linspace(psd_min, psd_max, psd_bin_count + 1)

    Z, xe, ye = np.histogram2d(x, y, bins=[x_bins, y_bins])
    return Z, xe, ye
