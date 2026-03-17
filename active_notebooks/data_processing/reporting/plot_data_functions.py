import pandas as pd
from data_processing.dataframe_validation import DetectorDataframeColumn, EnergyColumn, get_df_col
from data_processing.types import GraphData


def get_tail_vs_total_data(df: pd.DataFrame) -> GraphData:
    energy_col = get_df_col(df, DetectorDataframeColumn.ENERGY)
    short_col = get_df_col(df, DetectorDataframeColumn.ENERGYSHORT)
    data: GraphData = {"x": energy_col, "y": energy_col - short_col}
    return data


def get_psd_histogram_data(df: pd.DataFrame, energy_column: EnergyColumn) -> GraphData:
    energy_col = get_df_col(df, energy_column)
    psd_col = get_df_col(df, DetectorDataframeColumn.PSD)
    data: GraphData = {"x": energy_col, "y": psd_col}
    return data
