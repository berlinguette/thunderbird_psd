import pandas as pd
from data_processing.types import VectorLikeFunction
from data_processing.dataframe_validation import DetectorDataframeColumn, get_df_col

def add_recalibration_column(df: pd.DataFrame, calibration_fn: VectorLikeFunction) -> pd.DataFrame:
    raw_energy_col = get_df_col(df, DetectorDataframeColumn.ENERGY)
    recalib_col: pd.Series = calibration_fn(raw_energy_col)
    df[DetectorDataframeColumn.RECALIBRATED_ENERGY.value] = recalib_col
    return df
