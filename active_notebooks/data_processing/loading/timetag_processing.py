import pandas as pd
import polars as pl
from datetime import datetime
from typing import Final
from data_processing.dataframe_validation import DetectorDataframeColumn, get_df_col

NS_IN_SEC: Final = 1e12
SEC_IN_MIN: Final = 60
MIN_IN_HOUR: Final = 60
NS_IN_HOUR: Final = NS_IN_SEC*SEC_IN_MIN*MIN_IN_HOUR

def calculate_timetag_hours(df: pd.DataFrame) -> pd.DataFrame:
    if DetectorDataframeColumn.TIMETAG.value not in df:
        return df
    hours_col = get_df_col(df, DetectorDataframeColumn.TIMETAG) / NS_IN_HOUR
    df.loc[:, DetectorDataframeColumn.TIME_HOURS.value] = hours_col
    return df

def calculate_timetag_hours_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    timetag_col = DetectorDataframeColumn.TIMETAG.value
    schema = lf.collect_schema()
    if timetag_col not in schema:
        return lf
    hours_lf = lf.with_columns((pl.col(timetag_col) / NS_IN_HOUR).alias(DetectorDataframeColumn.TIME_HOURS.value))
    return hours_lf

def calculate_event_time(df: pd.DataFrame, start_time: datetime) -> pd.DataFrame:
    if DetectorDataframeColumn.TIMETAG.value not in df:
        return df
    timetag_col = get_df_col(df, DetectorDataframeColumn.TIMETAG)
    timetag_deltas: pd.Series = pd.to_timedelta(
        timetag_col * 1E-3, 
        unit='ns')
    event_time_col = timetag_deltas + start_time  # type: ignore
    df.loc[:, DetectorDataframeColumn.EVENT_TIME.value] = event_time_col
    df.loc[:, DetectorDataframeColumn.EVENT_TIME_PS.value] = (timetag_col % 1000).astype(int)
    return df
