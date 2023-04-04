import pandas as pd
from datetime import datetime
from dataframe_validation import DataframeColumn, get_df_col


def calculate_timetag_hours(df: pd.DataFrame) -> pd.DataFrame:
    if DataframeColumn.TIMETAG.value not in df:
        return df
    hours_col = get_df_col(df, DataframeColumn.TIMETAG) * 1e-12/3600
    df[DataframeColumn.TIME_HOURS.value] = hours_col
    return df

def calculate_event_time(df: pd.DataFrame, start_time: datetime) -> pd.DataFrame:
    if DataframeColumn.TIMETAG.value not in df:
        return df
    timetag_col = get_df_col(df, DataframeColumn.TIMETAG)
    timetag_deltas: pd.Series = pd.to_timedelta(
        timetag_col * 1E-3, 
        unit='ns')
    event_time_col = timetag_deltas + start_time  # type: ignore
    df[DataframeColumn.EVENT_TIME.value] = event_time_col
    df[DataframeColumn.EVENT_TIME_PS.value] = (timetag_col % 1000).astype(int)
    return df
