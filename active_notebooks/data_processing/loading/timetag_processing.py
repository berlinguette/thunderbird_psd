import pandas as pd
from datetime import datetime
from data_processing.dataframe_validation import DetectorDataframeColumn, get_df_col


def calculate_timetag_hours(df: pd.DataFrame) -> pd.DataFrame:
    if DetectorDataframeColumn.TIMETAG.value not in df:
        return df
    hours_col = get_df_col(df, DetectorDataframeColumn.TIMETAG) * 1e-12/3600
    df.loc[:, DetectorDataframeColumn.TIME_HOURS.value] = hours_col
    return df

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
