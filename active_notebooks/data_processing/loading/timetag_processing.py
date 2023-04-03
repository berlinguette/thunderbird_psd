import pandas as pd
from datetime import datetime
from dataframe_validation import PsdDfColumn


def calculate_timetag_hours(df: pd.DataFrame) -> pd.DataFrame:
    if PsdDfColumn.TIMETAG.value not in df:
        return df
    hours_col = df[PsdDfColumn.TIMETAG.value] * 1e-12/3600
    df[PsdDfColumn.TIME_HOURS.value] = hours_col
    return df

def calculate_event_time(df: pd.DataFrame, start_time: datetime) -> pd.DataFrame:
    if PsdDfColumn.TIMETAG.value not in df:
        return df
    timetag_col = df[PsdDfColumn.TIMETAG.value]
    timetag_deltas: pd.Series = pd.to_timedelta(
        timetag_col * 1E-3, 
        unit='ns')
    event_time_col = timetag_deltas + start_time  # type: ignore
    df[PsdDfColumn.EVENT_TIME.value] = event_time_col
    df[PsdDfColumn.EVENT_TIME_PS.value] = (timetag_col % 1000).astype(int)
    return df
