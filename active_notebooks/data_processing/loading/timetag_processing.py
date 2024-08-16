"""This module is responsible for converting CAEN timetags into other formats."""
import pandas as pd
from datetime import datetime
from data_processing.dataframe_validation import DetectorDataframeColumn, get_df_col


def calculate_timetag_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the elapsed hours for each event's timetag.

    :param df: DataFrame with the 'TIMETAG' column
    :type df: pd.DataFrame
    :return: Given dataframe with a new 'TIMETAG_HOURS' column containing elapsed event time in hours
    :rtype: pd.DataFrame
    """
    if DetectorDataframeColumn.TIMETAG.value not in df:
        return df
    hours_col = get_df_col(df, DetectorDataframeColumn.TIMETAG) * 1e-12/3600
    df[DetectorDataframeColumn.TIME_HOURS.value] = hours_col
    return df

def calculate_event_time(df: pd.DataFrame, start_time: datetime) -> pd.DataFrame:
    """Calculates the elapsed hours for each event's timetag.

    :param df: DataFrame with the 'TIMETAG' column
    :type df: pd.DataFrame
    :param start_time: Clock time (with timezone) when neutron detection was started
    :type start_time: datetime
    :return: Given dataframe with two new columns: 'EVENT_TIME' with the event clock time, and 'EVENT_TIME_PS" with the picosecond remainder
    :rtype: pd.DataFrame
    """
    """Calculates the elapsed hours for each event's timetag.

    :param df: DataFrame with the 'TIMETAG' column
    :type df: pd.DataFrame
    :return: Given dataframe with a new 'TIMETAG_HOURS' column
    :rtype: pd.DataFrame
    """
    if DetectorDataframeColumn.TIMETAG.value not in df:
        return df
    timetag_col = get_df_col(df, DetectorDataframeColumn.TIMETAG)
    timetag_deltas: pd.Series = pd.to_timedelta(
        timetag_col * 1E-3, 
        unit='ns')
    event_time_col = timetag_deltas + start_time  # type: ignore
    df[DetectorDataframeColumn.EVENT_TIME.value] = event_time_col
    df[DetectorDataframeColumn.EVENT_TIME_PS.value] = (timetag_col % 1000).astype(int)
    return df
