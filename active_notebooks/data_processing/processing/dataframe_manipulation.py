"""This module is responsible for neutron dataframe manipulation."""
import pandas as pd
from data_processing.dataframe_validation import DetectorDataframeColumn


def generate_full_neutron_df(
    classified_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    classification_col: DetectorDataframeColumn = DetectorDataframeColumn.NEUTRON_CLASS
) -> pd.DataFrame:
    """Makes a full neutron dataframe from the neutron detector dataframe with classification and the signals dataframe

    :param classified_df: Neutron detector dataframe with neutron classification column
    :type classified_df: pd.DataFrame
    :param signals_df: Detector signals dataframe
    :type signals_df: pd.DataFrame
    :param classification_col: Classification column, defaults to DetectorDataframeColumn.NEUTRON_CLASS
    :type classification_col: DetectorDataframeColumn, optional
    :return: Full neutron dataframe
    :rtype: pd.DataFrame
    """
    neutrons_only = classified_df.query(classification_col.value)
    full_neutron_df = neutrons_only.join(signals_df)
    return full_neutron_df


def generate_neutron_signals(
    neutron_events_df: pd.DataFrame,
    signal_length: int = 200
) -> pd.DataFrame:
    """Generate neutron signals dataframe from neutron events dataframe.

    :param neutron_events_df: Neutron events dataframe
    :type neutron_events_df: pd.DataFrame
    :param signal_length: Length of output signal, defaults to 200
    :type signal_length: int, optional
    :return: Neutron signals dataframe
    :rtype: pd.DataFrame
    """
    signal_start = '0'
    signal_end = str(signal_length-1)
    neutron_signals = neutron_events_df.loc[:, signal_start:signal_end]
    neutron_signals = neutron_signals.astype(int)
    neutron_signals = neutron_signals.transpose()
    return neutron_signals


# def select_neutron_traces(
#     neutron_events_df: pd.DataFrame,
#     query_params: dict[str, tuple[float | None, float | None]]
# ) -> pd.DataFrame:
#     psd_start, psd_end = query_params.get('PSD', (None, None))
#     eng_start, eng_end = query_params.get('Energy', (None, None))
#     time_start, time_end = query_params.get('Time', (None, None))

#     query_cols = ["`tail / total`", 'CALIB_ENERGY', 'EVENT_TIME']
#     query_var_name = ['psd', 'eng', 'time']
#     start_vars = [psd_start, eng_start, time_start]
#     end_vars = [psd_end, eng_end, time_end]

#     start_queries = [f"{col} >= @{var_name}_start"
#                      for col, var_name, start_var
#                      in zip(query_cols, query_var_name, start_vars)
#                      if start_var is not None]
#     end_queries = [f"{col} >= @{var_name}_start"
#                      for col, var_name, end_var
#                      in zip(query_cols, query_var_name, end_vars)
#                      if end_var is not None]
#     queries = start_queries + end_queries

#     query_string = ' & '.join(queries)
#     matching_neutrons = neutron_events_df.query(query_string)
#     return matching_neutrons
