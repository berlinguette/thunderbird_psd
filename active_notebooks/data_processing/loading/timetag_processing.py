from datetime import datetime
from copy import copy

import pandas as pd
from data_processing.dataframe_validation import DetectorDataframeColumn, get_df_col
from data_processing.typing.dataclasses import experiment_neutron_data as end

def calculate_timetag_hours(
    multiple_exp_data: end.MultipleExperimentData
) -> end.MultipleExperimentData:
    new_multi_exp_data = {}
    for exp_id, exp_data in multiple_exp_data.items():
        for detector, exp_det_data_list in exp_data.items():
            new_exp_det_data_list = []
            for exp_det_data in exp_det_data_list:
                old_raw_data = exp_det_data.raw_data
                old_neutron_df = old_raw_data.neutron_detector
                
                new_neutron_df = _calculate_timetag_hours_for_df(old_neutron_df)
                
                new_raw_data = copy(old_raw_data)
                new_raw_data.neutron_detector = new_neutron_df
                
                new_exp_det_data = copy(exp_det_data)
                new_exp_det_data.raw_data = new_raw_data
                
                new_exp_det_data_list.append(new_exp_det_data)
            new_multi_exp_data[exp_id][detector] = new_exp_det_data_list
    return new_multi_exp_data


def _calculate_timetag_hours_for_df(df: pd.DataFrame) -> pd.DataFrame:
    if DetectorDataframeColumn.TIMETAG.value not in df:
        return df
    hours_col = get_df_col(df, DetectorDataframeColumn.TIMETAG) * 1e-12 / 3600
    df[DetectorDataframeColumn.TIME_HOURS.value] = hours_col
    return df


def calculate_event_time(
    multiple_exp_data: end.MultipleExperimentData
) -> end.MultipleExperimentData:
    new_multi_exp_data = {}
    for exp_id, exp_data in multiple_exp_data.items():
        for detector, exp_det_data_list in exp_data.items():
            new_exp_det_data_list = []
            for exp_det_data in exp_det_data_list:
                old_raw_data = exp_det_data.raw_data
                old_neutron_df = old_raw_data.neutron_detector
                
                new_neutron_df = _calculate_timetag_hours_for_df(old_neutron_df)  # TODO fix this!
                
                new_raw_data = copy(old_raw_data)
                new_raw_data.neutron_detector = new_neutron_df
                
                new_exp_det_data = copy(exp_det_data)
                new_exp_det_data.raw_data = new_raw_data
                
                new_exp_det_data_list.append(new_exp_det_data)
            new_multi_exp_data[exp_id][detector] = new_exp_det_data_list
    return new_multi_exp_data


# def calculate_event_time(
#     dfs: ChannelDataframes, start_time: datetime
# ) -> ChannelDataframes:
#     return {
#         idx: _calculate_event_time_for_df(df, start_time) for idx, df in dfs.items()
#     }


def _calculate_event_time_for_df(
    df: pd.DataFrame, start_time: datetime
) -> pd.DataFrame:
    if DetectorDataframeColumn.TIMETAG.value not in df:
        return df
    timetag_col = get_df_col(df, DetectorDataframeColumn.TIMETAG)
    timetag_deltas: pd.Series = pd.to_timedelta(timetag_col * 1e-3, unit="ns")
    event_time_col = timetag_deltas + start_time  # type: ignore
    df[DetectorDataframeColumn.EVENT_TIME.value] = event_time_col
    df[DetectorDataframeColumn.EVENT_TIME_PS.value] = (timetag_col % 1000).astype(int)
    return df
