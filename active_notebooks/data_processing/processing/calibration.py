"""This module is responsible for energy calibration of detector data."""
import pandas as pd
from enum import Enum
from typing import Final
from data_processing.types import VectorLikeFunction, CalibrationParams
from data_processing.dataframe_validation import DetectorDataframeColumn, get_df_col

class Detector(Enum):
    """Code for Thunderbird neutron detectors
    """
    ZERO = 0
    ONE = 1

"""Dictionary calibration parameters for each detector
"""
DetectorCalibrationParams: Final = {
    Detector.ZERO: CalibrationParams(p1=2.179, p2=41.67),
    Detector.ONE: CalibrationParams(p1=1.884, p2=26.42)
}

def recalibrate(df: pd.DataFrame, detector: Detector) -> pd.DataFrame:
    """Make a new energy column in the detector dataframe based on the detector used.
    Calibration uses the "ENERGY" column, which contains the integrated pulse area from the raw ADC data

    :param df: Dataframe of neutron detector data
    :type df: pd.DataFrame
    :param detector: Detector used
    :type detector: Detector
    :return: Dataframe with new calibrated energy column
    :rtype: pd.DataFrame
    """
    calib_params = DetectorCalibrationParams[detector]
    calib_fn = _make_calibration_fn(calib_params)
    return _add_recalibration_column(df, calib_fn)

def _add_recalibration_column(df: pd.DataFrame, calibration_fn: VectorLikeFunction) -> pd.DataFrame:
    raw_energy_col = get_df_col(df, DetectorDataframeColumn.ENERGY)
    recalib_col: pd.Series = calibration_fn(raw_energy_col)
    df[DetectorDataframeColumn.RECALIBRATED_ENERGY.value] = recalib_col
    return df

def _make_calibration_fn(params: CalibrationParams) -> VectorLikeFunction:
    # return lambda x: params.p1*x+params.p2
    return lambda x: (x-params.p2)/(params.p1*1000)
