from typing import Final

import pandas as pd
from data_processing.dataframe_validation import DetectorDataframeColumn, get_df_col
from data_processing.types import (
    CalibrationParams,
    ChannelDataframes,
    VectorLikeFunction,
)
from data_processing.typing.enums import Detector

DetectorCalibrationParams: Final = {
    Detector.ZERO: CalibrationParams(p1=2.179, p2=41.67),
    Detector.ONE: CalibrationParams(p1=1.884, p2=26.42),
}


def recalibrate(dfs: ChannelDataframes) -> ChannelDataframes:
    return {idx: _recalibrate_for_df(df, Detector(idx)) for idx, df in dfs.items()}


def _recalibrate_for_df(df: pd.DataFrame, detector: Detector) -> pd.DataFrame:
    calib_params = DetectorCalibrationParams[detector]
    calib_fn = _make_calibration_fn(calib_params)
    return _add_recalibration_column(df, calib_fn)


def _add_recalibration_column(
    df: pd.DataFrame, calibration_fn: VectorLikeFunction
) -> pd.DataFrame:
    raw_energy_col = get_df_col(df, DetectorDataframeColumn.ENERGY)
    recalib_col: pd.Series = calibration_fn(raw_energy_col)
    df[DetectorDataframeColumn.RECALIBRATED_ENERGY.value] = recalib_col
    return df


def _make_calibration_fn(params: CalibrationParams) -> VectorLikeFunction:
    return lambda x: (x - params.p2) / (params.p1 * 1000)
