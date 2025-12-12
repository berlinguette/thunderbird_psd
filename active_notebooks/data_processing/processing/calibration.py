import pandas as pd
from enum import Enum
from typing import Final
from data_processing.types import VectorLikeFunction, VectorLike, LinearCalibrationParams, LogCurveCalibrationParams
from data_processing.dataframe_validation import DetectorDataframeColumn, get_df_col
from abc import ABC, abstractmethod
import numpy as np
from scipy.special import lambertw

class Detector(Enum):
    ZERO = 0
    ONE = 1

class CalibrationType(Enum):
    LINEAR = "LINEAR"
    LOG_CURVE = "LOG_CURVE"
    
DetectorLinearCalibrationParams: Final = {
    Detector.ZERO: LinearCalibrationParams(p1=2.179, p2=41.67),
    Detector.ONE: LinearCalibrationParams(p1=1.884, p2=26.42)
}

DetectorLogCurveCalibrationParams: Final = {
    Detector.ZERO: LogCurveCalibrationParams(a=-3.087, b=3.146e+7, c=710.7),
    Detector.ONE: LogCurveCalibrationParams(a=-3.087, b=3.146e+7, c=710.7)
}

class AbstractCalibrator(ABC):
    def __init__(self, detector: Detector):
        self._detector = detector

    @property
    @abstractmethod
    def source_column(self) -> DetectorDataframeColumn:
        ...

    @abstractmethod
    def _calibration_fn(self) -> VectorLikeFunction:
        ...
    
    def calibrate(self, detector_df: pd.DataFrame) -> pd.DataFrame:
        raw_col = get_df_col(detector_df, self.source_column)
        recalib_col: pd.Series = self._calibration_fn(raw_col)
        detector_df[DetectorDataframeColumn.RECALIBRATED_ENERGY.value] = recalib_col
        return detector_df

class LinearCalibrator(AbstractCalibrator):
    def __init__(self, detector: Detector):
        super().__init__(detector=detector)
        self._calibration_params = DetectorLinearCalibrationParams[self._detector]

    @property
    def source_column(self) -> DetectorDataframeColumn:
        return DetectorDataframeColumn.ENERGY

    def _calibration_fn(self) -> VectorLikeFunction:
        p1 = self._calibration_params.p1
        p2 = self._calibration_params.p2
        return lambda x: (x-p2)/(p1*1000)
    

class LogCurveCalibrator(AbstractCalibrator):
    def __init__(self, detector: Detector):
        super().__init__(detector=detector)
        self._calibration_params = DetectorLogCurveCalibrationParams[self._detector]

    @property
    def source_column(self) -> DetectorDataframeColumn:
        return DetectorDataframeColumn.PULSE_HEIGHT
    
    def _cal_fn(self, x: VectorLike) -> float:
        a = self._calibration_params.a
        b = self._calibration_params.b
        c = self._calibration_params.c
        exponent = c/(a*b) + x/a
        w_param = (c*np.exp(exponent))/(a*b)
        y = (-c + a*b*lambertw(w_param))/(b*c)
        return float(np.real(y))
    
    def _calibration_fn(self) -> VectorLikeFunction:
        return self._cal_fn


class CalibratorFactory:
    @staticmethod
    def make_calibrator(detector: Detector, calibration_type: CalibrationType) -> AbstractCalibrator:
        match calibration_type:
            case CalibrationType.LINEAR:
                return LinearCalibrator(detector)
            case CalibrationType.LOG_CURVE:
                return LogCurveCalibrator(detector)
            case _:
                raise ValueError(f"Invalid calibration type: {calibration_type}")


def recalibrate(df: pd.DataFrame, detector: Detector, calibration_type: CalibrationType) -> pd.DataFrame:
    calibrator = CalibratorFactory.make_calibrator(detector, calibration_type)
    recalibrated_df = calibrator.calibrate(df)
    return recalibrated_df
