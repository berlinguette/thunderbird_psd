from dataclasses import dataclass
from datetime import datetime
from data_processing.typing.enums import Detector
from pandas import DataFrame
from data_processing.types import CalibrationType, WindowType, NeutronWindowSettings, WindowBorders
from data_processing.processing.neutron_window_strategy.abstract_strategy import AbstractNeutronWindowStrategy
from typing import Literal
from numpy import ndarray


@dataclass
class RawExperimentalData:
    neutron_detector: DataFrame
    start_time: datetime | None
    reactor: DataFrame | None = None


@dataclass
class AnalysisSettings:
    calibration_type: CalibrationType
    border_type: WindowType
    border_strategy_settings: NeutronWindowSettings
    time_bin_length: int | None = None
    

@dataclass
class PsdHistogram:
    histogram: ndarray
    x_edges: ndarray
    y_edges: ndarray


@dataclass
class BimodalFitting:
    bad_slice_indexes: ndarray | None
    fit_results: DataFrame


@dataclass
class NeutronBorders:
    border_strategy: AbstractNeutronWindowStrategy
    border: WindowBorders | None = None


@dataclass
class NeutronClassification:
    classified: DataFrame
    neutrons_only: DataFrame
    gamma_only: DataFrame


@dataclass
class TimeSeriesData:
    time_bin_edges: ndarray
    neutron_rates: DataFrame | None = None
    gamma_rates: DataFrame | None = None
    reactor: DataFrame | None = None
    merged: DataFrame | None = None
    
    
@dataclass
class GammaSpectrumData:
    spectrum: DataFrame
    bin_edges: ndarray


@dataclass
class ExperimentDetectorData:
    raw_data: RawExperimentalData
    analysis_settings: AnalysisSettings | None = None
    psd_histogram: PsdHistogram | None = None
    bimodal_fitting: BimodalFitting | None = None
    borders: NeutronBorders | None = None
    classification: NeutronClassification | None = None
    gamma_spectrum: GammaSpectrumData | None = None
    

ExperimentData = dict[Detector, list[ExperimentDetectorData]]
MultipleExperimentData = dict[str, ExperimentData]
# all experiment data
# - by experiment id
# -- by detector
# --- data model