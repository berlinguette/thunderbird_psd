"""This module is responsible for helping to make valid neutron detection dataframes."""
from enum import Enum
from typing import Final, Literal

from numpy import int64
from pandas import DataFrame, Series


class DetectorDataframeColumn(Enum):
    CALIB_ENERGY = "CALIB_ENERGY"
    RECALIBRATED_ENERGY = "RECALIB_ENERGY"
    ENERGYSHORT = "ENERGYSHORT"
    ENERGY = "ENERGY"
    TIMETAG = "TIMETAG"
    PSD = "tail / total"
    TIME_HOURS = "TIMETAG_HOURS"
    EVENT_TIME = "EVENT_TIME"
    EVENT_TIME_PS = "EVENT_TIME_PS"
    NEUTRON_CLASS = "NASA"
    NEUTRON_RECALC_CLASS = "NASA_RECALC"
    NEW_N_CLASS = "IS_NEUTRON"


STARTING_COLUMNS: Final = [
    DetectorDataframeColumn.CALIB_ENERGY,
    DetectorDataframeColumn.ENERGYSHORT,
    DetectorDataframeColumn.ENERGY,
    DetectorDataframeColumn.TIMETAG,
]
STARTING_COL_NAMES: Final = [e.value for e in STARTING_COLUMNS]
col_types = {
    DetectorDataframeColumn.CALIB_ENERGY: float,
    DetectorDataframeColumn.ENERGYSHORT: int,
    DetectorDataframeColumn.ENERGY: int,
    DetectorDataframeColumn.TIMETAG: int64,
}
STARTING_COL_TYPES: Final = {k.value: v for k, v in col_types.items()}
EnergyColumn = Literal[DetectorDataframeColumn.CALIB_ENERGY, DetectorDataframeColumn.RECALIBRATED_ENERGY]


class SliceFitDataframeColumn(Enum):
    INDEX = "i"
    GAMMA_MU = "mu1"
    GAMMA_SIGMA = "sigma1"
    GAMMA_AMPLITUDE = "a1"
    NEUTRON_MU = "mu2"
    NEUTRON_SIGMA = "sigma2"
    NEUTRON_AMPLITUDE = "a2"
    SLICE_ENERGY_MINIMUM = "slice_energy_min"
    SLICE_ENERGY_MAXIMUM = "slice_energy_max"
    FOM = "fom"
    
FIT_COLUMNS: Final = [
    SliceFitDataframeColumn.INDEX, 
    SliceFitDataframeColumn.GAMMA_MU, 
    SliceFitDataframeColumn.GAMMA_SIGMA, 
    SliceFitDataframeColumn.GAMMA_AMPLITUDE, 
    SliceFitDataframeColumn.NEUTRON_MU, 
    SliceFitDataframeColumn.NEUTRON_SIGMA, 
    SliceFitDataframeColumn.NEUTRON_AMPLITUDE, 
    SliceFitDataframeColumn.SLICE_ENERGY_MINIMUM, 
    SliceFitDataframeColumn.SLICE_ENERGY_MAXIMUM, 
    SliceFitDataframeColumn.FOM
]
FIT_COLUMN_NAMES: Final = [e.value for e in FIT_COLUMNS]
FIT_ERROR_COLUMNS: Final = [
    SliceFitDataframeColumn.INDEX, 
    SliceFitDataframeColumn.GAMMA_MU, 
    SliceFitDataframeColumn.GAMMA_SIGMA, 
    SliceFitDataframeColumn.GAMMA_AMPLITUDE, 
    SliceFitDataframeColumn.NEUTRON_MU, 
    SliceFitDataframeColumn.NEUTRON_SIGMA, 
    SliceFitDataframeColumn.NEUTRON_AMPLITUDE, 
    SliceFitDataframeColumn.SLICE_ENERGY_MINIMUM, 
    SliceFitDataframeColumn.SLICE_ENERGY_MAXIMUM
]
FIT_ERROR_COLUMN_NAMES: Final = [e.value for e in FIT_ERROR_COLUMNS]

class NonReactorDataframeColumn(Enum):
    TIME = "Time"
    DATA = "Data"
    UNITS = "Units"
    NORMALIZED_DATA = "Normalized Data"
    NORMALIZED_UNITS = "Normalized Units"

class BinningDataframeColumn(Enum):
    TIME_BIN = "Time Bin"
    ENERGY_BIN = "Energy Bin"
    BIN_MIDPOINT = "Bin midpoint"
    BIN_TIME = "Bin time (s)"
    COUNT = "count"
    COUNT_ERROR = "count_error"
    NEUTRON_RATE = "Neutron rate (cps)"
    NEUTRON_RATE_ERROR = "Neutron error (cps)"
    GAMMA_RATE = "Background gamma rate (cps)"
    GAMMA_RATE_ERROR = "Gamma error (cps)"

def get_df_col(df: DataFrame, col: DetectorDataframeColumn|SliceFitDataframeColumn) -> Series:
    """Get given column from given dataframe.

    :param df: Dataframe to get column from
    :type df: DataFrame
    :param col: Column name
    :type col: DetectorDataframeColumn | SliceFitDataframeColumn
    :return: Desired dataframe column
    :rtype: Series
    """
    return df[col.value]
