from enum import Enum
from typing import Final, Literal

from numpy import int64
from pandas import DataFrame, Series


class DetectorDataframeColumn(Enum):
    CALIB_ENERGY = "CALIB_ENERGY"
    RECALIBRATED_ENERGY = "RECALIB_ENERGY"
    ENERGYSHORT = "ENERGYSHORT"
    ENERGY = "ENERGY"
    PULSE_HEIGHT = "PULSE_HEIGHT"
    TIMETAG = "TIMETAG"
    FLAGS = "FLAGS"
    PSD = "tail / total"
    TIME_HOURS = "TIMETAG_HOURS"
    EVENT_TIME = "EVENT_TIME"
    EVENT_TIME_PS = "EVENT_TIME_PS"
    NEUTRON_CLASS = "NASA"
    NEUTRON_RECALC_CLASS = "NASA_RECALC"
    NEW_N_CLASS = "IS_NEUTRON"
    DEAD_TIME = "DEAD_TIME"
    TIME_STAMP_ROLLOVER = "TIME_STAMP_ROLLOVER"
    TIME_STAMP_RESET = "TIME_STAMP_RESET"
    FAKE_EVENT = "FAKE_EVENT"
    MEMORY_FULL = "MEMORY_FULL"
    TRIGGER_LOST = "TRIGGER_LOST"
    N_TRIGGERS_LOST = "N_TRIGGERS_LOST"
    EVENT_OR_TRAP_SATURATING = "EVENT_OR_TRAP_SATURATING"
    MAX_TRIGGERS_COUNTED = "MAX_TRIGGERS_COUNTED"
    BOARD_WAS_BUSY = "BOARD_WAS_BUSY"
    INPUT_SATURATING = "INPUT_SATURATING"
    N_TRIGGERS_COUNTED = "N_TRIGGERS_COUNTED"
    NO_TIME_CORRELATION_MATCH = "NO_TIME_CORRELATION_MATCH"
    FINE_TIME_STAMP = "FINE_TIME_STAMP"
    PILEUP = "PILEUP"
    FAKE_EVENT_PLL_LOCK_LOSS = "FAKE_EVENT_PLL_LOCK_LOSS"
    FAKE_EVENT_OVERTEMP = "FAKE_EVENT_OVERTEMP"
    FAKE_EVENT_ADC_SHUTDOWN = "FAKE_EVENT_ADC_SHUTDOWN"


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
WITH_FLAGS_COLUMNS: Final = [
    DetectorDataframeColumn.CALIB_ENERGY,
    DetectorDataframeColumn.ENERGYSHORT,
    DetectorDataframeColumn.ENERGY,
    DetectorDataframeColumn.TIMETAG,
    DetectorDataframeColumn.FLAGS,
]
WITH_FLAGS_COL_NAMES: Final = [e.value for e in WITH_FLAGS_COLUMNS]
with_flags_col_types = {
    DetectorDataframeColumn.CALIB_ENERGY: float,
    DetectorDataframeColumn.ENERGYSHORT: int,
    DetectorDataframeColumn.ENERGY: int,
    DetectorDataframeColumn.TIMETAG: int64,
    DetectorDataframeColumn.FLAGS: str
}
WITH_FLAGS_COL_TYPES: Final = {k.value: v for k, v in with_flags_col_types.items()}
INDIVIDUAL_FLAG_COLUMNS: Final = [
    DetectorDataframeColumn.DEAD_TIME,
    DetectorDataframeColumn.TIME_STAMP_ROLLOVER,
    DetectorDataframeColumn.TIME_STAMP_RESET,
    DetectorDataframeColumn.FAKE_EVENT,
    DetectorDataframeColumn.MEMORY_FULL,
    DetectorDataframeColumn.TRIGGER_LOST,
    DetectorDataframeColumn.N_TRIGGERS_LOST,
    DetectorDataframeColumn.EVENT_OR_TRAP_SATURATING,
    DetectorDataframeColumn.MAX_TRIGGERS_COUNTED,
    DetectorDataframeColumn.BOARD_WAS_BUSY,
    DetectorDataframeColumn.INPUT_SATURATING,
    DetectorDataframeColumn.N_TRIGGERS_COUNTED,
    DetectorDataframeColumn.NO_TIME_CORRELATION_MATCH,
    DetectorDataframeColumn.FINE_TIME_STAMP,
    DetectorDataframeColumn.PILEUP,
    DetectorDataframeColumn.FAKE_EVENT_PLL_LOCK_LOSS,
    DetectorDataframeColumn.FAKE_EVENT_OVERTEMP,
    DetectorDataframeColumn.FAKE_EVENT_ADC_SHUTDOWN,
]
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
    return df[col.value]
