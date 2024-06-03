from enum import Enum
from typing import Final

from numpy import int64
from pandas import DataFrame, Series


class DetectorDataframeColumn(Enum):
    CALIB_ENERGY = "CALIB_ENERGY"
    ENERGYSHORT = "ENERGYSHORT"
    ENERGY = "ENERGY"
    TIMETAG = "TIMETAG"
    PSD = "tail / total"
    TIME_HOURS = "TIMETAG_HOURS"
    EVENT_TIME = "EVENT_TIME"
    EVENT_TIME_PS = "EVENT_TIME_PS"
    NEUTRON_CLASS = "NASA"
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


def get_df_col(df: DataFrame, col: DetectorDataframeColumn) -> Series:
    return df[col.value]
