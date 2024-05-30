from enum import Enum
from typing import Final

from numpy import int64
from pandas import DataFrame, Series


class DataframeColumn(Enum):
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
    DataframeColumn.CALIB_ENERGY,
    DataframeColumn.ENERGYSHORT,
    DataframeColumn.ENERGY,
    DataframeColumn.TIMETAG,
]
STARTING_COL_NAMES: Final = [e.value for e in STARTING_COLUMNS]
col_types = {
    DataframeColumn.CALIB_ENERGY: float,
    DataframeColumn.ENERGYSHORT: int,
    DataframeColumn.ENERGY: int,
    DataframeColumn.TIMETAG: int64,
}
STARTING_COL_TYPES: Final = {k.value: v for k, v in col_types.items()}


def get_df_col(df: DataFrame, col: DataframeColumn) -> Series:
    return df[col.value]
