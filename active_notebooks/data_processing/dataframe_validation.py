from enum import Enum
from typing import Final
from numpy import int64


class PsdDfColumn(Enum):
    CALIB_ENERGY = "CALIB_ENERGY"
    ENERGYSHORT = "ENERGYSHORT"
    ENERGY = "ENERGY"
    TIMETAG = "TIMETAG"
    PSD = "tail / total"
    TIME_HOURS = "TIMETAG_HOURS"
    EVENT_TIME = "EVENT_TIME"
    EVENT_TIME_PS = "EVENT_TIME_PS"


STARTING_COLUMNS: Final = [
    PsdDfColumn.CALIB_ENERGY,
    PsdDfColumn.ENERGYSHORT,
    PsdDfColumn.ENERGY,
    PsdDfColumn.TIMETAG
]
STARTING_COL_NAMES: Final = [e.value for e in STARTING_COLUMNS]
STARTING_COL_TYPES: Final = {
    PsdDfColumn.CALIB_ENERGY: float,
    PsdDfColumn.ENERGYSHORT: int,
    PsdDfColumn.ENERGY: int,
    PsdDfColumn.TIMETAG: int64
}
