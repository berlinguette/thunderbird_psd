import re
from pathlib import Path
from typing import Literal

import pandas as pd
from data_processing.dataframe_validation import (
    STARTING_COL_NAMES,
    STARTING_COL_TYPES,
    WITH_FLAGS_COL_NAMES,
    WITH_FLAGS_COL_TYPES,
    DetectorDataframeColumn,
    get_df_col,
)

from data_processing import paths

END_NUMBER_PATTERN = r"^(.*_)(\d+)$"
SAMPLES_COL_NAME = "SAMPLES"
DELIMITER = ";"


def load_parquet_psd(experiment_name: str, with_flags: bool = False) -> pd.DataFrame:
    col_names = WITH_FLAGS_COL_NAMES if with_flags else STARTING_COL_NAMES
    psd_folder = paths.get_parq_root(experiment_name)

    # Load PSD data to "psd_report" DataFrame
    psd_df = (
        pd.read_parquet(psd_folder, columns=col_names)
        .pipe(_process_psd_data, with_flags=with_flags)
        .dropna()
    )
    if with_flags:
        flags_df = _process_caen_flag_series(psd_df[DetectorDataframeColumn.FLAGS.value])
        psd_df = psd_df.join(flags_df)
    return psd_df


def load_parquet_signals(experiment_name: str) -> pd.DataFrame:
    signals_folder = paths.get_signals_root(experiment_name)

    signals_data = pd.read_parquet(signals_folder)
    return signals_data


def load_caen_csvs(
    experiment_name: str,
    get_psd: bool = True,
    get_flags: bool = False,
    get_signals: bool = False,
    raw: bool = False,
) -> pd.DataFrame:
    """Loads experiment data from unconverted CAEN Compass CSV files.
    It is expected that these files are in the "1-Unconverted" folder as exported by Compass.

    :param experiment_name: Name of experiment
    :type experiment_name: str
    :param get_psd: Whether to return PSD columns
    :type get_psd: bool
    :param get_flags: Whether to return CAEN signal flag data
    :type get_flags: bool
    :param get_signals: Whether to return signal trace data
    :type get_signals: bool
    :param raw: Whether to load raw (True) or unfiltered (False) data, defaults to False
    :type raw: bool, optional
    :raises ValueError: when the given folder has no CSV files
    :raises ValueError: when get_flags is True but get_psd is False
    :raises ValueError: when both get_psd and get_signals are false
    :return: dataframe of all desired data. If PSD data is desired, a PSD column will be added, and only rows where PSD is within 0-0.5 will be included
    :rtype: pd.DataFrame
    """
    if not (get_psd or get_signals):
        raise ValueError("Ether PSD or signals columns (or both) must be returned")
    if get_flags and (not get_psd):
        raise ValueError("Flags columns cannot be returned without PSD columns")
    
    csv_folder_path = (
        paths.get_raw_csv_root(experiment_name)
        if raw
        else paths.get_unfiltered_csv_root(experiment_name)
    )
    source_files = [f for f in csv_folder_path.iterdir() if f.suffix.lower() == ".csv"]
    if len(source_files) < 1:
        raise ValueError("Folder has no csv files")

    headers, total_cols = _get_info_from_first_file(source_files)
    if SAMPLES_COL_NAME in headers:
        psd_cols, signal_cols = _get_split_cols(headers, total_cols)
        names = psd_cols + signal_cols
    else:
        names = headers
        signal_cols = []
    usecols = _get_usecols(signal_cols, get_psd, get_flags, get_signals)

    dfs = []
    for source_file in source_files:
        header = 0 if _has_header_line(source_file) else None
        df = pd.read_csv(
            source_file,
            sep=DELIMITER,
            header=header,
            names=names,
            usecols=usecols,
            dtype=str,
            on_bad_lines="skip",  # TODO test "warning" option
        )
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    # if wanted_data in ["psd_flags", "all_flags"]:
    #     full_df = _process_psd_data(full_df, with_flags=True)
    # if wanted_data in ["psd", "all"]:
    #     full_df = _process_psd_data(full_df)
    # if wanted_data in ["signals", "all", "all_flags"]:
    #     signals_dtypes = {k: "int32" for k in signal_cols}
    #     full_df = full_df.astype(signals_dtypes)
    if get_psd:
        full_df = _process_psd_data(full_df, with_flags=get_flags)
    if get_flags:
        flags_df = _process_caen_flag_series(full_df[DetectorDataframeColumn.FLAGS.value])
        full_df = full_df.join(flags_df)
    if get_signals:
        signals_dtypes = {k: "uint32" for k in signal_cols}
        full_df = full_df.astype(signals_dtypes)

    # TODO remove duplicate timetags (after benchmarking)
    full_df = full_df.dropna()

    return full_df


def _get_info_from_first_file(source_files: list[Path]):
    first_file_matches = [
        f for f in source_files if re.match(END_NUMBER_PATTERN, f.stem) is None
    ]
    if len(source_files) < 1:
        raise ValueError("Could not find csv file with headers")
    first_file = first_file_matches[0]

    with open(first_file, "r") as openfile:
        header_line = openfile.readline()
        data_line = openfile.readline()

    headers = header_line.strip().split(DELIMITER)
    data_sample = data_line.strip().split(DELIMITER)
    total_cols = len(data_sample)
    return headers, total_cols


def _get_split_cols(headers: list[str], total_cols: int) -> tuple[list[str], list[str]]:
    psd_cols = [col for col in headers if col != SAMPLES_COL_NAME]
    signal_cols = [str(n) for n in range(total_cols - len(psd_cols))]
    return psd_cols, signal_cols


def _has_header_line(source_file: Path) -> bool:
    return re.match(END_NUMBER_PATTERN, source_file.stem) is None


def _get_usecols(
    # wanted_data: Literal["psd", "psd_flags", "signals", "all"],
    signal_cols: list[str],
    psd: bool,
    flags: bool,
    signals: bool
) -> list[str]:
    # if wanted_data == "psd":
    usecols = []
    if psd:
        usecols = WITH_FLAGS_COL_NAMES if flags else STARTING_COL_NAMES
    # elif wanted_data == "signals":
    if signals:
        if len(signal_cols) < 1:
            raise ValueError("Signals columns were not present in data files")
        usecols += signal_cols
    return usecols


def _process_psd_data(full_df: pd.DataFrame, with_flags: bool = False) -> pd.DataFrame:
    col_types = WITH_FLAGS_COL_TYPES if with_flags else STARTING_COL_TYPES
    return (
        full_df.astype(col_types)
        .pipe(_add_psd_col)
        .pipe(_filter_for_valid_psd)
    )


def _add_psd_col(df: pd.DataFrame) -> pd.DataFrame:
    energy_col = get_df_col(df, DetectorDataframeColumn.ENERGY)
    short_col = get_df_col(df, DetectorDataframeColumn.ENERGYSHORT)
    psd_col = (energy_col - short_col) / energy_col
    df[DetectorDataframeColumn.PSD.value] = psd_col
    return df


def _filter_for_valid_psd(df: pd.DataFrame) -> pd.DataFrame:
    psd_col = get_df_col(df, DetectorDataframeColumn.PSD)
    valid_psd = psd_col.between(0, 0.5)
    return df[valid_psd]

flag_values = {
    DetectorDataframeColumn.DEAD_TIME: 0x1,
    DetectorDataframeColumn.TIME_STAMP_ROLLOVER: 0x2,
    DetectorDataframeColumn.TIME_STAMP_RESET: 0x4,
    DetectorDataframeColumn.FAKE_EVENT: 0x8,
    DetectorDataframeColumn.MEMORY_FULL: 0x10,
    DetectorDataframeColumn.TRIGGER_LOST: 0x20,
    DetectorDataframeColumn.N_TRIGGERS_LOST: 0x40,
    DetectorDataframeColumn.EVENT_OR_TRAP_SATURATING: 0x80,
    DetectorDataframeColumn.MAX_TRIGGERS_COUNTED: 0x100,
    DetectorDataframeColumn.BOARD_WAS_BUSY: 0x200,
    DetectorDataframeColumn.INPUT_SATURATING: 0x400,
    DetectorDataframeColumn.N_TRIGGERS_COUNTED: 0x800,
    DetectorDataframeColumn.NO_TIME_CORRELATION_MATCH: 0x1000,
    DetectorDataframeColumn.FINE_TIME_STAMP: 0x4000,
    DetectorDataframeColumn.PILEUP: 0x8000,
    DetectorDataframeColumn.FAKE_EVENT_PLL_LOCK_LOSS: 0x80000,
    DetectorDataframeColumn.FAKE_EVENT_OVERTEMP: 0x100000,
    DetectorDataframeColumn.FAKE_EVENT_ADC_SHUTDOWN: 0x200000
    
}

def _is_bit_set(value: int, bitmask: int) -> bool:
    return (value & bitmask != 0)

def _ensure_flags_is_int(flags: int | str) -> int:
    if isinstance(flags, str):
        int_flags = int(flags, base=16)
    else:
        int_flags = flags
    return int_flags

def _process_caen_flag_series(flag_series: pd.Series) -> pd.DataFrame:
    int_flags_series = flag_series.apply(_ensure_flags_is_int)
    flags_df = pd.DataFrame()
    for flag_col, flag_bitmask in flag_values.items():
        flags_df[flag_col.value] = _is_bit_set(int_flags_series, flag_bitmask)
    return flags_df
