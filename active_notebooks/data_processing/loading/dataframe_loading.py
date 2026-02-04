import re
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl
from pyarrow.parquet import read_schema
from data_processing.dataframe_validation import (
    STARTING_COL_NAMES,
    STARTING_COL_TYPES,
    WITH_FLAGS_COL_NAMES,
    WITH_FLAGS_COL_TYPES,
    STARTING_PH_COL_NAMES,
    STARTING_PH_COL_TYPES,
    WITH_FLAGS_PH_COL_NAMES,
    WITH_FLAGS_PH_COL_TYPES,
    WITH_FLAGS_PH_SCHEMA,
    DetectorDataframeColumn,
    get_df_col,
)

from data_processing import paths

END_NUMBER_PATTERN = r"^(.*_)(\d+)$"
SAMPLES_COL_NAME = "SAMPLES"
DELIMITER = ";"


def load_parquet_psd(experiment_name: str, with_flags: bool = False) -> tuple[pd.DataFrame, pl.LazyFrame]:
    df = _load_parquet_psd_pandas(experiment_name, with_flags=with_flags)
    lf = _load_parquet_psd_polars(experiment_name, with_flags=with_flags)
    return df, lf

def _load_parquet_psd_pandas(experiment_name: str, with_flags: bool = False) -> pd.DataFrame:
    base_col_names = WITH_FLAGS_COL_NAMES if with_flags else STARTING_COL_NAMES
    ph_col_names = WITH_FLAGS_PH_COL_NAMES if with_flags else STARTING_PH_COL_NAMES
    psd_folder = paths.get_parq_root(experiment_name)

    # TODO retry on PermissionError
    col_names = ph_col_names if _check_cols_exist(psd_folder, ph_col_names) else base_col_names
    psd_df = (
        pd.read_parquet(psd_folder, columns=col_names)
        .pipe(_process_psd_data_pandas, with_flags=with_flags)
        .dropna()
    )
    if with_flags:
        flags_df = _process_caen_flag_series_pandas(psd_df[DetectorDataframeColumn.FLAGS.value])
        psd_df = psd_df.join(flags_df)
    return psd_df


def _load_parquet_psd_polars(experiment_name: str, with_flags: bool = False) -> pl.LazyFrame:
    # col_types = WITH_FLAGS_COL_TYPES if with_flags else STARTING_COL_TYPES
    # schema = pl.Schema()
    psd_folder = paths.get_parq_root(experiment_name)

    psd_lf = (
        pl.scan_parquet(psd_folder, retries=3)
        .pipe_with_schema(_select_psd_cols_polars(with_flags))
        .pipe_with_schema(_process_psd_data_polars)
        # .drop_nans().drop_nulls()
    )
    # if with_flags:
    #     # flags_lf = _process_caen_flag_series_polars(psd_lf.select(DetectorDataframeColumn.FLAGS.value))
    #     flags_lf = (
    #         psd_lf.select(DetectorDataframeColumn.FLAGS.value)
    #         .pipe_with_schema(_process_caen_flag_series_polars)
    #     )
    #     psd_lf = pl.concat([psd_lf, flags_lf])
    
    return psd_lf


def load_parquet_signals(experiment_name: str) -> pd.DataFrame:
    signals_folder = paths.get_signals_root(experiment_name)

    # TODO retry on PermissionError
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
    if get_psd:
        full_df = _process_psd_data_pandas(full_df, with_flags=get_flags)
    if get_flags:
        flags_df = _process_caen_flag_series_pandas(full_df[DetectorDataframeColumn.FLAGS.value])
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


def _process_psd_data_pandas(full_df: pd.DataFrame, with_flags: bool = False) -> pd.DataFrame:
    if DetectorDataframeColumn.PULSE_HEIGHT.value in full_df.columns:
        col_types = WITH_FLAGS_PH_COL_TYPES if with_flags else STARTING_PH_COL_TYPES
    else:
        col_types = WITH_FLAGS_COL_TYPES if with_flags else STARTING_COL_TYPES
    return (
        full_df.astype(col_types)
        .pipe(_add_psd_col)
        .pipe(_filter_for_valid_psd)
    )


def _process_psd_data_polars(full_lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
    new_schema = WITH_FLAGS_PH_SCHEMA
    casts = [(name, new_schema.get(name)) for name, _ in schema.items()]
    casts = [pl.col(name).cast(new_type) for name, new_type in casts if new_type is not None]
    lf = (
        full_lf.with_columns(casts)
        .pipe(_add_psd_col_polars)
        .pipe(_filter_for_valid_psd_polars)
    )
    return lf

def _select_psd_cols_polars(with_flags: bool) -> Callable[[pl.LazyFrame, pl.Schema], pl.LazyFrame]:
    def col_selector(full_lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
        ph_present = DetectorDataframeColumn.PULSE_HEIGHT.value in schema.keys()
        # flags_present = DetectorDataframeColumn.FLAGS.value in schema.keys()
        match (ph_present, with_flags):
            case (True, True):
                cols = WITH_FLAGS_PH_COL_NAMES
            case (True, False):
                cols = STARTING_PH_COL_NAMES
            case (False, True):
                cols = WITH_FLAGS_COL_NAMES
            case (False, False):
                cols = STARTING_COL_NAMES
        lf = full_lf.select(cols)
        return lf
    
    return col_selector


def _add_psd_col(df: pd.DataFrame) -> pd.DataFrame:
    energy_col = get_df_col(df, DetectorDataframeColumn.ENERGY)
    short_col = get_df_col(df, DetectorDataframeColumn.ENERGYSHORT)
    psd_col = (energy_col - short_col) / energy_col
    df.loc[:, DetectorDataframeColumn.PSD.value] = psd_col
    return df


def _add_psd_col_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    def calc_psd(energy: int, energyshort: int) -> float:
        return (energy - energyshort) / energy
    
    energy_col_name = DetectorDataframeColumn.ENERGY.value
    energyshort_col_name = DetectorDataframeColumn.ENERGYSHORT.value
    
    result = lf.with_columns(
        pl.struct(
            [energy_col_name, energyshort_col_name]
        ).map_elements(
            lambda s: calc_psd(s[energy_col_name], s[energyshort_col_name]),
            return_dtype=pl.Float64
        ).alias("tail / total")
    )
    return result


def _filter_for_valid_psd(df: pd.DataFrame) -> pd.DataFrame:
    psd_col = get_df_col(df, DetectorDataframeColumn.PSD)
    valid_psd = psd_col.between(0, 0.5)
    return df.loc[valid_psd]


def _filter_for_valid_psd_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col(DetectorDataframeColumn.PSD.value).is_between(0, 0.5))


def _check_cols_exist(psd_folder: Path, columns: list[str]) -> bool:
    if not psd_folder.is_dir():
        raise ValueError("Provided path was not folder:", psd_folder.name)
    parq_files = sorted(
        [file for file in psd_folder.iterdir() if file.suffix == ".parquet" and "000"],
        key=lambda x: x.name
    )
    if len(parq_files) == 0:
        raise ValueError("Provided folder was empty:", psd_folder.name)
    pq_col_set = set(read_schema(parq_files[0]).names)
    return pq_col_set.issuperset(set(columns))

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

def _process_caen_flag_series_pandas(flag_series: pd.Series) -> pd.DataFrame:
    int_flags_series = flag_series.apply(_ensure_flags_is_int)
    flags_df = pd.DataFrame()
    for flag_col, flag_bitmask in flag_values.items():
        flags_df.loc[:, flag_col.value] = _is_bit_set(int_flags_series, flag_bitmask)
    return flags_df

def _process_caen_flag_series_polars(flag_lf: pl.LazyFrame) -> pl.LazyFrame:
    flag_col_name = DetectorDataframeColumn.FLAGS.value
    flags = flag_lf.with_columns(
        pl.col(flag_col_name).cast(pl.Int64)
    )

    col_exprs = [
        ((pl.col(flag_col_name) & bitmask) != 0).alias(flag_name)
        for flag_name, bitmask in flag_values.items()
    ]
    flags = flags.select(col_exprs)
    return flags