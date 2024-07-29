import re
from pathlib import Path
from typing import Literal

import pandas as pd
from data_processing.dataframe_validation import (
    STARTING_COL_NAMES,
    STARTING_COL_TYPES,
    DetectorDataframeColumn,
    get_df_col,
)

from active_notebooks.data_processing import paths

END_NUMBER_PATTERN = r"^(.*_)(\d+)$"
SAMPLES_COL_NAME = "SAMPLES"
DELIMITER = ";"


def load_parquet_psd(experiment_name: str) -> pd.DataFrame:
    psd_folder = paths.get_parq_root(experiment_name)

    # Load PSD data to "psd_report" DataFrame
    psd_df = (
        pd.read_parquet(psd_folder, columns=STARTING_COL_NAMES)
        .pipe(_process_psd_data)
        .dropna()
    )
    return psd_df


def load_parquet_signals(experiment_name: str) -> pd.DataFrame:
    signals_folder = paths.get_signals_root(experiment_name)

    signals_data = pd.read_parquet(signals_folder)
    return signals_data


def load_caen_csvs(
    experiment_name: str,
    wanted_data: Literal["psd", "signals", "all"] = "psd",
    raw: bool = False,
) -> pd.DataFrame:
    """Loads experiment data from unconverted CAEN Compass CSV files.
    It is expected that these files are in the "1-Unconverted" folder as exported by Compass.

    :param experiment_name: Name of experiment
    :type experiment_name: str
    :param wanted_data: What kind of data to load (only PSD related, only signal related, or all data), defaults to "psd"
    :type wanted_data: Literal['psd', 'signals', 'all'], optional
    :param raw: Whether to load raw (True) or unfiltered (False) data, defaults to False
    :type raw: bool, optional
    :raises ValueError: when the given folder has no CSV files
    :return: dataframe of all desired data. If PSD data is desired, a PSD column will be added, and only rows where PSD is within 0-0.5 will be included
    :rtype: pd.DataFrame
    """    
    csv_folder_path = (
        paths.get_raw_csv_root(experiment_name)
        if raw
        else paths.get_unfiltered_csv_root(experiment_name)
    )
    source_files = [f for f in csv_folder_path.iterdir() if f.suffix == ".csv"]
    if len(source_files) < 1:
        raise ValueError("Folder has no csv files")

    headers, total_cols = _get_info_from_first_file(source_files)
    if SAMPLES_COL_NAME in headers:
        psd_cols, signal_cols = _get_split_cols(headers, total_cols)
    else:
        psd_cols = headers
        signal_cols = []
    names, usecols = _get_names_and_usecols(wanted_data, psd_cols, signal_cols)

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
    if wanted_data in ["psd", "all"]:
        full_df = _process_psd_data(full_df)
    if wanted_data in ["signals", "all"]:
        signals_dtypes = {k: "int32" for k in signal_cols}
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


def _get_names_and_usecols(
    wanted_data: Literal["psd", "signals", "all"],
    psd_cols: list[str],
    signal_cols: list[str],
):
    if wanted_data == "psd":
        names = psd_cols
        usecols = STARTING_COL_NAMES
    elif wanted_data == "signals":
        if len(signal_cols) < 1:
            raise ValueError("Signals columns were not present in data files")
        names = signal_cols
        usecols = signal_cols
    else:
        names = psd_cols + signal_cols
        usecols = STARTING_COL_NAMES + signal_cols
    return names, usecols


def _process_psd_data(full_df: pd.DataFrame) -> pd.DataFrame:
    return (
        full_df.astype(STARTING_COL_TYPES)
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
