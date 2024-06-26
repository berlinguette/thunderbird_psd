import re
from pathlib import Path

import pandas as pd
from data_processing.arc_paths import get_parq_root, get_signals_root
from data_processing.dataframe_validation import (
    STARTING_COL_NAMES,
    STARTING_COL_TYPES,
    DetectorDataframeColumn,
    get_df_col,
)
from data_processing.typing.dataclasses import experiment_neutron_data as end
from data_processing.typing.enums import Detector


def load_psd(experiment_ids: list[str]) -> end.MultipleExperimentData:
    return {exp_id: load_psd_for_experiment(exp_id) for exp_id in experiment_ids}


def load_psd_for_experiment(exp_id: str) -> end.ExperimentData:
    psd_folder = get_parq_root(exp_id)
    channel_subfolders = _get_channel_subfolders(psd_folder)

    if len(channel_subfolders) == 0:
        exp_detector_data = load_psd_channel(psd_folder)
        return {Detector.ZERO: [exp_detector_data]}
    else:
        return {
            Detector(channel_idx): [load_psd_channel(channel_folder)]
            for channel_idx, channel_folder in channel_subfolders
        }


def load_psd_channel(channel_folder: Path) -> end.ExperimentDetectorData:
    psd_df = pd.read_parquet(channel_folder, columns=STARTING_COL_NAMES)
    psd_df = psd_df.astype(STARTING_COL_TYPES)

    # Calculate PSD value as new column "tail / total"
    energy_col = get_df_col(psd_df, DetectorDataframeColumn.ENERGY)
    short_col = get_df_col(psd_df, DetectorDataframeColumn.ENERGYSHORT)
    psd_col = (energy_col - short_col) / energy_col
    psd_df[DetectorDataframeColumn.PSD.value] = psd_col

    valid_psd = psd_col.between(0, 0.5)
    psd_df = psd_df[valid_psd]
    psd_df = psd_df.dropna()

    raw_data = end.RawExperimentalData(neutron_detector=psd_df)
    return end.ExperimentDetectorData(raw_data=raw_data)


def load_signals(experiment_ids: list[str]) -> dict[str, dict[Detector, pd.DataFrame]]:
    return {exp_id: load_signals_for_experiment(exp_id) for exp_id in experiment_ids}


def load_signals_for_experiment(experiment_name: str) -> dict[Detector, pd.DataFrame]:
    signals_folder = get_signals_root(experiment_name)
    channel_subfolders = _get_channel_subfolders(signals_folder)

    if len(channel_subfolders) == 0:
        signals_df = load_signals_channel(signals_folder)
        return {Detector.ZERO: signals_df}
    else:
        return {
            Detector(channel_idx): load_signals_channel(channel_folder)
            for channel_idx, channel_folder in channel_subfolders
        }


def load_signals_channel(channel_folder: Path) -> pd.DataFrame:
    signals_data = pd.read_parquet(channel_folder)
    return signals_data


def _get_channel_subfolders(root_folder: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(r"ch(\d+)")
    subfolders = [subitem for subitem in root_folder.iterdir() if subitem.is_dir()]
    matches = [pattern.match(subfolder.name) for subfolder in subfolders]
    matching_subfolders = [
        (match, subfolder)
        for match, subfolder in zip(matches, subfolders)
        if match is not None
    ]
    indexed_subfolders = [
        (_convert_to_int_nullable(match.group(1)), subfolder)
        for match, subfolder in matching_subfolders
    ]
    indexed_subfolders = [
        (index, subfolder)
        for index, subfolder in indexed_subfolders
        if index is not None
    ]
    return indexed_subfolders


def _convert_to_int_nullable(x: str) -> int | None:
    try:
        return int(x)
    except ValueError:
        return None
