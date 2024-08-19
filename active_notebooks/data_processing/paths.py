"""This module is responsible for getting paths to experimental data."""
from pathlib import Path

from data_processing.dot_env import config

maybe_unconverted = config.get("UNCONVERTED_DATA_FOLDER")
maybe_input = config.get("INPUT_DATA_FOLDER")
maybe_output = config.get("OUTPUT_DATA_FOLDER")
UNCONVERTED_DATA_FOLDER = Path(
    maybe_unconverted if maybe_unconverted is not None else "fix_unconverted"
)
INPUT_DATA_FOLDER = Path(maybe_input if maybe_input is not None else "fix_input")
OUTPUT_DATA_FOLDER = Path(maybe_output if maybe_output is not None else "fix_output")


def get_exp_root(experiment_name: str) -> Path:
    """Gets path to experiment root folder.

    :param experiment_name: Name of experiment
    :type experiment_name: str
    :return: Path to experiment root folder
    :rtype: Path
    """
    return INPUT_DATA_FOLDER / experiment_name


def get_exp_unconverted_root(experiment_name: str) -> Path:
    """Gets path to experiment root folder inside Unconverted Data folder

    :param experiment_name: Name of experiment
    :type experiment_name: str
    :return: Path to experiment unconverted data root folder
    :rtype: Path
    """
    return UNCONVERTED_DATA_FOLDER / experiment_name


def get_parq_root(experiment_name: str) -> Path:
    """Gets path to folder with experiment's PSD data in Parquet format.

    :param experiment_name: Name of experiment
    :type experiment_name: str
    :return: Path to experiment PSD data folder
    :rtype: Path
    """
    return get_exp_root(experiment_name) / "processed_data/unfiltered/psd"


def get_unfiltered_csv_root(experiment_name: str) -> Path:
    """Gets path to folder with unfiltered, unconverted CSV files.
    These files should be produced by the CAEN Compass software after pileup rejection.

    :param experiment_name: Name of experiment
    :type experiment_name: str
    :return: Path to experiment's unconverted "unfiltered" data
    :rtype: Path
    """
    return get_exp_unconverted_root(experiment_name) / "UNFILTERED"


def get_raw_csv_root(experiment_name: str) -> Path:
    """Gets path to folder with raw, unconverted CSV files.
    These files should be produced by the CAEN Compass software, with no other processing.

    :param experiment_name: Name of experiment
    :type experiment_name: str
    :return: Path to experiment's unconverted "raw" data
    :rtype: Path
    """
    return get_exp_unconverted_root(experiment_name) / "RAW"


def get_report_root(experiment_name: str) -> Path:
    """Gets path to folder where all data processing outputs go

    :param experiment_name: Name of experiment
    :type experiment_name: str
    :return: Path to experiment's report output folder
    :rtype: Path
    """
    report_root = OUTPUT_DATA_FOLDER / experiment_name
    report_root.mkdir(parents=True, exist_ok=True)
    return report_root


def get_signals_root(experiment_name: str) -> Path:
    """Gets path to folder with experiment's signal trace data in Parquet form

    :param experiment_name: Name of experiment
    :type experiment_name: str
    :return: Path to experiment signal trace data folder
    :rtype: Path
    """
    return get_exp_root(experiment_name) / "processed_data/unfiltered/signals"


def get_reactor_data_root(experiment_name: str) -> Path:
    """Gets path to folder with experiment's reactor data

    :param experiment_name: Name of experiment
    :type experiment_name: str
    :return: Path to experiment reactor data folder
    :rtype: Path
    """
    return get_exp_root(experiment_name) / "reactor_data"
