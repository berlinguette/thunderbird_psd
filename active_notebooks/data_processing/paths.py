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
    return INPUT_DATA_FOLDER / experiment_name


def get_exp_unconverted_root(experiment_name: str) -> Path:
    return UNCONVERTED_DATA_FOLDER / experiment_name


def get_parq_root(experiment_name: str) -> Path:
    return get_exp_root(experiment_name) / "processed_data/unfiltered/psd"


def get_unfiltered_csv_root(experiment_name: str) -> Path:
    return get_exp_unconverted_root(experiment_name) / "UNFILTERED"


def get_raw_csv_root(experiment_name: str) -> Path:
    return get_exp_unconverted_root(experiment_name) / "RAW"


def get_report_root(experiment_name: str) -> Path:
    report_root = OUTPUT_DATA_FOLDER / experiment_name
    report_root.mkdir(parents=True, exist_ok=True)
    return report_root


def get_signals_root(experiment_name: str) -> Path:
    return get_exp_root(experiment_name) / "processed_data/unfiltered/signals"


def get_reactor_data_root(experiment_name: str) -> Path:
    return get_exp_root(experiment_name) / "reactor_data"
