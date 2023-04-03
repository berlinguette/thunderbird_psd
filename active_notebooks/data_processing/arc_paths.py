from pathlib import Path

alloc_code = 'st-cberling-1'
PROJECT_FOLDER = Path('/arc/project') / alloc_code
SCRATCH_FOLDER = Path('/scratch') / alloc_code
INPUT_DATA_FOLDER = PROJECT_FOLDER / 'input_data'
OUTPUT_DATA_FOLDER = SCRATCH_FOLDER / 'output_data'

def get_parq_root(experiment_name: str) -> Path:
    return INPUT_DATA_FOLDER.joinpath(experiment_name, 
                                      'processed_data/unfiltered/psd')

def get_report_root(experiment_name: str) -> Path:
    report_root = OUTPUT_DATA_FOLDER / 'analysis' / experiment_name
    report_root.mkdir(parents=True, exist_ok=True)
    return report_root

def get_signals_root(experiment_name: str) -> Path:
    return INPUT_DATA_FOLDER.joinpath(experiment_name,
                                      'processed_data/unfiltered/signals')
