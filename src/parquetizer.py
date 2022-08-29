import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

import pandas as pd
from scipy.io import loadmat
from tqdm.contrib.concurrent import process_map

from logging_helpers.setup_logger import (cleanup_logger, message_debug,
                                          message_info, setup_logger)
from utilities.constants import BAR_FORMAT
from utilities.get_limited_files import get_limited_files

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger('parquetizer')


def _get_data_tuple(file_path: Path, label: str) -> Tuple[ndarray, str]:
    """Gets labelled data from a Matlab file

    Parameters
    ----------
    file_path : Path
        path to Matlab file
    label : str
        File label

    Returns
    -------
    Tuple[ndarray, str]
        tuple of (Matlab file data, label)
    """
    data = loadmat(file_path)
    data = data["A"].flatten()
    return data, label


def elapsed_time(t1: float) -> float:
    """Determines the time elapsed in seconds 

    Parameters
    ----------
    t1 : float
        starting time, as produced by time.perf_counter()

    Returns
    -------
    float
        the time elapsed from t1 to now
    """
    return time.perf_counter() - t1


def parquetize_folder(
    directory: Path,
    destination: Path,
    max_workers: int
):
    """Converts Matlab signal folder to a Parquet file

    - The signal folder represents one experimental data buffer, 
      and must have the same name as the PSData file that produced it
    - The folder must contain one or more Matlab files, one per neutron signal in the buffer
      - Each of these Matlab files must be named in the format 
        "{folder_name}-{signal_number}.mat", where signal_number is padded to 4 digits
    - The generated Parquet files are named in the format "{folder_name}.parquet"

    Parameters
    ----------
    directory : Path
        The source folder for one experimental data buffer
    destination : Path
        The destination directory for the produced Parquet file
    max_workers : int
        Maximum number of concurrent Matlab files loading
    """
    file_paths = [f for f in directory.iterdir()]
    end_folder_name = directory.name

    # TODO allow on-screen when tqdm team fixes process pool issues
    # setup_logger(f'pq-{end_folder_name}', directory.parent.parent)
    # each process needs own separate logger in processpool
    new_logger = logging.getLogger(f'proc-{end_folder_name}')
    # we're in sample_dataset/raw_data/mat, one folder deeper than usual
    setup_logger(new_logger, directory.parent.parent.parent)
    message_debug(f"Converting {end_folder_name}...",
                  new_logger, on_screen=False)

    batch_number = f"b{end_folder_name.split('-')[1]}"
    file_names = map(lambda x: x.name, file_paths)

    def get_label(file_name: str, batch_number: str = batch_number):
        serial_number = re.search("\_(\d+)(.mat)", file_name)
        return f"{batch_number}s{serial_number.group(1)}"

    labels = map(get_label, file_names)

    with ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
        result = thread_executor.map(_get_data_tuple, file_paths, labels)

    labelled_data = list(zip(*result))

    df = pd.DataFrame(labelled_data[0], index=labelled_data[1])
    df.columns = df.columns.astype(str)

    df.to_parquet(str(destination / f"{end_folder_name}.parquet"))
    message_debug(f"Folder {end_folder_name} Completed",
                  new_logger, on_screen=False)


def parquetize_directory(directory: Path, destination: Path, config: Dict):
    """Converts each Matlab signal folder in this directory to a Parquet file

    - The source directory represents the complete set of neutron data.
    - This data must contain one or more folders, each representing one buffer
      - Each of these folders should have the same name as the PSData file that produced it
    - Each of those folders contain one or more Matlab files, one per neutron signal in the buffer
      - Each of these Matlab files must be named in the format 
        "{folder_name}-{signal_number}.mat", where signal_number is padded to 4 digits
    - The generated Parquet files are named in the format "{folder_name}.parquet"

    Parameters
    ----------
    directory : Path
        The source directory, containing data for all experiment buffers
    destination : Path
        The destination directory for the produced Parquet files
    config : Dict
        Configuration data. See configuration.py for more info
    """
    setup_logger(logger, directory.parent.parent)
    t1 = time.perf_counter()

    num_folders = config.get('files_limit')
    parquet_tasks = config.get('parquet_tasks', 0)
    parquet_files = config.get('parquet_files', 0)
    folder_paths = [f for f in get_limited_files(directory, num_folders)]

    results = process_map(
        parquetize_folder,
        folder_paths,
        repeat(destination),
        repeat(parquet_files),
        max_workers=parquet_tasks,
        desc='Matlab Folders',
        unit='dir',
        total=len(folder_paths),
        bar_format=BAR_FORMAT
    )

    message_info(f'Processed {len(list(results))} folders', logger)
    message_debug(
        f"Elapsed Time: {elapsed_time(t1):.4f} s", logger, on_screen=False)
    cleanup_logger(logger)


if __name__ == "__main__":
    from configuration import get_configuration

    directory = Path("sample_datasets/20220824_CERC_background/raw_data/mat")
    out_directory = Path(
        "sample_datasets/20220824_CERC_background/raw_data/parquet")

    config = get_configuration({})
    parquetize_directory(directory, out_directory, config)
