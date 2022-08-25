import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Dict

import pandas as pd
from scipy.io import loadmat
from tqdm.contrib.concurrent import process_map

from logging_helpers.setup_logger import (cleanup_logger, setup_logger,
                                          message_debug, message_info)
from utilities.get_limited_files import get_limited_files

logger = logging.getLogger('parquetizer')


def get_data_tuple(file_path, label):
    data = loadmat(file_path)
    data = data["A"].flatten()
    return data, label


def elapsed_time(t1):
    return time.perf_counter() - t1


def parquetize_folder(
    directory: Path,
    destination: Path,
    max_workers: int
) -> pd.DataFrame:
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
        serial_number = re.search("\_(\d\d\d\d)(.mat)", file_name)
        return f"{batch_number}s{serial_number.group(1)}"

    labels = map(get_label, file_names)

    with ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
        result = thread_executor.map(get_data_tuple, file_paths, labels)

    labelled_data = list(zip(*result))

    df = pd.DataFrame(labelled_data[0], index=labelled_data[1])
    df.columns = df.columns.astype(str)

    df.to_parquet(str(destination / f"{end_folder_name}.parquet"))
    message_debug(f"Folder {end_folder_name} Completed",
                   new_logger, on_screen=False)


def parquetize_directory(directory: Path, destination: Path, config: Dict):
    # folder_paths = [f for f in directory.iterdir()]
    # folder_paths = folder_paths[0:num_folders]
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
        desc='Matlab Folders', unit='folder', total=len(folder_paths)
    )

    message_info(f'Processed {len(list(results))} folders', logger)
    message_debug(
        f"Elapsed Time: {elapsed_time(t1):.4f} s", logger, on_screen=False)
    cleanup_logger(logger)


if __name__ == "__main__":
    from configuration import get_configuration

    directory = Path("sample_dataset/raw_data/mat/")
    out_directory = Path("sample_dataset/processed_data/parquets/")

    config = get_configuration({})
    parquetize_directory(directory, out_directory, config)
