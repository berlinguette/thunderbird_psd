import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Dict

import pandas as pd
from scipy.io import loadmat

from setup_logger import setup_logger

from get_limited_files import get_limited_files

from setup_logger import (
    cleanup_logger,
    setup_logger


def get_data_tuple(file_path, label):
    data = loadmat(file_path)
    data = data["A"].flatten()
    return data, label


def elapsed_time(t1):
    return time.perf_counter() - t1


def parquetize_folder(
    directory: Path,
    destination: Path,
    max_workers: int,
    logger: Logger
) -> pd.DataFrame:
    file_paths = [f for f in directory.iterdir()]
    end_folder_name = directory.name

    logger.info(f"Converting {end_folder_name}...")

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
    logger.info(f"File {end_folder_name} Completed")


def parquetize_directory(directory: Path, destination: Path, config: Dict):
    # folder_paths = [f for f in directory.iterdir()]
    # folder_paths = folder_paths[0:num_folders]
    logger = setup_logger('parquetizer', directory.parent.parent)
    t1 = time.perf_counter()
    num_folders = config.get('files_limit')
    parquet_tasks = config.get('parquet_tasks', 0)
    parquet_files = config.get('parquet_files', 0)
    folder_paths = [f for f in get_limited_files(directory, num_folders)]

    with ProcessPoolExecutor(max_workers=parquet_tasks) as process_executor:
        process_executor.map(
            parquetize_folder,
            folder_paths,
            repeat(destination),
            repeat(parquet_files))
    cleanup_logger(logger)


if __name__ == "__main__":
    directory = Path("sample_dataset/raw_data/mat/")
    out_directory = Path("sample_dataset/processed_data/parquets/")

    parquetize_directory(directory, out_directory)
