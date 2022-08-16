import re
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import pandas as pd
from scipy.io import loadmat


# Settings
MAX_PROCESSES = 4
MAX_THREADS_PER_PROCESS = 4


def get_data_tuple(file_path, label):
    data = loadmat(file_path)
    data = data["A"].flatten()
    return data, label


def elapsed_time(t1):
    return time.perf_counter() - t1


def parquetize_folder(directory: Path, destination: Path) -> pd.DataFrame:
    file_paths = [f for f in directory.iterdir()]
    end_folder_name = directory.name

    print(f"Converting {end_folder_name}...")

    batch_number = f"b{end_folder_name.split('-')[1]}"
    file_names = map(lambda x: x.name, file_paths)

    def get_label(file_name: str, batch_number: str = batch_number):
        serial_number = re.search("\_(\d\d\d\d)(.mat)", file_name)
        return f"{batch_number}s{serial_number.group(1)}"

    labels = map(get_label, file_names)

    with ThreadPoolExecutor(max_workers=MAX_THREADS_PER_PROCESS) as thread_executor:
        result = thread_executor.map(get_data_tuple, file_paths, labels)

    labelled_data = list(zip(*result))

    df = pd.DataFrame(labelled_data[0], index=labelled_data[1])
    df.columns = df.columns.astype(str)

    df.to_parquet(str(destination / f"{end_folder_name}.parquet"))
    print(f"File {end_folder_name} Completed")


def parquetize_directory(directory: Path, destination: Path, num_folders=10):
    folder_paths = [f for f in directory.iterdir()]
    folder_paths = folder_paths[0:num_folders]

    with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as process_executor:
        process_executor.map(parquetize_folder, folder_paths, repeat(destination))


if __name__ == "__main__":
    directory = Path("sample_dataset/raw_data/mat/")
    out_directory = Path("sample_dataset/processed_data/parquets/")

    t1 = time.perf_counter()
    parquetize_directory(directory, out_directory)
    print(f"Elapsed Time: {time.perf_counter() - t1} s")
