import os
import re
import time
from multiprocessing.pool import Pool
from pathlib import Path

import pandas as pd
from scipy.io import loadmat

from parquetize import parquetize_folder


def get_data_tuple(file_path, label):
    data = loadmat(file_path)
    data = data["A"].flatten()
    return data, label


def elapsed_time(t1):
    return time.perf_counter() - t1


def parquetize_folder(directory: Path, dest: str) -> pd.DataFrame:
    file_paths = [f for f in directory.iterdir()]
    end_folder_name = os.path.basename(directory)
    batch_number = "b" + end_folder_name.split("-")[1]
    file_names = map(lambda x: os.path.basename(x), file_paths)

    def get_label(file_name: str, batch_number: str = batch_number):
        serial_number = os.path.basename(file_name)
        serial_number = re.search("\_(\d\d\d\d)(.mat)", file_name)
        return batch_number + "s" + serial_number.group(1)

    labels = map(get_label, file_names)

    t1 = time.perf_counter()
    with Pool(5) as pool:
        result = pool.starmap(get_data_tuple, zip(file_paths, labels))
    print(f"Processing time: {elapsed_time(t1)}s")

    labelled_data = list(zip(*result))

    df = pd.DataFrame(labelled_data[0], index=labelled_data[1])
    df.columns = df.columns.astype(str)
    print(df.head())
    df.to_parquet(dest + end_folder_name + ".parquet")


def runny():
    directory = Path(
        "C:/Users/Alvin/Desktop/Thunderbird/thunderbird_psd/sample_dataset/raw_data/mat/20220730-0001/"
    )

    eject_dir = "C:/Users/Alvin/Desktop/Thunderbird/thunderbird_psd/sample_dataset/processed_data/parquets/"

    parquetize_folder(directory, eject_dir)


if __name__ == "__main__":
    t1 = time.perf_counter()
    runny()
    print(f"Elapsed Time: {time.perf_counter() - t1} s")
