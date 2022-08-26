import os
from scipy.io import loadmat
import pandas as pd
import glob
import re
import time
from concurrent.futures import ProcessPoolExecutor

from pathlib import Path

def get_data_tuple(file_path, label):
    data = loadmat(file_path)
    data = data["A"].flatten()
    return data, label


def elapsed_time(t1):
    return time.perf_counter() - t1

def parquetize_folder(directory: Path, dest: str) -> pd.DataFrame:
    t1 = time.perf_counter()
    # file_paths = glob.glob(directory + "/*.mat")
    file_paths = [f for f in directory.iterdir()]
    print(f"Data: {elapsed_time(t1)}s")

    end_folder_name = os.path.basename(directory)
    
    batch_number = "b" + end_folder_name.split('-')[1]    
    
    file_names = map(lambda x: os.path.basename(x), file_paths)

    def get_label(file_name: str, batch_number: str = batch_number):
        serial_number = os.path.basename(file_name)
        serial_number = re.search("\_(\d\d\d\d)(.mat)", file_name)
        return batch_number + "s" + serial_number.group(1)
    
    labels = map(get_label, file_names)
    # print(f"Breakdown time: {elapsed_time(t1)}s")

    t1 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=6) as executor:
        result = executor.map(
            get_data_tuple, file_paths, labels
        )
    print(f"Processing time: {elapsed_time(t1)}s")

    # t1 = time.perf_counter()
    labelled_data = list(zip(*result))
    # print(f"Zipping time: {elapsed_time(t1)}s")

    # t1 = time.perf_counter()
    df = pd.DataFrame(labelled_data[0], index=labelled_data[1])
    df.columns = df.columns.astype(str)
    df.to_parquet(dest + "/" + end_folder_name + ".parquet")
    # print(f"Conversion time: {time.perf_counter() - t1}s")


import time

def main():
    directory = Path("C:/Users/User/Documents/GitHub/thunderbird_psd/sample_dataset/raw_data/psdata/20220730-0001")
    eject_dir = "C:/Users/User/Documents/GitHub/thunderbird_psd/sample_dataset/processed_data"
    parquetize_folder(directory, eject_dir)

if __name__ == '__main__':
    t1 = time.perf_counter()
    main()
    print(f"Elapsed Time: {time.perf_counter() - t1} s")
