from pathlib import Path

import numpy as np
import pandas as pd
import tomli
import tomli_w


def load_exp_info(exp_info_path: Path) -> dict:
    # TODO: add checking
    with open(exp_info_path, "rb") as f:
        exp_info = tomli.load(f)

    return exp_info


def process_file(file_path: Path, exp_info_path: Path):

    df = pd.read_parquet(file_path)
    df.columns = df.columns.astype("int16")
    num_initial_signals = df.shape[0]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    num_missing_signals = num_initial_signals - df.shape[0]

    dc_offset = df.iloc[0:50].mean()
    df = -1 * (df - dc_offset)

    sample_interval = load_exp_info(exp_info_path)["picoscope"]["sample_interval"]


def process_directory(root_directory: Path):
    pass


if __name__ == "__main__":
    pass
