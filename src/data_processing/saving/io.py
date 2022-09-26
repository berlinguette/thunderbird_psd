import datetime
from pathlib import Path

import pandas as pd
import os
import tomli
import tomli_w
from data_processing.cleaning.cleaning_configs import *

def create_folder(folder_path: Path):
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        raise FileExistsError("""The file you are trying to create already exists, clean
        out the buffer before rerunning the script""")


def load_exp_info(exp_info_path: Path) -> dict:
    # TODO: add checking
    with open(exp_info_path, "rb") as f:
        exp_info = tomli.load(f)

    return exp_info


def dump_settings(destination: Path) -> None:
    multipeak_filter_settings = {
        "height": PK_FIND_HEIGHT,
        "prominence": PK_FIND_PROMINENCE,
    }

    incomplete_filter_settings = {"threshold": TRIGGER_THRESHOLD}

    snr_filter_settings = {"threshold": MIN_SNR}

    report = {
        "last_updated": datetime.datetime.now(),
        "multipeak_filter_settings": multipeak_filter_settings,
        "incomplete_filter_settings": incomplete_filter_settings,
        "snr_filter_settings": snr_filter_settings,
    }

    with open(destination, "w+b") as f:
        tomli_w.dump(report, f)


def save_parquet(df: pd.DataFrame, filename: str, destination: Path) -> None:
    df = df.T
    df.columns = df.columns.astype("str")
    df.to_parquet(destination / filename)


def save_cleaning_report(uid: str, signal_stats: dict, destination: Path) -> None:
    try:
        with open(destination, "r+b") as f:
            saved_report = tomli.load(f)

            if uid in saved_report.keys():
                saved_report[uid].update(signal_stats)
            else:
                saved_report[uid] = signal_stats

        with open(destination, "wb") as f:
            sorted_report = {key: saved_report[key] for key in sorted(saved_report)}
            tomli_w.dump(sorted_report, f)

    except FileNotFoundError:
        with open(destination, "w+b") as f:
            tomli_w.dump({uid: signal_stats}, f)


def save_results(results: pd.DataFrame, destination: Path) -> None:
    try:
        df = pd.read_csv(destination, index_col=0)
    except FileNotFoundError:
        results.to_csv(destination, index_label="buffer_id")
    else:
        df.update(results)
        missing_rows = results.loc[results.index.difference(df.index)]
        df = pd.concat([df, missing_rows])
        df.to_csv(destination, encoding="utf-8", index_label="buffer_id")
