import numpy as np
import pandas as pd
from scipy import signal


def rms(series: pd.Series, index: int) -> pd.Series:
    n = len(series.values[0:index])
    total = sum(series.values[0:index] ** 2)
    return np.sqrt(1 / n * total)


def subtract_rms(series: pd.Series, baseline_rms: pd.Series) -> pd.Series:
    return series - baseline_rms[series.name]


def is_single(series: list) -> bool:
    return True if len(series) == 1 else False


def filter_multipeaks(
    df: pd.DataFrame, height: float, prominence: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = df.apply(
        lambda series: signal.find_peaks(
            series.values, height=height, prominence=prominence
        )
    )
    peak_idx, props = output.iloc[0, :], output.iloc[1, :]

    peak_idx: pd.Series
    return df.T[peak_idx.apply(is_single)].T, props


def filter_incomplete_triggers(df: pd.DataFrame, threshold: float):
    entry_filter = df.iloc[0] > threshold
    exit_filter = df.iloc[-1] > threshold
    df = df.T[~entry_filter].T
    df = df.T[~exit_filter].T
    return df


def snr_filter(peak_height: float, baseline_rms: float, min_snr: float) -> bool:
    return False if peak_height / baseline_rms < min_snr else True


def filter_low_snr(
    df: pd.DataFrame, peak_map: dict, baseline_rms: dict, min_snr: float
) -> pd.DataFrame:
    filt = df.apply(
        lambda series: snr_filter((peak_map), baseline_rms[series.name], min_snr)
    )
    return df.T[filt].T
