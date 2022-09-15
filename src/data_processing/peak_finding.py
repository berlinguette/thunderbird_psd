import pandas as pd


def get_left_bases(
    series: pd.Series,
    peak_idx: int,
    peak_offset: int = 0
) -> int:
    """Returns the index of the min within `peak_idx - peak_offset`"""
    return series[peak_idx - peak_offset: peak_idx].idxmin()


def get_right_bases(
    peak_idx: int,
    tail_offset: int = 0
) -> int:
    """Returns the index of `peak_idx + tail_offset`"""
    return peak_idx + tail_offset


def get_bases(
    series: pd.Series,
    peak_idx: int,
    peak_offset: int = 10,
    tail_offset: int = 5
) -> tuple[list, list]:
    """Gets the left and right indices. See :func:`get_left_bases` and :func:`get_right_bases` for more information."""
    return get_left_bases(series, peak_idx, peak_offset), get_right_bases(series, peak_idx, tail_offset)
