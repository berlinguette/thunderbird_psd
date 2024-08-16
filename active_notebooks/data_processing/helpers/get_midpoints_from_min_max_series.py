from pandas import Index, Series
from pandas.arrays import IntervalArray


def get_midpoints_from_min_max_series(
    min_series: Series, max_series: Series, index: Index
) -> Series:
    """Gets the midpoint of each interval defined by values in two series.

    :param min_series: Minimum values of n intervals
    :type min_series: Series
    :param max_series: Maximum values of n intervals
    :type max_series: Series
    :param index: Desired index of the returned midpoints Series
    :type index: Index
    :return: Midpoint values of n intervals
    :rtype: Series
    """
    intervals = Series(
        IntervalArray.from_arrays(min_series, max_series),
        index=index,
    )
    midpoints = Series(intervals.array.mid, index=index)  # type: ignore
    return midpoints
