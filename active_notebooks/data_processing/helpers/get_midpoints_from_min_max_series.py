from pandas import Series, Index
from pandas.arrays import IntervalArray

def get_midpoints_from_min_max_series(min_series: Series, max_series: Series, index: Index) -> Series:
    intervals = Series(
        IntervalArray.from_arrays(min_series, max_series),
        index=index,
    )
    midpoints = Series(intervals.array.mid, index=index)
    return midpoints
