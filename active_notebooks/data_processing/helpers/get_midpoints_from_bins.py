from data_processing.helpers.get_bin_widths import get_bin_widths
from data_processing.helpers.get_left_right_bin_edges import get_left_right_bin_edges
from data_processing.helpers.validate_bins import validate_bins
from numpy import ndarray


def get_midpoints_from_bins(bins: ndarray) -> ndarray:
    """Gets the midpoint of each bin in a histogram

    :param bins: Array of (n+1) bin edges
    :type bins: ndarray
    :return: Array of n bin midpoints
    :rtype: ndarray
    """
    validate_bins(bins)

    # left_edges = bins[:-1]
    # right_edges = bins[1:]
    left_edges, _ = get_left_right_bin_edges(bins)
    widths = get_bin_widths(bins)
    return left_edges + (widths / 2)
