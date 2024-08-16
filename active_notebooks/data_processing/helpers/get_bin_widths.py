from data_processing.helpers.get_left_right_bin_edges import get_left_right_bin_edges
from data_processing.helpers.validate_bins import validate_bins
from numpy import ndarray


def get_bin_widths(bins: ndarray) -> ndarray:
    """Gets the width of each bin in a histogram

    :param bins: Array of (n+1) bin edges
    :type bins: ndarray
    :return: Array of n bin widths
    :rtype: ndarray
    """
    validate_bins(bins)
    left_edges, right_edges = get_left_right_bin_edges(bins)
    widths = right_edges - left_edges
    return widths
