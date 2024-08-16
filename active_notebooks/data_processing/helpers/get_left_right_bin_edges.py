from data_processing.helpers.validate_bins import validate_bins
from numpy import ndarray


def get_left_right_bin_edges(bins: ndarray) -> tuple[ndarray, ndarray]:
    """Gets the left and right edges for each histogram bin as separate lists.

    :param bins: Array of (n+1) bin edges
    :type bins: ndarray
    :return: Array of n left bin edges, and array of n right bin edges
    :rtype: tuple[ndarray, ndarray]
    """
    validate_bins(bins)
    left_edges = bins[:-1]
    right_edges = bins[1:]
    return left_edges, right_edges
