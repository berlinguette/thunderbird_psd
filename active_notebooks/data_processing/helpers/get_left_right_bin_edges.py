from data_processing.helpers.validate_bins import validate_bins
from numpy import ndarray


def get_left_right_bin_edges(bins: ndarray) -> tuple[ndarray, ndarray]:
    validate_bins(bins)
    left_edges = bins[:-1]
    right_edges = bins[1:]
    return left_edges, right_edges
