from numpy import ndarray


def validate_bins(bins: ndarray):
    if bins.ndim != 1:
        raise ValueError("Bins must be a 1D array")
    if bins.shape[0] < 2:
        raise ValueError("Bins must have at least 2 entries")
