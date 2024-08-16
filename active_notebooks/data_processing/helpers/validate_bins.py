from numpy import ndarray


def validate_bins(bins: ndarray):
    """Checks if an array could contain valid bin edge values

    :param bins: Possible array of bin edges
    :type bins: ndarray
    :raises ValueError: if the array has the wrong shape or too few entries
    """
    if bins.ndim != 1:
        raise ValueError("Bins must be a 1D array")
    if bins.shape[0] < 2:
        raise ValueError("Bins must have at least 2 entries")
