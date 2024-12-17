from numpy import ndarray
import numpy as np
from math import log10
from typing import TypeVar


class Histogram:
    def __init__(self, counts: ndarray, midpoints: ndarray):
        counts_shape = counts.shape
        if len(counts_shape) != 1:
            raise ValueError("Counts must be 1-dimensional")

        mids_shape = midpoints.shape
        if len(mids_shape) != 1:
            raise ValueError("Edges must be 1-dimensional")

        counts_size, *_ = counts_shape
        edges_size, *_ = mids_shape
        if counts_size != edges_size:
            raise ValueError("Edges size must be equal to Counts size")

        self._counts = counts
        self._mids = midpoints

    @property
    def counts(self) -> ndarray:
        return self._counts

    @property
    def midpoints(self) -> ndarray:
        return self._mids


class Histogram2D:
    def __init__(self, counts: ndarray, x_midpoints: ndarray, y_midpoints: ndarray):
        counts_shape = counts.shape
        if len(counts_shape) != 2:
            raise ValueError("Counts must be 2-dimensional")

        x_mids_shape = x_midpoints.shape
        if len(x_mids_shape) != 1:
            raise ValueError("X edges must be 1-dimensional")

        y_mids_shape = y_midpoints.shape
        if len(y_mids_shape) != 1:
            raise ValueError("Y edges must be 1-dimensional")

        counts_x_size, counts_y_size, *_ = counts_shape
        x_edges_size, *_ = x_mids_shape
        y_edges_size, *_ = y_mids_shape
        if counts_x_size != x_edges_size:
            raise ValueError("X edges size must be equal to Counts size in x dimension")
        if counts_y_size != y_edges_size:
            raise ValueError("Y edges size must be equal to Counts size in y dimension")

        self._counts = counts
        self._x_mids = x_midpoints
        self._y_mids = y_midpoints

    @property
    def counts(self) -> ndarray:
        return self._counts

    @property
    def x_midpoints(self) -> ndarray:
        return self._x_mids

    @property
    def y_midpoints(self) -> ndarray:
        return self._y_mids


def weight_factor(
    big_r: Histogram2D,
    big_phi: Histogram,
    big_n: Histogram,
    sigma: Histogram | None = None,
) -> Histogram2D:
    """Calculate the weight factor for this iteration of the GRAVEL algorithm.

    :param big_r: Neutron response matrix
    :type big_r: Histogram2D
    :param big_phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type big_phi: Histogram
    :param big_n: Neutron response spectrum
    :type big_n: Histogram
    :param sigma: Estimation of error in the neutron response spectrum,
        defaults to None (root of neutron response spectrum)
    :type sigma: Histogram | None, optional
    :raises ValueError: if dimensions do not match (N with R's x-axis, Phi with R's
        y-axis)
    :return: Weight factor
    :rtype: Histogram2D
    """
    if sigma is None:
        sigma = Histogram(np.sqrt(big_n.counts), big_n.midpoints)

    compatible, reason = _are_histograms_compatible_2d(big_r, big_n, 0)
    if not compatible:
        raise ValueError(f"N does not match x-axis of R ({reason})")
    compatible, reason = _are_histograms_compatible_2d(big_r, big_phi, 1)
    if not compatible:
        raise ValueError(f"Phi does not match y-axis of R ({reason})")
    compatible, reason = _are_histograms_compatible_2d(big_r, sigma, 0)
    if not compatible:
        raise ValueError(f"Sigma does not match x-axis of R ({reason})")
    close_enough = _are_r_dimensions_close_enough(big_r)
    if not close_enough:
        raise ValueError(
            f"Dimensions of R {big_r.counts.shape} are not within 1 order of magnitude"
        )

    R = big_r.counts
    phi = big_phi.counts
    N = big_n.counts
    error = sigma.counts

    phi = np.reshape(phi, (1, -1))
    N = np.reshape(N, (-1, 1))
    error = np.reshape(error, (-1, 1))

    numer = R * phi
    denom = np.sum(R * phi, axis=1, keepdims=True)
    right = np.square(N) / np.square(error)

    return Histogram2D((numer / denom) * right, big_r.x_midpoints, big_r.y_midpoints)


def next_phi(
    big_r: Histogram2D,
    big_phi: Histogram,
    big_n: Histogram,
    sigma: Histogram | None = None,
) -> Histogram:
    """Calculate the next neutron spectrum for this iteration of the GRAVEL algorithm.

    :param big_r: Neutron response matrix
    :type big_r: Histogram2D
    :param big_phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type big_phi: Histogram
    :param big_n: Neutron response spectrum
    :type big_n: Histogram
    :param sigma: Estimation of error in the neutron response spectrum,
        defaults to None (root of neutron response spectrum)
    :type sigma: Histogram | None, optional
    :raises ValueError: if dimensions do not match (N with R's x-axis, Phi with R's
        y-axis)
    :return: Next neutron spectrum
    :rtype: Histogram
    """
    if sigma is None:
        sigma = Histogram(np.sqrt(big_n.counts), big_n.midpoints)

    compatible, reason = _are_histograms_compatible_2d(big_r, big_n, 0)
    if not compatible:
        raise ValueError(f"N does not match x-axis of R ({reason})")
    compatible, reason = _are_histograms_compatible_2d(big_r, big_phi, 1)
    if not compatible:
        raise ValueError(f"Phi does not match y-axis of R ({reason})")
    compatible, reason = _are_histograms_compatible_2d(big_r, sigma, 0)
    if not compatible:
        raise ValueError(f"Sigma does not match x-axis of R ({reason})")
    close_enough = _are_r_dimensions_close_enough(big_r)
    if not close_enough:
        raise ValueError(
            f"Dimensions of R {big_r.counts.shape} are not within 1 order of magnitude"
        )

    big_w = weight_factor(big_r, big_phi, big_n, sigma=sigma)

    R = big_r.counts
    phi = big_phi.counts
    N = big_n.counts
    error = sigma.counts
    W = big_w.counts

    phi = np.reshape(phi, (1, -1))
    N = np.reshape(N, (-1, 1))
    error = np.reshape(error, (-1, 1))

    ln_denom = np.sum(R * phi, axis=1, keepdims=True)
    ln_result = np.log(N / ln_denom)
    exp_numer = np.sum(W * ln_result, axis=0, keepdims=True)
    exp_denom = np.sum(W, axis=0, keepdims=True)
    exp_result = np.exp(exp_numer / exp_denom)

    return Histogram((phi * exp_result).reshape(-1), big_phi.midpoints)


def stopping_criteria(
    big_r: Histogram2D,
    big_phi: Histogram,
    big_n: Histogram,
    sigma: Histogram | None = None,
) -> float:
    """Calculate the stopping criteria value for this iteration of the GRAVEL algorithm.

    The stopping criteria value is chi^2 divided by the degrees of freedom.
    Degrees of freedom for neutron response matrix R is (m-1)*(n-1), where m and n are
    the dimensions of R.

    :param big_r: Neutron response matrix
    :type big_r: Histogram2D
    :param big_phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type big_phi: Histogram
    :param big_n: Neutron response spectrum
    :type big_n: Histogram
    :param sigma: Estimation of error in the neutron response spectrum,
        defaults to None (root of neutron response spectrum)
    :type sigma: Histogram | None, optional
    :raises ValueError: if dimensions do not match (N with R's x-axis, Phi with R's
        y-axis)
    :return: Stopping criteria value
    :rtype: float
    """
    if sigma is None:
        sigma = Histogram(np.sqrt(big_n.counts), big_n.midpoints)

    compatible, reason = _are_histograms_compatible_2d(big_r, big_n, 0)
    if not compatible:
        raise ValueError(f"N does not match x-axis of R ({reason})")
    compatible, reason = _are_histograms_compatible_2d(big_r, big_phi, 1)
    if not compatible:
        raise ValueError(f"Phi does not match y-axis of R ({reason})")
    compatible, reason = _are_histograms_compatible_2d(big_r, sigma, 0)
    if not compatible:
        raise ValueError(f"Sigma does not match x-axis of R ({reason})")
    close_enough = _are_r_dimensions_close_enough(big_r)
    if not close_enough:
        raise ValueError(
            f"Dimensions of R {big_r.counts.shape} are not within 1 order of magnitude"
        )

    R = big_r.counts
    phi = big_phi.counts
    N = big_n.counts
    error = sigma.counts

    m, n = R.shape
    DOF = (m - 1) * (n - 1)

    phi = np.reshape(phi, (1, -1))
    N = np.reshape(N, (-1, 1))
    error = np.reshape(error, (-1, 1))

    sum_numer = np.sum(R * phi, axis=1, keepdims=True)
    numer = np.square(sum_numer - N)
    frac = numer / np.square(error)
    chi_sq = np.sum(frac, axis=0, keepdims=True)
    chi_sq = chi_sq.flatten().tolist()[0]

    return chi_sq / DOF


def unfold_spectrum(
    big_n: Histogram,
    big_r: Histogram2D,
    starting_phi: Histogram | None = None,
    sigma: Histogram | None = None,
    tolerance: float = 0.1,
    max_iterations: int = 500,
) -> Histogram:
    """Unfold neutron response spectrum into neutron spectrum.

    This function performs the GRAVEL algorithm iteratively, as seen in [CITATION HERE].

    :param big_n: Neutron response spectrum
    :type big_n: Histogram
    :param big_r: Neutron response matrix
    :type big_r: Histogram2D
    :param starting_phi: Starting neutron spectrum, defaults to None
        (i.e. 1 for all energy bins)
    :type starting_phi: Histogram | None, optional
    :param sigma: Estimation of error in the neutron response spectrum,
        defaults to None (root of neutron response spectrum)
    :type sigma: Histogram | None, optional
    :param tolerance: How close stopping criteria value must be to stopping value (1)
        to stop the GRAVEL algorithm
    :type tolerance: float
    :param max_iterations: Maximum number of iterations to perform before stopping the
        GRAVEL algorithm early
    :type max_iterations: int
    :raises ValueError: if dimensions do not match (N with R's x-axis, Phi with R's
        y-axis), or if
    :raises RuntimeError: when stopping criteria is not met within the maximum number of
        iterations
    :return: _description_
    :rtype: Histogram
    """
    if sigma is None:
        sigma = Histogram(np.sqrt(big_n.counts), big_n.midpoints)
    if starting_phi is None:
        big_r_y_size = big_r.counts.shape[1]
        uniform_phi = np.ones(big_r_y_size)
        uniform_phi_edges = big_r.y_midpoints
        starting_phi = Histogram(uniform_phi, uniform_phi_edges)

    compatible, reason = _are_histograms_compatible_2d(big_r, big_n, 0)
    if not compatible:
        raise ValueError(f"N does not match x-axis of R ({reason})")
    compatible, reason = _are_histograms_compatible_2d(big_r, starting_phi, 1)
    if not compatible:
        raise ValueError(f"Phi does not match y-axis of R ({reason})")
    compatible, reason = _are_histograms_compatible_2d(big_r, sigma, 0)
    if not compatible:
        raise ValueError(f"Sigma does not match x-axis of R ({reason})")
    close_enough = _are_r_dimensions_close_enough(big_r)
    if not close_enough:
        raise ValueError(
            f"Dimensions of R {big_r.counts.shape} are not within 1 order of magnitude"
        )

    big_phi = starting_phi
    stop_value = 1 + tolerance
    iters = 0
    chi_n = stopping_criteria(big_r, big_phi, big_n, sigma=sigma)

    while chi_n > stop_value:
        print(chi_n)
        big_phi = next_phi(big_r, big_phi, big_n, sigma=sigma)
        chi_n = stopping_criteria(big_r, big_phi, big_n, sigma=sigma)

        iters += 1
        if iters >= max_iterations:
            raise RuntimeError(
                "Spectrum could not be unfolded within max allowable iterations"
            )

    return big_phi


T = TypeVar("T", Histogram, None)
def strip_zeroes(big_r: Histogram2D, big_n: Histogram, sigma: T = None) -> tuple[Histogram2D, Histogram, T]:
    compatible, reason = _are_histograms_compatible_2d(big_r, big_n, 0)
    if not compatible:
        raise ValueError(f"N does not match x-axis of R ({reason})")
    if sigma is not None:
        compatible, reason = _are_histograms_compatible_2d(big_r, sigma, 0)
        if not compatible:
            raise ValueError(f"Sigma does not match x-axis of R ({reason})")
    
    R_counts = big_r.counts.copy()
    R_x_mids = big_r.x_midpoints.copy()
    N_counts = big_n.counts.copy()
    N_mids = big_n.midpoints.copy()
    
    nonzero_mask = N_counts != 0
    R_counts = R_counts[nonzero_mask,:]
    R_x_mids = R_x_mids[nonzero_mask]
    N_counts = N_counts[nonzero_mask]
    N_mids = N_mids[nonzero_mask]
    
    new_R = Histogram2D(R_counts, R_x_mids, big_r.y_midpoints)
    new_N = Histogram(N_counts, N_mids)
    
    if sigma is None:
        return new_R, new_N, None # type: ignore
    else:
        sig_counts = sigma.counts.copy()
        sig_mids = sigma.midpoints.copy()
        sig_counts = sig_counts[nonzero_mask]
        sig_mids = sig_mids[nonzero_mask]
        new_sigma = Histogram(sig_counts, sig_mids)
        return new_R, new_N, new_sigma # type: ignore


def _are_histograms_compatible(a: Histogram, b: Histogram) -> tuple[bool, str]:
    a_size, *_ = a.counts.shape
    b_size, *_ = b.counts.shape
    if a_size != b_size:
        return False, "Sizes do not match"
    if not all(a.midpoints == b.midpoints):
        return False, "Bin edges do not match"
    return True, ""


def _are_histograms_compatible_2d(
    a: Histogram2D, b: Histogram, axis: int
) -> tuple[bool, str]:
    a_size = a.counts.shape[axis]
    b_size, *_ = b.counts.shape
    if a_size != b_size:
        return False, "Sizes do not match"
    a_edges = a.x_midpoints if axis == 0 else a.y_midpoints
    if not all(a_edges == b.midpoints):
        return False, "Bin edges do not match"
    return True, ""


def _are_r_dimensions_close_enough(r: Histogram2D) -> bool:
    x_size, y_size = r.counts.shape
    return abs(log10(x_size) - log10(y_size)) <= 1
