from math import log10
from typing import TypedDict, TypeVar

import numpy as np
from numpy import ndarray


class Histogram:
    # TODO add docstrings where missing
    """Stores data for a histogram.

    This includes counts and midpoints for each bin.
    """

    def __init__(self, counts: ndarray, midpoints: ndarray):
        """Create a new Histogram.

        :param counts: count for each histogram bin
        :type counts: ndarray
        :param midpoints: midpoint of each histogram bin
        :type midpoints: ndarray
        :raises ValueError: if counts are not 1-dimensional
        :raises ValueError: if midpoints are not 1-dimensional
        :raises ValueError: if midpoints and counts are not the same size
        """
        counts_shape = counts.shape
        if len(counts_shape) != 1:
            raise ValueError("Counts must be 1-dimensional")

        mids_shape = midpoints.shape
        if len(mids_shape) != 1:
            raise ValueError("Midpoints must be 1-dimensional")

        counts_size, *_ = counts_shape
        edges_size, *_ = mids_shape
        if counts_size != edges_size:
            raise ValueError("Midpoints size must be equal to Counts size")

        self._counts = counts
        self._mids = midpoints

    @property
    def counts(self) -> ndarray:
        """Get histogram counts

        :return: histogram counts
        :rtype: ndarray
        """
        return self._counts

    @property
    def midpoints(self) -> ndarray:
        """Get histogram bin midpoints

        :return: histogram bin midpoints
        :rtype: ndarray
        """
        return self._mids


class Histogram2D:
    """Stores data for a 2-dimensional histogram.

    This includes counts and midpoints (on both axes) for each bin.

    We define the x and y axes based on the order in numpy's shape property, so the
    x axis is actually the column axis, while the y axis is the row axis
    """

    def __init__(self, counts: ndarray, x_midpoints: ndarray, y_midpoints: ndarray):
        """Create a new Histogram2D.

        :param counts: count for each histogram bin
        :type counts: ndarray
        :param x_midpoints: midpoint for each histogram bin on the x (column) axis
        :type x_midpoints: ndarray
        :param y_midpoints: midpoint for each histogram bin on the y (row) axis
        :type y_midpoints: ndarray
        :raises ValueError: if counts are not 2-dimensional
        :raises ValueError: if x axis midpoints are not 1-dimensional
        :raises ValueError: if y axis midpoints are not 1-dimensional
        :raises ValueError: if x axis midpoints are not the same size as count's x
            dimension
        :raises ValueError: if y axis midpoints are not the same size as counts' y
            dimension
        """
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
        """Get histogram counts

        :return: histogram counts
        :rtype: ndarray
        """
        return self._counts

    @property
    def x_midpoints(self) -> ndarray:
        """Get histogram bin midpoints for the x (column) axis

        :return: histogram bin midpoints for the x axis
        :rtype: ndarray
        """
        return self._x_mids

    @property
    def y_midpoints(self) -> ndarray:
        """_summary_

        :return: _description_
        :rtype: ndarray
        """
        return self._y_mids


class NDHistogram:
    """Stores data for a n-dimensional histogram.

    This includes counts and midpoints for each bin.
    """

    def __init__(self, counts: ndarray, midpoints: list[ndarray]):
        """Create a new NDHistogram.

        :param counts: count for each histogram bin
        :type counts: ndarray
        :param midpoints: midpoint for each histogram bin across all dimensions
        :type midpoints: list[ndarray]
        """
        counts_shape = counts.shape
        if len(counts_shape) != len(midpoints):
            raise ValueError(
                "Counts and midpoints must have the same number of dimensions"
            )

        for i, (dim_size, dim_mids) in enumerate(zip(counts_shape, midpoints)):
            mids_shape = dim_mids.shape
            if len(mids_shape) != 1:
                raise ValueError(f"Midpoints for axis {i} were not 1-dimensional")
            if mids_shape[0] != dim_size:
                raise ValueError(
                    f"Length of midpoints for axis {i} does not match size of counts on that axis"
                )

        self._counts = counts
        self._mids = midpoints

    @property
    def counts(self) -> ndarray:
        """Get histogram counts.

        :return: histogram counts
        :rtype: ndarray
        """
        return self._counts

    @property
    def midpoints(self) -> list[ndarray]:
        """Get histogram midpoints.

        :return: histogram midpoints
        :rtype: list[ndarray]
        """
        return self._mids

    @property
    def shape(self) -> tuple[int, ...]:
        """Get histogram shape.

        :return: Histogram shape
        :rtype: tuple[int, ...]
        """
        return self._counts.shape


class UnfoldingProcessInfo(TypedDict):
    errors: list[float]
    chis: list[float]
    phis: list[ndarray]
    weights: list[ndarray]


def weight_factor(
    big_r: Histogram2D,
    new_phi: Histogram,
    big_n: Histogram,
    sigma: Histogram | None = None,
) -> Histogram2D:
    """Calculate the weight factor for this iteration of the GRAVEL algorithm.

    :param big_r: Neutron response matrix
    :type big_r: Histogram2D
    :param new_phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type new_phi: Histogram
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
    compatible, reason = _are_histograms_compatible_2d(big_r, new_phi, 1)
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
    phi = new_phi.counts
    N = big_n.counts
    error = sigma.counts

    phi = np.reshape(phi, (1, -1))
    N = np.reshape(N, (-1, 1))
    error = np.reshape(error, (-1, 1))

    numer = R * phi
    denom = np.sum(R * phi, axis=1, keepdims=True)
    right = np.square(N) / np.square(error)

    return Histogram2D((numer / denom) * right, big_r.x_midpoints, big_r.y_midpoints)


def new_weight_factor(
    r: NDHistogram,
    phi: NDHistogram,
    n: NDHistogram,
    sigma: NDHistogram | None = None,
) -> NDHistogram:
    """Calculate the weight factor for this iteration of the GRAVEL algorithm.

    :param big_r: Neutron response matrix
    :type big_r: NDHistogram
    :param new_phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type new_phi: NDHistogram
    :param big_n: Neutron response spectrum
    :type big_n: NDHistogram
    :param sigma: Estimation of error in the neutron response spectrum,
        defaults to None (root of neutron response spectrum)
    :type sigma: NDHistogram | None, optional
    :raises ValueError: if dimensions do not match (N with R's x-axis, Phi with R's
        y-axis)
    :return: Weight factor
    :rtype: NDHistogram
    """
    if sigma is None:
        sigma = NDHistogram(np.sqrt(n.counts), n.midpoints)

    compatible, reason = _is_n_compatible(r, n)
    if not compatible:
        raise ValueError(f"R and N are incompatible: {reason}")
    compatible, reason = _is_phi_compatible(r, phi)
    if not compatible:
        raise ValueError(f"R and Phi are incompatible: {reason}")
    compatible, reason = _is_n_compatible(r, sigma)
    if not compatible:
        raise ValueError(f"R and Sigma are incompatible: {reason}")
    close = _new_are_r_dimensions_close_enough(r)
    if not close:
        raise ValueError(
            f"Dimensions of R {r.shape} are not within 1 order of magnitude"
        )

    _R = r.counts
    _phi = phi.counts
    _N = n.counts
    _sigma = sigma.counts

    numer = _R * _phi
    denom = np.sum(_R * _phi, axis=1, keepdims=True)
    right = np.square(_N) / np.square(_sigma)
    result = (numer / denom) * right

    return NDHistogram(result, r.midpoints)


def next_phi(
    big_r: Histogram2D,
    new_phi: Histogram,
    big_n: Histogram,
    sigma: Histogram | None = None,
) -> Histogram:
    """Calculate the next neutron spectrum for this iteration of the GRAVEL algorithm.

    :param big_r: Neutron response matrix
    :type big_r: Histogram2D
    :param new_phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type new_phi: Histogram
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
    compatible, reason = _are_histograms_compatible_2d(big_r, new_phi, 1)
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

    big_w = weight_factor(big_r, new_phi, big_n, sigma=sigma)

    R = big_r.counts
    phi = new_phi.counts
    N = big_n.counts
    W = big_w.counts

    phi = np.reshape(phi, (1, -1))
    N = np.reshape(N, (-1, 1))

    ln_denom = np.sum(R * phi, axis=1, keepdims=True)
    ln_result = np.log(N / ln_denom)
    exp_numer = np.sum(W * ln_result, axis=0, keepdims=True)
    exp_numer = np.nan_to_num(exp_numer)
    exp_denom = np.sum(W, axis=0, keepdims=True)

    # where denom is 0, change denom to 1, numer to 0
    # this makes frac = 0, and exp(frac) = 1
    # as used in https://github.com/tylerdolezal/Neutron-Unfolding/blob/main/gravel.py, ln 36-39
    denom_mask = exp_denom == 0
    exp_numer[denom_mask] = 0
    exp_denom[denom_mask] = 1

    exp_frac = exp_numer / exp_denom
    exp_result = np.exp(exp_frac)

    return Histogram((phi * exp_result).reshape(-1), new_phi.midpoints)


def new_next_phi(
    r: NDHistogram,
    phi: NDHistogram,
    n: NDHistogram,
    sigma: NDHistogram | None = None,
) -> NDHistogram:
    """Calculate the next neutron spectrum for this iteration of the GRAVEL algorithm.

    :param big_r: Neutron response matrix
    :type big_r: NDHistogram
    :param new_phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type new_phi: NDHistogram
    :param big_n: Neutron response spectrum
    :type big_n: NDHistogram
    :param sigma: Estimation of error in the neutron response spectrum,
        defaults to None (root of neutron response spectrum)
    :type sigma: NDHistogram | None, optional
    :raises ValueError: if dimensions do not match (N with R's x-axis, Phi with R's
        y-axis)
    :return: Next neutron spectrum
    :rtype: NDHistogram
    """
    if sigma is None:
        sigma = NDHistogram(np.sqrt(n.counts), n.midpoints)

    compatible, reason = _is_n_compatible(r, n)
    if not compatible:
        raise ValueError(f"R and N are incompatible: {reason}")
    compatible, reason = _is_phi_compatible(r, phi)
    if not compatible:
        raise ValueError(f"R and Phi are incompatible: {reason}")
    compatible, reason = _is_n_compatible(r, sigma)
    if not compatible:
        raise ValueError(f"R and Sigma are incompatible: {reason}")
    close = _new_are_r_dimensions_close_enough(r)
    if not close:
        raise ValueError(
            f"Dimensions of R {r.shape} are not within 1 order of magnitude"
        )

    w = new_weight_factor(r, phi, n, sigma=sigma)

    _R = r.counts
    _phi = phi.counts
    _N = n.counts
    _W = w.counts

    ln_denom = np.sum(_R * _phi, axis=1, keepdims=True)
    ln_result = np.log(_N / ln_denom)
    exp_numer = np.sum(_W * ln_result, axis=0, keepdims=True)
    exp_numer = np.nan_to_num(exp_numer)
    exp_denom = np.sum(_W, axis=0, keepdims=True)

    # where denom is 0, change denom to 1, numer to 0
    # this makes frac = 0, and exp(frac) = 1
    # as used in https://github.com/tylerdolezal/Neutron-Unfolding/blob/main/gravel.py, ln 36-39
    denom_mask = exp_denom == 0
    exp_numer[denom_mask] = 0
    exp_denom[denom_mask] = 1

    exp_frac = exp_numer / exp_denom
    exp_result = np.exp(exp_frac)
    result = _phi * exp_result

    r_mid0, r_mid1, *_ = r.midpoints
    phi_mid0 = np.array([r_mid0.mean])
    phi_mid1 = r_mid1.copy()
    phi_mids = [phi_mid0, phi_mid1]

    return NDHistogram(result, phi_mids)


def stopping_criteria(
    big_r: Histogram2D,
    new_phi: Histogram,
    big_n: Histogram,
    sigma: Histogram | None = None,
) -> float:
    """Calculate the stopping criteria value for this iteration of the GRAVEL algorithm.

    The stopping criteria value is chi^2 divided by the degrees of freedom.
    Degrees of freedom for neutron response matrix R is (m-1)*(n-1), where m and n are
    the dimensions of R.

    :param big_r: Neutron response matrix
    :type big_r: Histogram2D
    :param new_phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type new_phi: Histogram
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
    compatible, reason = _are_histograms_compatible_2d(big_r, new_phi, 1)
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
    phi = new_phi.counts
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


def new_stopping_criteria(
    r: NDHistogram,
    phi: NDHistogram,
    n: NDHistogram,
    sigma: NDHistogram | None = None,
) -> float:
    """Calculate the stopping criteria value for this iteration of the GRAVEL algorithm.

    The stopping criteria value is chi^2 divided by the degrees of freedom.
    Degrees of freedom for neutron response matrix R is (m-1)*(n-1), where m and n are
    the dimensions of R.

    :param big_r: Neutron response matrix
    :type big_r: NDHistogram
    :param new_phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type new_phi: NDHistogram
    :param big_n: Neutron response spectrum
    :type big_n: NDHistogram
    :param sigma: Estimation of error in the neutron response spectrum,
        defaults to None (root of neutron response spectrum)
    :type sigma: NDHistogram | None, optional
    :raises ValueError: if dimensions do not match (N with R's x-axis, Phi with R's
        y-axis)
    :return: Stopping criteria value
    :rtype: float
    """
    if sigma is None:
        sigma = NDHistogram(np.sqrt(n.counts), n.midpoints)

    compatible, reason = _is_n_compatible(r, n)
    if not compatible:
        raise ValueError(f"R and N are incompatible: {reason}")
    compatible, reason = _is_phi_compatible(r, phi)
    if not compatible:
        raise ValueError(f"R and Phi are incompatible: {reason}")
    compatible, reason = _is_n_compatible(r, sigma)
    if not compatible:
        raise ValueError(f"R and Sigma are incompatible: {reason}")
    close = _new_are_r_dimensions_close_enough(r)
    if not close:
        raise ValueError(
            f"Dimensions of R {r.shape} are not within 1 order of magnitude"
        )

    _R = r.counts
    _phi = phi.counts
    _N = n.counts
    _sigma = sigma.counts

    _m, _n = r.shape
    DOF = (_m - 1) * (_n - 1)

    sum_numer = np.sum(_R * _phi, axis=1, keepdims=True)
    numer = np.square(sum_numer - _N)
    frac = numer / np.square(_sigma)
    chi_sq = np.sum(frac, axis=0, keepdims=True)
    chi_sq = chi_sq.flatten().tolist()[0]
    result = chi_sq / DOF

    return result


def unfold_spectrum(
    big_n: Histogram,
    big_r: Histogram2D,
    starting_phi: Histogram | None = None,
    sigma: Histogram | None = None,
    tolerance: float = 0.01,
    max_iterations: int = 500,
    full_info: bool = False,
) -> tuple[Histogram, UnfoldingProcessInfo | None]:
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
        to stop the GRAVEL algorithm, defaults to 0.1
    :type tolerance: float, optional
    :param max_iterations: Maximum number of iterations to perform before stopping the
        GRAVEL algorithm early, defaults to 500
    :type max_iterations: int, optional
    :param full_info: Whether to return extra data (errors, neutron spectra, weight
        factors) collected during the unfolding process, defaults to False
    :type full_info: bool, optional
    :return: Unfolded spectrum, and (if full_info is True) a dictionary of intermediate
        data collected during the unfolding process (or None if full_info is False)
    :rtype: tuple[Histogram, UnfoldingProcessInfo | None]
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

    new_r, new_n, new_phi, new_sigma = strip_zeroes(
        big_r, big_n, starting_phi, sigma=sigma
    )

    iters = 0
    chis = []
    phis = []
    weights = []
    errors = []
    iter_text_len = len(str(max_iterations))

    chi_n = stopping_criteria(new_r, new_phi, new_n, sigma=new_sigma)
    chi_last = chi_n
    delta_chi_last = 1
    delta_delta = 1

    while delta_delta > tolerance:
        weight = weight_factor(new_r, new_phi, new_n, sigma=new_sigma)
        new_phi = next_phi(new_r, new_phi, new_n, sigma=new_sigma)
        chi_n = stopping_criteria(new_r, new_phi, new_n, sigma=new_sigma)

        delta_chi = chi_n - chi_last
        delta_delta = abs(delta_chi - delta_chi_last)
        chi_last = chi_n
        delta_chi_last = delta_chi

        if full_info:
            chis.append(chi_n)
            phis.append(new_phi)
            weights.append(weight)
            errors.append(delta_chi)

        if iters % 10 == 0:
            print(
                f"Iter. {iters: {iter_text_len}d}: chi = {chi_n:.3g}, rel_rate = {delta_delta: .3g}"
            )
        iters += 1
        if iters >= max_iterations:
            break

    unfolding_info = (
        UnfoldingProcessInfo(errors=errors, chis=chis, phis=phis, weights=weights)
        if full_info
        else None
    )
    return new_phi, unfolding_info


def new_unfold_spectrum(
    n: NDHistogram,
    r: NDHistogram,
    phi0: NDHistogram | None = None,
    sigma: NDHistogram | None = None,
    tolerance: float = 0.01,
    max_iterations: int = 500,
    full_info: bool = False,
) -> tuple[NDHistogram, UnfoldingProcessInfo | None]:
    """Unfold neutron response spectrum into neutron spectrum.

    This function performs the GRAVEL algorithm iteratively, as seen in [CITATION HERE].

    :param n: Neutron response spectrum
    :type n: NDHistogram
    :param r: Neutron response matrix
    :type r: NDHistogram
    :param phi0: Starting neutron spectrum, defaults to None
        (i.e. 1 for all energy bins)
    :type phi0: NDHistogram | None, optional
    :param sigma: Estimation of error in the neutron response spectrum,
        defaults to None (root of neutron response spectrum)
    :type sigma: NDHistogram | None, optional
    :param tolerance: How close stopping criteria value must be to stopping value (1)
        to stop the GRAVEL algorithm, defaults to 0.1
    :type tolerance: float, optional
    :param max_iterations: Maximum number of iterations to perform before stopping the
        GRAVEL algorithm early, defaults to 500
    :type max_iterations: int, optional
    :param full_info: Whether to return extra data (errors, neutron spectra, weight
        factors) collected during the unfolding process, defaults to False
    :type full_info: bool, optional
    :return: Unfolded spectrum, and (if full_info is True) a dictionary of intermediate
        data collected during the unfolding process (or None if full_info is False)
    :rtype: tuple[Histogram, UnfoldingProcessInfo | None]
    """
    if sigma is None:
        sigma = NDHistogram(np.sqrt(n.counts), n.midpoints)
    if phi0 is None:
        big_r_y_size = r.shape[1]
        uniform_phi = np.ones(big_r_y_size)
        uniform_phi = np.reshape(uniform_phi, (1, -1))
        r_mids0, r_mids1 = r.midpoints
        phi_mids0 = np.array([r_mids0.mean])
        phi_mids1 = r_mids1.copy()
        phi_mids = [phi_mids0, phi_mids1]
        phi0 = NDHistogram(uniform_phi, phi_mids)

    compatible, reason = _is_n_compatible(r, n)
    if not compatible:
        raise ValueError(f"R and N are incompatible: {reason}")
    compatible, reason = _is_phi_compatible(r, phi0)
    if not compatible:
        raise ValueError(f"R and Phi are incompatible: {reason}")
    compatible, reason = _is_n_compatible(r, sigma)
    if not compatible:
        raise ValueError(f"R and Sigma are incompatible: {reason}")
    close = _new_are_r_dimensions_close_enough(r)
    if not close:
        raise ValueError(
            f"Dimensions of R {r.shape} are not within 1 order of magnitude"
        )

    iters = 0
    chis = []
    phis = []
    weights = []
    errors = []
    iter_text_len = len(str(max_iterations))

    chi_n = new_stopping_criteria(r, phi0, n, sigma=sigma)
    phi_k = NDHistogram(phi0.counts.copy(), [mids.copy() for mids in phi0.midpoints])
    chi_last = chi_n
    delta_chi_last = 1
    delta_delta = 1

    while delta_delta > tolerance:
        weight = new_weight_factor(r, phi_k, n, sigma=sigma)
        new_phi = new_next_phi(r, phi_k, n, sigma=sigma)
        chi_n = new_stopping_criteria(r, phi_k, n, sigma=sigma)

        delta_chi = chi_n - chi_last
        delta_delta = abs(delta_chi - delta_chi_last)
        chi_last = chi_n
        delta_chi_last = delta_chi

        if full_info:
            chis.append(chi_n)
            phis.append(new_phi)
            weights.append(weight)
            errors.append(delta_chi)

        if iters % 10 == 0:
            print(
                f"Iter. {iters: {iter_text_len}d}: chi = {chi_n:.3g}, rel_rate = {delta_delta: .3g}"
            )
        iters += 1
        if iters >= max_iterations:
            break

    unfolding_info = (
        UnfoldingProcessInfo(errors=errors, chis=chis, phis=phis, weights=weights)
        if full_info
        else None
    )
    return new_phi, unfolding_info


T = TypeVar("T", Histogram, None)


def strip_zeroes(
    big_r: Histogram2D, big_n: Histogram, starting_phi: Histogram, sigma: T = None
) -> tuple[Histogram2D, Histogram, Histogram, T]:
    """Strips any zero channels from incoming spectrum unfolding data.

    Zero channels in the neutron response spectrum (N) are removed, as well as matching
    bin midpoints. The corresponding channels on the x axis of the neutron response
    matrix (R) are also removed, as well as the matching bin midpoints on the x axis.

    If sigma is provided, matching channels and bin midpoints will be removed.

    :param big_r: Neutron response matrix
    :type big_r: Histogram2D
    :param big_n: Neutron response spectrum
    :type big_n: Histogram
    :param starting_phi: Initial neutron spectrum guess
    :type starting_phi: Histogram
    :param sigma: Estimation of error in the neutron response spectrum, or None,
        defaults to None
    :type sigma: T (Histogram or None), optional
    :raises ValueError: if N is incompatible with R's x axis
    :raises ValueError: if phi is incompatible with R's y axis
    :raises ValueError: if sigma (if provided) is incompatible with R's x axis
    :return: Zero-stripped R, N, and phi, and zero-stripped sigma (if provided, or None
        if not)
    :rtype: tuple[Histogram2D, Histogram, Histogram, T]
    """
    compatible, reason = _are_histograms_compatible_2d(big_r, big_n, 0)
    if not compatible:
        raise ValueError(f"N does not match x-axis of R ({reason})")
    compatible, reason = _are_histograms_compatible_2d(big_r, starting_phi, 1)
    if not compatible:
        raise ValueError(f"Phi does not match y-axis of R ({reason})")
    if sigma is not None:
        compatible, reason = _are_histograms_compatible_2d(big_r, sigma, 0)
        if not compatible:
            raise ValueError(f"Sigma does not match x-axis of R ({reason})")

    R_counts = big_r.counts.copy()
    R_x_mids = big_r.x_midpoints.copy()
    R_y_mids = big_r.y_midpoints.copy()
    N_counts = big_n.counts.copy()
    N_mids = big_n.midpoints.copy()
    phi_counts = starting_phi.counts.copy()
    phi_mids = starting_phi.midpoints.copy()

    # strip L channels where count = 0 (and matching in R)
    nonzero_mask = N_counts != 0
    R_counts = R_counts[nonzero_mask, :]
    R_x_mids = R_x_mids[nonzero_mask]
    N_counts = N_counts[nonzero_mask]
    N_mids = N_mids[nonzero_mask]

    # strip R slices where sum = 0 (and matching in starting_phi)
    zerosum_mask = np.sum(R_counts, axis=0) != 0
    R_counts = R_counts[:, zerosum_mask]
    R_y_mids = R_y_mids[zerosum_mask]
    phi_counts = phi_counts[zerosum_mask]
    phi_mids = phi_mids[zerosum_mask]

    new_R = Histogram2D(R_counts, R_x_mids, R_y_mids)
    new_N = Histogram(N_counts, N_mids)
    new_phi = Histogram(phi_counts, phi_mids)

    if sigma is None:
        return new_R, new_N, new_phi, None  # type: ignore
    else:
        sig_counts = sigma.counts.copy()
        sig_mids = sigma.midpoints.copy()
        sig_counts = sig_counts[nonzero_mask]
        sig_mids = sig_mids[nonzero_mask]
        new_sigma = Histogram(sig_counts, sig_mids)
        return new_R, new_N, new_phi, new_sigma  # type: ignore


S = TypeVar("S", NDHistogram, None)


def clean_data(
    r: NDHistogram,
    n: NDHistogram,
    phi: NDHistogram,
    sigma: S = None,
    L_cut: float | None = None,
) -> tuple[NDHistogram, NDHistogram, NDHistogram, S]:
    """Cleans incoming spectrum unfolding data.
    
    Cleaning consists of removing channels from one of the provided histograms (along 
    with associated bin midpoints), then removing any corresponding channels from
    "connected" histograms to maintain shape matching. For example, if channels are
    removed from N, the corresponding channels on the 1st axis of R are also removed. 
    
    Cleaning involves the following steps:
        1. Removing channels from N where bin midpoint is less than L_cut (if provided)
        2. Removing channels from N where counts are 0
        3. Removing channels from R (on either axis) where the sum is 0

    If sigma is provided, matching channels and bin midpoints will be removed as
    described above.

    :param r: Neutron response matrix
    :type r: NDHistogram
    :param n: Neutron response spectrum
    :type n: NDHistogram
    :param phi: Initial neutron spectrum guess
    :type phi: NDHistogram
    :param sigma: Optional estimation of error in the neutron response spectrum,
        defaults to None
    :type sigma: NDHistogram | None, optional
    :param L_cut: if not None, light outputs to remove from N before stripping zeroes;
        defaults to None (i.e. )
    :type L_cut: float | None, optional
    :raises ValueError: if N is incompatible with R's x axis
    :raises ValueError: if phi is incompatible with R's y axis
    :raises ValueError: if sigma (if provided) is incompatible with R's x axis
    :return: Zero-stripped R, N, and phi, and zero-stripped sigma (if provided, or None
        if not)
    :rtype: tuple[NDHistogram, NDHistogram, NDHistogram, NDHistogram] (if sigma provided)
    :rtype: tuple[NDHistogram, NDHistogram, NDHistogram, None] (if sigma not provided)]
    """
    compatible, reason = _is_n_compatible(r, n)
    if not compatible:
        raise ValueError(f"R and N are incompatible: {reason}")
    compatible, reason = _is_phi_compatible(r, phi)
    if not compatible:
        raise ValueError(f"R and Phi are incompatible: {reason}")
    if sigma is not None:
        compatible, reason = _is_n_compatible(r, sigma)
        if not compatible:
            raise ValueError(f"R and Sigma are incompatible: {reason}")
        
    R_counts = r.counts.copy()
    R_mids = [mids.copy() for mids in r.midpoints]
    L_mids, E_mids, *_ = R_mids
    N_counts = n.counts.copy()
    phi_counts = phi.counts.copy()
    
    # cut off low L if L_cut exists
    if L_cut is not None:
        L_cut_mask = L_mids >= L_cut
        N_counts = N_counts[L_cut_mask, :]
        R_counts = R_counts[L_cut_mask, :]
        L_mids = L_mids[L_cut_mask]
    
    # strip L channels from N where count = 0 (and matching in R)
    L_mask = np.sum(N_counts, axis=1) != 0
    N_counts = N_counts[L_mask, :]
    R_counts = R_counts[L_mask, :]
    L_mids = L_mids[L_mask]
    
    # strip rows/cols from R where sum = 0 (and matching in N/phi)
    R_L_mask = np.sum(R_counts, axis=1) != 0
    R_E_mask = np.sum(R_counts, axis=0) != 0
    R_counts = R_counts[R_L_mask, R_E_mask]
    N_counts = N_counts[R_L_mask, :]
    phi_counts = phi_counts[:, R_E_mask]
    L_mids = L_mids[R_L_mask]
    E_mids = E_mids[R_E_mask]
    
    reduced_L_mids = np.array([L_mids.mean()])
    reduced_E_mids = np.array([E_mids.mean()])
    
    new_r = NDHistogram(R_counts, [L_mids, E_mids])
    new_n = NDHistogram(N_counts, [L_mids, reduced_E_mids])
    new_phi = NDHistogram(phi_counts, [reduced_L_mids, E_mids])
    
    if sigma is None:
        return new_r, new_n, new_phi, None  # type: ignore
    else:
        sig_counts = sigma.counts.copy()
        if L_cut is not None:
            sig_counts = sig_counts[L_cut_mask, :]
        sig_counts = sig_counts[L_mask, :]
        new_sigma = NDHistogram(sig_counts, [L_mids, reduced_E_mids])
        return new_r, new_n, new_phi, new_sigma  # type: ignore


def _are_histograms_compatible(a: Histogram, b: Histogram) -> tuple[bool, str]:
    """Determine if two Histograms are compatible.

    Histograms are compatible if:
    - Counts have the same shape
    - Midpoints have the same shape and values

    :param a: First Histogram
    :type a: Histogram
    :param b: Second Histogram
    :type b: Histogram
    :return: Whether Histograms are compatible, and a reason if not compatible
    :rtype: tuple[bool, str]
    """
    a_size, *_ = a.counts.shape
    b_size, *_ = b.counts.shape
    if a_size != b_size:
        return False, "Sizes do not match"
    if not all(a.midpoints == b.midpoints):
        return False, "Bin midpoints do not match"
    return True, ""


def _are_histograms_compatible_2d(
    a: Histogram2D, b: Histogram, axis: int
) -> tuple[bool, str]:
    """Determine if a Histogram2D and a Histogram are compatible on a given axis.

    Histograms are compatible on an axis if:
    - Counts have the same shape
    - Midpoints have the same shape and values

    :param a: 2-dimensional histogram
    :type a: Histogram2D
    :param b: 1-dimensional histogram
    :type b: Histogram
    :return: Whether Histograms are compatible, and a reason if not compatible
    :rtype: tuple[bool, str]
    """
    a_size = a.counts.shape[axis]
    b_size, *_ = b.counts.shape
    if a_size != b_size:
        return False, "Sizes do not match"
    a_edges = a.x_midpoints if axis == 0 else a.y_midpoints
    if not all(a_edges == b.midpoints):
        return False, "Bin midpoints do not match"
    return True, ""


def _are_r_dimensions_close_enough(r: Histogram2D) -> bool:
    """Determine if dimensions of neutron response matrix (R) are sufficiently close.

    The GRAVEL algorithm requires that the dimensions of R must be within one order of
    magnitude for the algorithm to work.

    :param r: Neutron response matrix
    :type r: Histogram2D
    :return: Whether the dimensions of R are close enough to each other
    :rtype: bool
    """
    x_size, y_size = r.counts.shape
    return abs(log10(x_size) - log10(y_size)) <= 1


def _is_n_compatible(r: NDHistogram, n: NDHistogram) -> tuple[bool, str]:
    """Determine if neutron response spectrum N is compatible with response matrix R.
    
    N is considered compatible if:
        - the shape of N is either (m, 1) or (m, n), where (m, n) is the shape of R
        - the axis 0 midpoints of N must match those of R
        - the axis 1 midpoints of N must match those of R if that axis size is not 1 

    :param r: Neutron response matrix
    :type r: NDHistogram
    :param n: Neutron response spectrum
    :type n: NDHistogram
    :return: Whether N is compatible, and a reason if not compatible
    :rtype: tuple[bool, str]
    """    
    # N compatible: shape (m, 1) or (m, n), m mids must match, n mids must match if size matches
    R_m, R_n, *_ = r.shape
    N_m, N_n, *_ = n.shape
    shape_match = R_m == N_m and (N_n == 1 or N_n == R_n)
    if not shape_match:
        return False, "Shapes do not match"
    
    R_mids0, R_mids1, *_ = r.midpoints
    N_mids0, N_mids1, *_ = n.midpoints
    mids_match0 = len(R_mids0) == len(N_mids0) and (R_mids0 == N_mids0).all()
    if not mids_match0:
        return False, "Midpoints on axis 0 do not match"
    
    if N_n == R_n:
        mids_match1 = len(R_mids1) == len(N_mids1) and (R_mids1 == N_mids1).all()
        if not mids_match1:
            return False, "Midpoints on axis 1 do not match"
    
    return True, ""


def _is_phi_compatible(r: NDHistogram, phi: NDHistogram) -> tuple[bool, str]:
    """Determine if the neutron spectrum phi is compatible with the response matrix R.
    
    Phi is considered compatible if:
        - the shape of phi is (1, n), where (m, n) is the shape of R
        - the axis 1 midpoints of phi must match those of R

    :param r: Neutron response matrix
    :type r: NDHistogram
    :param phi: Neutron spectrum
    :type phi: NDHistogram
    :return: Whether N is compatible, and a reason if not compatible
    :rtype: tuple[bool, str]
    """    
    # Phi compatible: shape (1, n), n mids must match
    _, R_n, *_ = r.shape
    phi_m, phi_n, *_ = phi.shape
    shape_match = phi_m == 1 and R_n == phi_n
    if not shape_match:
        return False, "Shapes do not match"
    
    _, R_mids1, *_ = r.midpoints
    _, phi_mids1, *_ = phi.midpoints
    mids_match = len(R_mids1) == len(phi_mids1) and (R_mids1 == phi_mids1).all()
    if not mids_match:
        return False, "Midpoints on axis 1 do not match"
    
    return True, ""


def _new_are_r_dimensions_close_enough(r: NDHistogram) -> bool:
    """Determine if dimensions of neutron response matrix (R) are sufficiently close.

    The GRAVEL algorithm requires that the dimensions of R must be within one order of
    magnitude for the algorithm to work.

    :param r: Neutron response matrix
    :type r: NDHistogram
    :return: Whether the dimensions of R are close enough to each other
    :rtype: bool
    """
    if len(r.shape) != 2:
        raise ValueError("R must be 2-dimensional")
    x_size, y_size = r.shape
    return abs(log10(x_size) - log10(y_size)) <= 1
