from math import log10
from typing import TypedDict, TypeVar

import numpy as np
from numpy import ndarray


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
    r: NDHistogram,
    n: NDHistogram,
    phi: NDHistogram,
    sigma: NDHistogram | None = None,
) -> NDHistogram:
    """Calculate the weight factor for this iteration of the GRAVEL algorithm.

    :param big_r: Neutron response matrix
    :type big_r: NDHistogram
    :param big_n: Neutron response spectrum
    :type big_n: NDHistogram
    :param new_phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type new_phi: NDHistogram
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
    close = _are_r_dimensions_close_enough(r)
    if not close:
        raise ValueError(
            f"Dimensions of R {r.shape} are not within 1 order of magnitude"
        )

    _R = r.counts
    _phi = phi.counts
    _N = n.counts
    _sigma = sigma.counts

    numer = _R * _phi
    denom = r_dot(r, phi).counts
    right = _nan_divide(np.square(_N), np.square(_sigma))
    result = (numer / denom) * right

    return NDHistogram(result, r.midpoints)


def next_phi(
    r: NDHistogram,
    n: NDHistogram,
    phi: NDHistogram,
    sigma: NDHistogram | None = None,
) -> NDHistogram:
    """Calculate the next neutron spectrum for this iteration of the GRAVEL algorithm.

    :param big_r: Neutron response matrix
    :type big_r: NDHistogram
    :param big_n: Neutron response spectrum
    :type big_n: NDHistogram
    :param phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type phi: NDHistogram
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
    close = _are_r_dimensions_close_enough(r)
    if not close:
        raise ValueError(
            f"Dimensions of R {r.shape} are not within 1 order of magnitude"
        )

    w = weight_factor(r, n, phi, sigma=sigma)

    _R = r.counts
    _phi = phi.counts
    _N = n.counts
    _W = w.counts

    ln_denom = r_dot(r, phi).counts
    ln_result = np.log(_nan_divide(_N, ln_denom))
    exp_numer = np.nansum(_W * ln_result, axis=0, keepdims=True)
    exp_denom = np.nansum(_W, axis=0, keepdims=True)

    exp_frac = _nan_divide(exp_numer, exp_denom)
    exp_result = np.exp(exp_frac)
    result = _phi * exp_result

    r_mid0, r_mid1, *_ = r.midpoints
    phi_mid0 = np.array([r_mid0.mean])
    phi_mid1 = r_mid1.copy()
    phi_mids = [phi_mid0, phi_mid1]

    return NDHistogram(result, phi_mids)


def stopping_criteria(
    r: NDHistogram,
    n: NDHistogram,
    phi: NDHistogram,
    sigma: NDHistogram | None = None,
) -> float:
    """Calculate the stopping criteria value for this iteration of the GRAVEL algorithm.

    The stopping criteria value is chi^2 divided by the degrees of freedom.
    Degrees of freedom for neutron response matrix R is (m-1)*(n-1), where m and n are
    the dimensions of R.

    :param r: Neutron response matrix
    :type r: NDHistogram
    :param n: Neutron response spectrum
    :type n: NDHistogram
    :param phi: Neutron spectrum calculated in the previous GRAVEL iteration
    :type phi: NDHistogram
    :param sigma: Estimation of error in the neutron response spectrum,
        defaults to None (root of neutron response spectrum)
    :type sigma: NDHistogram | None, optional
    :raises ValueError: if dimensions do not match (N/sigma with R's x-axis, Phi with
        R's y-axis)
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
    close = _are_r_dimensions_close_enough(r)
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

    sum_numer = np.nansum(_R * _phi, axis=1, keepdims=True)
    numer = np.square(sum_numer - _N)
    frac = _nan_divide(numer, np.square(_sigma))
    chi_sq = np.nansum(frac, axis=0, keepdims=True)
    chi_sq = chi_sq.flatten().tolist()[0]
    result = chi_sq / DOF

    return result


def unfold_spectrum(
    r: NDHistogram,
    n: NDHistogram,
    phi0: NDHistogram | None = None,
    sigma: NDHistogram | None = None,
    L_cut: float | None = None,
    tolerance: float = 0.01,
    max_iterations: int = 500,
    full_info: bool = False,
) -> tuple[NDHistogram, UnfoldingProcessInfo | None]:
    """Unfold neutron response spectrum into neutron spectrum.

    This function performs the GRAVEL algorithm iteratively, as seen in [CITATION HERE].

    :param r: Neutron response matrix
    :type r: NDHistogram
    :param n: Neutron response spectrum
    :type n: NDHistogram
    :param phi0: Starting neutron spectrum, defaults to None
        (i.e. 1 for all energy bins)
    :type phi0: NDHistogram | None, optional
    :param sigma: Estimation of error in the neutron response spectrum,
        defaults to None (root of neutron response spectrum)
    :type sigma: NDHistogram | None, optional
    :param L_cut: if not None, light outputs to remove from N before stripping zeroes;
        defaults to None (i.e. no removal)
    :type L_cut: float | None, optional
    :param L_cut: Light levels to cut out
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

    new_r, new_n, new_sigma = cut_low_l(r, n, sigma=sigma, L_cut=L_cut)
    new_phi0 = NDHistogram(phi0.counts.copy(), [mids.copy() for mids in phi0.midpoints])

    compatible, reason = _is_n_compatible(new_r, new_n)
    if not compatible:
        raise ValueError(f"R and N are incompatible: {reason}")
    compatible, reason = _is_phi_compatible(new_r, new_phi0)
    if not compatible:
        raise ValueError(f"R and Phi are incompatible: {reason}")
    compatible, reason = _is_n_compatible(new_r, new_sigma)
    if not compatible:
        raise ValueError(f"R and Sigma are incompatible: {reason}")
    close = _are_r_dimensions_close_enough(new_r)
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

    chi_n = stopping_criteria(new_r, new_n, new_phi0, sigma=new_sigma)
    phi_k = NDHistogram(
        new_phi0.counts.copy(), [mids.copy() for mids in new_phi0.midpoints]
    )
    chi_last = chi_n
    delta_chi_last = 1
    delta_delta = 1

    while delta_delta > tolerance:
        weight = weight_factor(new_r, new_n, phi_k, sigma=new_sigma)
        phi_k = next_phi(new_r, new_n, phi_k, sigma=new_sigma)
        chi_n = stopping_criteria(new_r, new_n, phi_k, sigma=new_sigma)

        delta_chi = chi_n - chi_last
        delta_delta = abs(delta_chi - delta_chi_last)
        chi_last = chi_n
        delta_chi_last = delta_chi

        if full_info:
            chis.append(chi_n)
            phis.append(phi_k)
            weights.append(weight)
            errors.append(delta_delta)

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
    return phi_k, unfolding_info


T = TypeVar("T", NDHistogram, None)


def clean_data(
    r: NDHistogram,
    n: NDHistogram,
    phi: NDHistogram,
    sigma: T = None,
    L_cut: float | None = None,
) -> tuple[NDHistogram, NDHistogram, NDHistogram, T]:
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
        defaults to None (i.e. no removal)
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
    R_counts = R_counts[R_L_mask, :]
    R_counts = R_counts[:, R_E_mask]
    N_counts = N_counts[R_L_mask, :]
    if N_counts.shape[1] != 1:
        N_counts = N_counts[:, R_E_mask]
    phi_counts = phi_counts[:, R_E_mask]
    L_mids = L_mids[R_L_mask]
    E_mids = E_mids[R_E_mask]

    reduced_L_mids = np.array([L_mids.mean()])
    reduced_E_mids = np.array([E_mids.mean()])

    new_r = NDHistogram(R_counts, [L_mids, E_mids])
    if N_counts.shape[1] == 1:
        new_n = NDHistogram(N_counts, [L_mids, reduced_E_mids])
    else:
        new_n = NDHistogram(N_counts, [L_mids, E_mids])
    new_phi = NDHistogram(phi_counts, [reduced_L_mids, E_mids])

    if sigma is None:
        return new_r, new_n, new_phi, None  # type: ignore
    else:
        sig_counts = sigma.counts.copy()
        if L_cut is not None:
            sig_counts = sig_counts[L_cut_mask, :]
        sig_counts = sig_counts[L_mask, :]
        sig_counts = sig_counts[R_L_mask, :]
        if sig_counts.shape[1] != 1:
            sig_counts = sig_counts[:, R_E_mask]
            sig_mids = [L_mids, E_mids]
        else:
            sig_mids = [L_mids, reduced_E_mids]
        new_sigma = NDHistogram(sig_counts, sig_mids)
        return new_r, new_n, new_phi, new_sigma  # type: ignore


def cut_low_l(
    r: NDHistogram, n: NDHistogram, L_cut: float | None = None, sigma: T = None
) -> tuple[NDHistogram, NDHistogram, T]:
    """Cuts out lower L values from relevant histograms.

    Any L channels that are lower than the provided cutoff value are identified. The
    corresponding rows and midpoints are then removed from the provided histograms.

    The L_cut value is made optional for compatibility with other functions in this
    module. If not provided, no cut is made.

    :param r: Neutron response matrix
    :type r: NDHistogram
    :param n: Neutron response spectrum
    :type n: NDHistogram
    :param L_cut: Light output cutoff value, defaults to None (i.e. no cut)
    :type L_cut: float | None, optional
    :param sigma: Optional estimation of error in the neutron response spectrum,
        defaults to None
    :type sigma: NDHistogram | None, optional
    :raises ValueError: if dimensions do not match (N or sigma with R's axis 0)
    :return: L-cut R, N and sigma (if provided, or None if not)
    :rtype: tuple[NDHistogram, NDHistogram, NDHistogram] (if sigma is provided)
    :rtype: tuple[NDHistogram, NDHistogram, None] (if sigma is not provided)
    """
    compatible, reason = _is_n_compatible(r, n)
    if not compatible:
        raise ValueError(f"R and N are incompatible: {reason}")
    if sigma is not None:
        compatible, reason = _is_n_compatible(r, sigma)
        if not compatible:
            raise ValueError(f"R and Sigma are incompatible: {reason}")

    R_counts = r.counts.copy()
    R_mids = [mids.copy() for mids in r.midpoints]
    L_mids, R_E_mids, *_ = R_mids
    N_counts = n.counts.copy()
    N_mids = [mids.copy() for mids in n.midpoints]
    _, N_E_mids, *_ = N_mids

    if sigma is None:
        new_sigma = None
    else:
        sigma_counts = sigma.counts.copy()
        sigma_mids = [mids.copy() for mids in sigma.midpoints]
        new_sigma = NDHistogram(sigma_counts, sigma_mids)

    if L_cut is not None:
        L_cut_mask = L_mids >= L_cut
        N_counts = N_counts[L_cut_mask, :]
        R_counts = R_counts[L_cut_mask, :]
        L_mids = L_mids[L_cut_mask]
        if new_sigma is not None:
            sigma_counts = new_sigma.counts[L_cut_mask, :]
            _, sigma_E_mids, *_ = new_sigma.midpoints
            new_sigma = NDHistogram(sigma_counts, [L_mids, sigma_E_mids])

    new_r = NDHistogram(R_counts, [L_mids, R_E_mids])
    new_n = NDHistogram(N_counts, [L_mids, N_E_mids])

    return (new_r, new_n, new_sigma)  # type: ignore


def r_dot(r: NDHistogram, phi: NDHistogram) -> NDHistogram:
    """Finds the sum (over axis 1) of the product between R and Phi.

    This is also the dot product between R and Phi.

    If the shape of R is (m, n), the shape of Phi must be (1, n) or an exception will be
    raised. As well, the midpoints on axis 1 must be compatible.

    The shape of the output will be (m, 1).

    If any values in R or phi are NaN, they will be ignored during summation.

    :param r: Neutron response matrix
    :type r: NDHistogram
    :param phi: Current neutron spectrum estimate
    :type phi: NDHistogram
    :raises ValueError: if R and Phi are not compatible
    :return: Dot product of R and Phi
    :rtype: NDHistogram
    """
    compatible, reason = _is_phi_compatible(r, phi)
    if not compatible:
        raise ValueError(f"R and Phi are incompatible: {reason}")

    _r = r.counts
    _phi = phi.counts

    L_mids, E_mids = r.midpoints
    reduced_E_mids = np.array([E_mids.mean()])

    r_dot_counts = np.nansum(_r * _phi, axis=1, keepdims=True)
    result = NDHistogram(r_dot_counts, [L_mids, reduced_E_mids])

    return result


def _nan_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Divides two NDArrays (a/b) such that all invalid divisions produce NaN values.

    As well, all warnings related to dividing by zero are suppressed.

    :param a: divident array
    :type a: np.ndarray
    :param b: divisor array
    :type b: np.ndarray
    :return: quotient array
    :rtype: np.ndarray
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = a / b
    quotient = np.nan_to_num(quotient, nan=np.nan, posinf=np.nan, neginf=np.nan)
    return quotient


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
    # mids_match0 = len(R_mids0) == len(N_mids0) and (R_mids0 == N_mids0).all()
    # if not mids_match0:
    #     return False, "Midpoints on axis 0 do not match"
    mids0_exact_match = (R_mids0 == N_mids0).all()
    mids0_close_match = np.isclose(R_mids0, N_mids0).all()
    mids0_match = len(R_mids0) == len(N_mids0) and (
        mids0_exact_match or mids0_close_match
    )
    if mids0_match:
        if mids0_close_match and not mids0_exact_match:
            N_mids0 = R_mids0
    else:
        return False, "Midpoints on axis 0 do not match"

    if N_n == R_n:
        # mids_match1 = len(R_mids1) == len(N_mids1) and (R_mids1 == N_mids1).all()
        mids1_exact_match = (R_mids1 == N_mids1).all()
        mids1_close_match = np.isclose(R_mids1, N_mids1).all()
        mids1_match = len(R_mids1) == len(N_mids1) and (
            mids1_exact_match or mids1_close_match
        )
        if mids1_match:
            if mids1_close_match and not mids1_exact_match:
                N_mids1 = R_mids1
        else:
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
    # mids_match = len(R_mids1) == len(phi_mids1) and (R_mids1 == phi_mids1).all()
    # if not mids_match:
    #     return False, "Midpoints on axis 1 do not match"
    
    mids1_exact_match = (R_mids1 == phi_mids1).all()
    mids1_close_match = np.isclose(R_mids1, phi_mids1).all()
    mids1_match = len(R_mids1) == len(phi_mids1) and (
        mids1_exact_match or mids1_close_match
    )
    if mids1_match:
        if mids1_close_match and not mids1_exact_match:
            N_mids1 = R_mids1
    else:
            return False, "Midpoints on axis 1 do not match"

    return True, ""


def _are_r_dimensions_close_enough(r: NDHistogram) -> bool:
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
