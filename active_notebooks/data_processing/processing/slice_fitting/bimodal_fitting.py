from typing import Any

import numpy as np
from data_processing.helpers.get_midpoints_from_bins import get_midpoints_from_bins
from data_processing.processing.figure_of_merit import bimodal
from data_processing.processing.slice_fitting.helpers import split_params
from data_processing.types import (
    BimodalBounds,
    BimodalParams,
    GaussianParams,
    unpack_bimodal_params,
)
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def get_bimodal_fit_guess(
    bins: np.ndarray, histogram_slice: np.ndarray
) -> BimodalParams:
    # TODO use peak finder to get peak positions
    bin_mids = get_midpoints_from_bins(bins)

    # TODO check with Sergey re: best parameters for this
    peak_search_settings: dict[str, Any] = dict(
        height=None,
        threshold=None,
        distance=None,  # TODO add more?
    )
    peaks, properties = find_peaks(histogram_slice, **peak_search_settings)
    if len(peaks) > 2:
        pass  # TODO what to do with >2 peaks?
    # TODO get bimodal params for each peak
    left_peak_idx = peaks[0]
    A_left = histogram_slice[left_peak_idx]
    mu_left = bin_mids[left_peak_idx]
    sigma_left = 0.05  # STUB how to get this from peak width?
    if len(peaks) == 1:
        A_right = 0
        mu_right = mu_left + 0.2
        sigma_right = sigma_left
    else:
        right_peak_idx = peaks[1]
        A_right = histogram_slice[right_peak_idx]
        mu_right = bin_mids[right_peak_idx]
        sigma_right = 0.025  # STUB how to get this from peak width?

    return BimodalParams(mu_left, sigma_left, A_left, mu_right, sigma_right, A_right)


def get_bimodal_fit(
    bins: np.ndarray,
    histogram_slice: np.ndarray,
    guess: BimodalParams | None = None,
    bounds: BimodalBounds | None = None,
) -> tuple[GaussianParams, GaussianParams, np.ndarray]:
    """Fits a histogram slice to a bimodal distribution

    Parameters
    ----------
    bins: ndarray
        Lower bounds of each PSD bin in the histogram
    histogram_slice: ndarray
        Slice of the 2D PSD/Energy histogram taken for a specific energy (i.e. PSD vs Counts)
    bounds: BimodalBounds
        Lower and upper bounds of fit parameters for this slice

    Returns
    -------
    gamma_params: GaussianParams
        Parameters of the gaussian fit for gamma rays
    neutron_params: GaussianParams
        Parameters of the gaussian fit for neutrons
    cov: ndarray
        Estimated covariance of all bimodial parameters
    """
    bounds_tuple = None
    if bounds is not None:
        lo_bounds, hi_bounds = bounds
        unpacked_lo = unpack_bimodal_params(lo_bounds)
        unpacked_hi = unpack_bimodal_params(hi_bounds)
        bounds_tuple = (unpacked_lo, unpacked_hi)

    params, cov = curve_fit(
        bimodal, bins, histogram_slice, p0=guess, bounds=bounds_tuple
    )

    params = BimodalParams(*params)
    gamma_params, neutron_params = split_params(params)

    return gamma_params, neutron_params, cov
