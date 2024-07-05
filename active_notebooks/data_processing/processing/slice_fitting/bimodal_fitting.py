from typing import Any
from math import ceil

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
from scipy.signal import find_peaks, peak_widths


def get_bimodal_fit_guess(
    bins: np.ndarray, histogram_slice: np.ndarray
) -> BimodalParams:
    bin_mids = get_midpoints_from_bins(bins)
    sample_psd_rate = bin_mids[1] - bin_mids[0]  # assume mids increase linearly

    slice_samples = histogram_slice.shape[0]
    peak_search_settings: dict[str, Any] = dict(
        prominence=max(histogram_slice) / 10,
        wlen=ceil(slice_samples/2),
        distance=slice_samples/4,
    )
    peaks, properties = find_peaks(histogram_slice, **peak_search_settings)
    
    if len(peaks) > 2:
        # pick out 2 peaks with lowest index
        indexes = np.argsort(peaks)
        peaks = peaks[indexes[:2]]
        
        new_properties = {}
        for k, v in properties.items():
            new_v = v[indexes[:2]]
            new_properties[k] = new_v
        properties = new_properties
    
    prominences = properties.get("prominences")
    left_bases = properties.get("left_bases")
    right_bases = properties.get("right_bases")
    prom_data_is_none = any([x is None for x in [prominences, left_bases, right_bases]])
    prominence_data = None if prom_data_is_none else (prominences, left_bases, right_bases)
    peak_width_settings: dict[str, Any] = dict(
        rel_height=0.5,
        prominence_data=prominence_data,
    )
    widths, *_ = peak_widths(histogram_slice, peaks, **peak_width_settings)
    
    # TODO get bimodal params for each peak
    left_peak_idx = peaks[0]
    left_peak_width = widths[0]
    A_left = histogram_slice[left_peak_idx]
    mu_left = bin_mids[left_peak_idx]
    sigma_left = sample_psd_rate * left_peak_width
    
    if len(peaks) == 1:
        A_right = 0
        mu_right = 0
        sigma_right = 0
    else:
        right_peak_idx = peaks[1]
        right_peak_width = widths[1]
        A_right = histogram_slice[right_peak_idx]
        mu_right = bin_mids[right_peak_idx]
        sigma_right = sample_psd_rate * right_peak_width

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
