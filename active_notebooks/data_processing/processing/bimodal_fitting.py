from multiprocessing.pool import IMapIterator, Pool
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from data_processing.dataframe_validation import (
    FIT_COLUMN_NAMES,
    FIT_ERROR_COLUMN_NAMES,
    DetectorDataframeColumn,
    EnergyColumn,
    get_df_col,
)
from data_processing.processing.figure_of_merit import FOM, bimodal
from data_processing.types import (
    BimodalBounds,
    BimodalParams,
    FitErrorResult,
    FitResult,
    GaussianParams,
    UnpackedFitErrorResult,
    UnpackedFitResult,
    unpack_bimodal_params,
)
from scipy.optimize import curve_fit


def split_params(params: BimodalParams) -> tuple[GaussianParams, GaussianParams]:
    """Separates bimodal function parameters into 2 sets, one per component gaussian

    Parameters
    ----------
    params: BimodalParams
        parameters of a bimodal function

    Returns
    -------
    lower_gaussian_params: GaussianParams
        Parameters of the lower gaussian (i.e. lower mu value)
    upper_gaussian_params: GaussianParams
        Parameters of the upper gaussian (i.e. higher mu value)
    """
    lower_gauss = GaussianParams(abs(params.mu1), abs(params.sigma1), abs(params.a1))
    upper_gauss = GaussianParams(abs(params.mu2), abs(params.sigma2), abs(params.a2))
    return lower_gauss, upper_gauss


def get_bimodal_fit(
    bins: np.ndarray, histogram_slice: np.ndarray, bounds: BimodalBounds
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
    lo_bounds, hi_bounds = bounds
    unpacked_lo = unpack_bimodal_params(lo_bounds)
    unpacked_hi = unpack_bimodal_params(hi_bounds)
    bounds_tuple = (unpacked_lo, unpacked_hi)

    params, cov = curve_fit(bimodal, bins, histogram_slice, bounds=bounds_tuple)

    params = BimodalParams(*params)
    gamma_params, neutron_params = split_params(params)

    return gamma_params, neutron_params, cov


class SliceFitter:
    # based on work by Steven EngelHardt
    # https://www.stevenengelhardt.com/2013/01/16/python-multiprocessing-module-and-closures/
    def __init__(
        self,
        psd_bin_midpoints: np.ndarray,
        energy_bin_edges: np.ndarray,
        default_bounds: BimodalBounds,
        bounds: Sequence[tuple[tuple[int, int], BimodalBounds]] | None = None,
    ):
        self.psd_bin_midpoints = psd_bin_midpoints
        self.energy_bin_edges = energy_bin_edges
        self.default_bounds = default_bounds
        self.bounds = bounds

    def __call__(self, numbered_slice) -> tuple[FitResult, FitErrorResult]:
        i, slice = numbered_slice
        slice_left_edge = self.energy_bin_edges[i]
        slice_right_edge = self.energy_bin_edges[i + 1]
        fit_bounds = self.default_bounds

        if self.bounds is not None:
            for i_range, bound in self.bounds:
                if i in range(*i_range):
                    fit_bounds = bound
        try:
            gamma_params, neutron_params, cov = get_bimodal_fit(
                self.psd_bin_midpoints, slice, fit_bounds
            )
        except RuntimeError:
            fit_result = FitResult(
                i, None, None, slice_left_edge, slice_right_edge, None
            )
            fit_error_result = FitErrorResult(
                i, None, None, slice_left_edge, slice_right_edge
            )
            return fit_result, fit_error_result

        fom = FOM(*gamma_params[:-1], *neutron_params[:-1])

        perr: BimodalParams = BimodalParams(*np.sqrt(np.diag(cov)))

        fit_result = FitResult(
            i, gamma_params, neutron_params, slice_left_edge, slice_right_edge, fom
        )
        fit_error_result = FitErrorResult(
            i, *split_params(perr), slice_left_edge, slice_right_edge
        )
        return fit_result, fit_error_result


def _unpack_slice_fit_pool_results(
    results: IMapIterator,
) -> tuple[list[UnpackedFitResult], list[UnpackedFitErrorResult]]:
    zipped_results = zip(*results)
    slice_params_from_zip: tuple[FitResult]
    slice_err_from_zip: tuple[FitErrorResult]
    slice_params_from_zip, slice_err_from_zip = zipped_results

    slice_params_sorted: list[FitResult] = sorted(
        list(slice_params_from_zip), key=lambda x: x[0]
    )
    slice_err_sorted: list[FitErrorResult] = sorted(
        list(slice_err_from_zip), key=lambda x: x[0]
    )

    slice_params_unpacked: list[UnpackedFitResult] = [
        (
            params.index,
            params.gamma_params.mu if params.gamma_params is not None else None,
            params.gamma_params.sigma if params.gamma_params is not None else None,
            params.gamma_params.a if params.gamma_params is not None else None,
            params.neutron_params.mu if params.neutron_params is not None else None,
            params.neutron_params.sigma if params.neutron_params is not None else None,
            params.neutron_params.a if params.neutron_params is not None else None,
            params.slice_left_edge,
            params.slice_right_edge,
            params.fom,
        )
        for params in slice_params_sorted
    ]
    slice_err_unpacked: list[UnpackedFitErrorResult] = [
        (
            err.index,
            err.gamma_params.mu if err.gamma_params is not None else None,
            err.gamma_params.sigma if err.gamma_params is not None else None,
            err.gamma_params.a if err.gamma_params is not None else None,
            err.neutron_params.mu if err.neutron_params is not None else None,
            err.neutron_params.sigma if err.neutron_params is not None else None,
            err.neutron_params.a if err.neutron_params is not None else None,
            err.slice_left_edge,
            err.slice_right_edge,
        )
        for err in slice_err_sorted
    ]

    return slice_params_unpacked, slice_err_unpacked


def scan_histogram_slices(
    histogram: np.ndarray,
    energy_bin_edges: np.ndarray,
    psd_bin_edges: np.ndarray,
    default_bounds: BimodalBounds,
    bounds: Sequence[tuple[tuple[int, int], BimodalBounds]] | None = None,
    start_idx: int = 0,
    end_idx: int | None = None,
    cores: int = 4,
    use_chunks: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Determines bimodal fit and FOM for every energy slice
    in a 2D PSD/Energy histogram

    Parameters
    ----------
    histogram: ndarray
        2D PSD/Energy histogram.
        The histogram shape should be [N, M], where N is the number of energy bins, and M is the number of PSD bins.
        This matches the output of Numpy's histogram2d function.
    energy_bin_edges: ndarray
        Edge values for each energy bin in the histogram.
        For N energy bins, there must be N+1 edges.
    psd_bin_edges: ndarray
        Edge values for each PSD bin in the histogram.
        For M PSD bins, there must be M+1 edges.
    default_bounds: BimodalBounds
        Default lower and upper bounds of fit parameters
    bounds: list[tuple[tuple[int, int], BimodalBounds]] | None, default None
        Allows custom bounds for slice ranges.
        Each list entry must have a tuple of start and stop indexes, and corresponding fit bounds.
        Bounds are used when the slice index falls within the start/stop range (start inclusive, stop exclusive).
        If index ranges overlap, the last matching range is used.
        If bounds is None, only default_bounds are used.
    start_idx: int, default 0
        Starting index (inclusive) of slice range to fit to bimodal
    end_idx: int | None, default None
        Ending index (exclusive) of slice range to fit to bimodal
    cores: int, default 4
        Number of logical cores present on this computer.
        Used to control parallelization of the scan.
    use_chunks: bool, default False
        Whether to split slices into larger chunks during parallelization.
        This can help speed up the scan on larger histograms.

    Returns
    -------
    fit_dataframe: DataFrame
        DataFrame of fit parameters including FOM (as columns) for each slice (as rows)
    error_dataframe: DataFrame
        DataFrame of (1 standard deviation) errors in fit parameters (as columns) for each slice (as rows)
    """
    end_idx = len(histogram) if end_idx is None else min(len(histogram), end_idx)
    pool_size = max(
        2 * cores, 4
    )  # based on https://jupyter-tutorial.readthedocs.io/en/stable/performance/multiprocessing.html

    psd_bin_left_edges = psd_bin_edges[:-1]
    psd_bin_right_edges = psd_bin_edges[1:]
    psd_bin_centers = (psd_bin_right_edges + psd_bin_left_edges) / 2
    energy_bin_edges_limited = energy_bin_edges[start_idx : end_idx + 1]

    energy_slices = list(histogram[start_idx:end_idx, :])

    if use_chunks:
        chunksize, extra = divmod(len(energy_slices), pool_size * 4)
        if extra > 0:
            chunksize += 1
    else:
        chunksize = 1

    pool = Pool(pool_size)
    results = pool.imap_unordered(
        SliceFitter(psd_bin_centers, energy_bin_edges_limited, default_bounds, bounds),
        enumerate(energy_slices),
        chunksize=chunksize,
    )
    slice_params, slice_err = _unpack_slice_fit_pool_results(results)

    df = pd.DataFrame(slice_params, columns=FIT_COLUMN_NAMES)
    err_df = pd.DataFrame(slice_err, columns=FIT_ERROR_COLUMN_NAMES)

    return df, err_df


def get_psd_energy_histogram(
    df: pd.DataFrame,
    energy_column: EnergyColumn,
    energy_width: float = 0.0150,
    psd_bin_count: int = 100,
    psd_min: float = 0.0,
    psd_max: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = get_df_col(df, energy_column)
    y = get_df_col(df, DetectorDataframeColumn.PSD)

    within_psd = y.between(psd_min, psd_max)
    y = y[within_psd == True].copy()
    x = x[within_psd == True].copy()

    x_bins: np.ndarray = np.linspace(0, x.max(), int(x.max() / energy_width) + 1)
    print(f"Energy width = {x_bins[1]-x_bins[0]} MeVee")
    y_bins: np.ndarray = np.linspace(psd_min, psd_max, psd_bin_count + 1)

    Z, xe, ye = np.histogram2d(x, y, bins=[x_bins, y_bins])
    return Z, xe, ye
