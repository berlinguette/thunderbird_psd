from multiprocessing.pool import Pool
from typing import Sequence

import numpy as np
import pandas as pd
from data_processing.dataframe_validation import (
    FIT_COLUMN_NAMES,
    FIT_ERROR_COLUMN_NAMES,
)
from data_processing.processing.slice_fitting.helpers import (
    unpack_slice_fit_pool_results,
)
from data_processing.processing.slice_fitting.slice_fitters import SliceFitterFactory
from data_processing.types import BimodalBounds, SliceFitStyle


def scan_histogram_slices(
    histogram: np.ndarray,
    energy_bin_edges: np.ndarray,
    psd_bin_edges: np.ndarray,
    default_bounds: BimodalBounds,
    bounds: Sequence[tuple[tuple[int, int], BimodalBounds]] | None = None,
    fit_style: SliceFitStyle = "bounds",
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
    slice_fitter_factory = SliceFitterFactory()
    slice_fitter = slice_fitter_factory.make_slice_fitter(
        fit_style, psd_bin_centers, energy_bin_edges_limited, default_bounds, bounds
    )
    # slice_fitter_factory = TestSliceFitterFactory()
    # slice_fitter = slice_fitter_factory.make_slice_fitter(0)
    results = pool.imap_unordered(
        slice_fitter,
        enumerate(energy_slices),
        chunksize=chunksize,
    )
    slice_params, slice_err = unpack_slice_fit_pool_results(results)

    df = pd.DataFrame(slice_params, columns=FIT_COLUMN_NAMES)
    err_df = pd.DataFrame(slice_err, columns=FIT_ERROR_COLUMN_NAMES)

    return df, err_df
