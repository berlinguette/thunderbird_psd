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
from data_processing.helpers import get_midpoints_from_bins
from data_processing.processing.slice_fitting.slice_fitters import SliceFitterFactory
from data_processing.types import BimodalBounds, SliceFitStyle, BoundsSequence


def scan_histogram_slices(
    histogram: np.ndarray,
    energy_bin_edges: np.ndarray,
    psd_bin_edges: np.ndarray,
    fit_style: SliceFitStyle = "bounds",
    default_bounds: BimodalBounds | None = None,
    bounds: BoundsSequence | None = None,
    start_idx: int = 0,
    end_idx: int | None = None,
    cores: int = 4,
    use_chunks: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Determines bimodal fit and FOM for every energy slice in a 2D PSD/Energy histogram
    The histogram should have the shape [N, M], where N is the number of energy bins, and M is the number of PSD bids.
    This matches the output of numpy's histogram2d function.

    :param histogram: PSD/energy 2D histogram
    :type histogram: np.ndarray
    :param energy_bin_edges: Energy bin edges, with shape [N+1]
    :type energy_bin_edges: np.ndarray
    :param psd_bin_edges: PSD bin edges, with shape [M+1]
    :type psd_bin_edges: np.ndarray
    :param fit_style: Style of best fit to use, defaults to "bounds"
    :type fit_style: SliceFitStyle, optional
    :param default_bounds: Default lower and upper bounds of fit parameters to use on slices where no bounds are defined, defaults to None
    :type default_bounds: BimodalBounds | None, optional
    :param bounds: Information on bounds to use for specified ranges of slices, defaults to None
    :type bounds: BoundsSequence | None, optional
    :param start_idx: Index of slice to start from, defaults to 0
    :type start_idx: int, optional
    :param end_idx: Index of slice to end on, defaults to None
    :type end_idx: int | None, optional
    :param cores: CPU cores to use, defaults to 4
    :type cores: int, optional
    :param use_chunks: Whether to split slices into larger chunks during parallelization, defaults to False
    :type use_chunks: bool, optional
    :return: Dataframe of fit parameters, dataframe of fit errors
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    end_idx = len(histogram) if end_idx is None else min(len(histogram), end_idx)
    pool_size = max(
        2 * cores, 4
    )  # based on https://jupyter-tutorial.readthedocs.io/en/stable/performance/multiprocessing.html

    # psd_bin_left_edges = psd_bin_edges[:-1]
    # psd_bin_right_edges = psd_bin_edges[1:]
    # psd_bin_centers = (psd_bin_right_edges + psd_bin_left_edges) / 2
    psd_bin_centers = get_midpoints_from_bins(psd_bin_edges)
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
