import numpy as np
import pandas as pd
from typing import Optional
from scipy.optimize import curve_fit
from data_processing.processing.figure_of_merit import bimodal, FOM
from multiprocessing.pool import Pool
from data_processing.dataframe_validation import DataframeColumn, get_df_col

BimodalParams = tuple[float, float, float, float, float, float]
GaussianParams = tuple[float, float, float]
BimodalBounds = tuple[BimodalParams, BimodalParams]

def split_params(
    params: BimodalParams
) -> tuple[GaussianParams, GaussianParams]:
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
    params = tuple([abs(param) for param in params])
    return params[0:3], params[3:]

def get_bimodal_fit(
    bins: np.ndarray, 
    histogram_slice: np.ndarray, 
    bounds: BimodalBounds
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
    params, cov = curve_fit(
        bimodal,
        bins,
        histogram_slice,
        bounds=bounds,
    )
    
    gamma_params, neutron_params = split_params(params)
    
    return gamma_params, neutron_params, cov


class SliceFitter:
    # based on work by Steven EngelHardt
    # https://www.stevenengelhardt.com/2013/01/16/python-multiprocessing-module-and-closures/
    def __init__(
        self, 
        bins: np.ndarray, 
        default_bounds: BimodalBounds,
        bounds: list[tuple[tuple[int, int], BimodalBounds]] | None = None
    ):
        self.bins = bins
        self.default_bounds = default_bounds
        self.bounds = bounds
        
    def __call__(self, numbered_slice):
        i, slice = numbered_slice
        fit_bounds = self.default_bounds

        if self.bounds is not None:
            for i_range, bound in self.bounds:
                if i in range(*i_range):
                    fit_bounds = bound
        try:
            gamma_params, neutron_params, cov = get_bimodal_fit(self.bins, slice, fit_bounds)
        except RuntimeError as err:
            print(f"Slice {i} fit failed: {err}")
            return None

        fom = FOM(*gamma_params[:-1], *neutron_params[:-1])

        perr = np.sqrt(np.diag(cov))

        return (i, *gamma_params, *neutron_params, fom), (i, *perr)


def scan_histogram_slices(
    bins: np.ndarray, 
    histogram: np.ndarray, 
    default_bounds: BimodalBounds,
    bounds: list[tuple[tuple[int, int], BimodalBounds]] | None = None, 
    start_idx: int = 0, 
    end_idx: int | None = None,
    cores: int = 4,
    use_chunks: bool = False
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    """Determines bimodal fit and FOM for every energy slice 
    in a 2D PSD/Energy histogram
    
    Parameters
    ----------
    bins: ndarray
        Lower bounds of each PSD bin in the histogram
    histogram: ndarray
        2D PSD/Energy histogram
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
    pool_size = max(2*cores, 4)  # based on https://jupyter-tutorial.readthedocs.io/en/stable/performance/multiprocessing.html

    energy_slices = list(histogram[:, start_idx:end_idx].T)

    if use_chunks:
        chunksize, extra = divmod(len(energy_slices), pool_size*4)
        if extra > 0:
            chunksize += 1
    else:
        chunksize = 1
    
    pool = Pool(pool_size)
    results = pool.imap_unordered(
        SliceFitter(bins, default_bounds, bounds), 
        enumerate(energy_slices), 
        chunksize=chunksize)
    print([len(result) if result is None else "None" for result in results])
    if any([result is None for result in results]):
        return None
    slice_params, slice_err = zip(*results)
    
    slice_params = sorted(list(slice_params), key=lambda x: x[0])
    slice_err = sorted(list(slice_err), key=lambda x: x[0])

    columns = ['i', 'mu1', 'sigma1', 'a1', 'mu2', 'sigma2', 'a2']
    df = pd.DataFrame(slice_params, columns=columns + ['fom'])
    err_df = pd.DataFrame(slice_err, columns=columns)

    return df, err_df


def get_psd_energy_histogram(
    df: pd.DataFrame, 
    resolution: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = get_df_col(df, DataframeColumn.CALIB_ENERGY)
    y = get_df_col(df, DataframeColumn.PSD)
    Z, xe, ye = np.histogram2d(x, y, resolution)
    return Z, xe, ye
