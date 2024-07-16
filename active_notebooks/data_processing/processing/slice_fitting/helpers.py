from multiprocessing.pool import IMapIterator

import pandas as pd
from data_processing.types import (
    BimodalParams,
    FitErrorResult,
    FitResult,
    GaussianParams,
    UnpackedFitErrorResult,
    UnpackedFitResult,
)


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


def unpack_slice_fit_pool_results(
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


def find_failed_slices(
    df: pd.DataFrame,
    experiment_id: str,
    nan_total_threshold: int = 5,  # max bad slice fits total
    nan_window_threshold: int = 4,  # max bad slice fits in a "window"
    nan_rolling_window: int = 7,  # window size
) -> tuple[pd.DataFrame, list | None]:
    bad_slice_indexes = None
    row_is_nan = df.isna().any(axis=1)
    nan_rows = df[row_is_nan]

    if nan_rows.shape[0] > 0:
        nan_indexes = list(nan_rows.index)
        total_nan_rows = len(nan_indexes)
        rolling_nan_count = row_is_nan.rolling(window=nan_rolling_window).sum().max()

        print(f"Fit issues in {experiment_id}")
        print(f"Fit failed on following slice indexes: {nan_indexes}")

        if (
            total_nan_rows > nan_total_threshold
            or rolling_nan_count > nan_window_threshold
        ):
            print(f"Experiment {experiment_id} could not be classified")
            print(f"Total failed slices: {total_nan_rows}")
            print(
                f"Max failed slices in a {nan_rolling_window} slice window:"
                + f" {rolling_nan_count}"
            )
            bad_slice_indexes = nan_indexes

        df = df.dropna().copy()
    return df, bad_slice_indexes
