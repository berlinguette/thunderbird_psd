import pandas as pd
from data_processing.dataframe_validation import SliceFitDataframeColumn, get_df_col
from data_processing.types import VectorLikeFunction, WindowBorders
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.signal import savgol_filter


def generate_nasa_neutron_window(
    slice_fit_df: pd.DataFrame,
    window_offset: float = 0.2,
    sigma: float = 5,
    lower_energy_bound: float = 0.1966,
    recalculate_lower_energy_bound: bool = False,
) -> WindowBorders:
    energy_bin_midpoints = _get_energy_midpoints(slice_fit_df)
    bottom_border = _generate_nasa_window_bottom_border(slice_fit_df, sigma)
    top_border = _generate_nasa_window_top_border(slice_fit_df, sigma, window_offset)
    if recalculate_lower_energy_bound:
        lower_energy_bound = _generate_window_left_border(
            slice_fit_df, energy_bin_midpoints, (0.10, 0.35)
        )
    return WindowBorders(
        left=lower_energy_bound, bottom=bottom_border, top=top_border, right=None
    )


def generate_n_distro_neutron_window(
    slice_fit_df: pd.DataFrame,
    sigma: float = 3,
    fom_energy_range: tuple[float, float] = (0.10, 0.35),
) -> WindowBorders:
    energy_bin_left_edges = get_df_col(
        slice_fit_df, SliceFitDataframeColumn.SLICE_ENERGY_MINIMUM
    )
    energy_bin_midpoints = _get_energy_midpoints(slice_fit_df)
    left_border = _generate_window_left_border(
        slice_fit_df, energy_bin_midpoints, fom_energy_range=fom_energy_range
    )
    right_border = 0.688  # keVee, Compton edge + detector resolution
    bottom_border = _generate_new_window_bottom_border(
        slice_fit_df, energy_bin_left_edges, sigma
    )
    top_border = _generate_new_window_top_border(
        slice_fit_df, energy_bin_left_edges, sigma
    )
    return WindowBorders(
        left=left_border, right=right_border, bottom=bottom_border, top=top_border
    )


def generate_rectangle_neutron_window(left: float, bottom: float, width: float, height: float) -> WindowBorders:
    right = left + width
    top = bottom + height
    bottom_border = interp1d([left, right], [bottom, bottom], bounds_error=False)
    top_border = interp1d([left, right], [top, top], bounds_error=False)
    return WindowBorders(left=left, right=right, bottom=bottom_border, top=top_border)


def _generate_window_left_border(
    slice_fit_df: pd.DataFrame,
    energy_bin_midpoints: pd.Series,
    fom_energy_range: tuple[float, float],
) -> float:
    # create interp: x = energy_bin_midpoints, y = fom
    fom_series = get_df_col(slice_fit_df, SliceFitDataframeColumn.FOM)
    energy_fom_curve = interp1d(energy_bin_midpoints, fom_series - 1.27)
    # use root_scalar to find E where FOM = 1.27 and return
    root_results = root_scalar(energy_fom_curve, bracket=fom_energy_range)
    if root_results.converged:
        return root_results.root
    else:
        raise ValueError(
            f"Left border could not be found after {root_results.iterations}: "
            + f"{root_results.flag}"
        )


def _generate_nasa_window_bottom_border(
    slice_fit_df: pd.DataFrame, sigma: float
) -> VectorLikeFunction:
    return _generate_nasa_gamma_fn(slice_fit_df, sigma)


def _generate_nasa_window_top_border(
    slice_fit_df: pd.DataFrame, sigma: float, window_offset: float
) -> VectorLikeFunction:
    return _generate_nasa_gamma_fn(slice_fit_df, sigma, offset=window_offset)


def _generate_nasa_gamma_fn(
    slice_fit_df: pd.DataFrame, sigma: float, offset: float = 0
) -> VectorLikeFunction:
    gamma_mu_series = get_df_col(slice_fit_df, SliceFitDataframeColumn.GAMMA_MU)
    gamma_sigma_series = get_df_col(slice_fit_df, SliceFitDataframeColumn.GAMMA_SIGMA)

    gamma_border = savgol_filter(
        gamma_mu_series + sigma * gamma_sigma_series,
        window_length=21,
        polyorder=3,
    )  # reduce noise
    offset_border = gamma_border + offset  # type: ignore

    slice_xs = _get_energy_midpoints(slice_fit_df)
    border_fn: VectorLikeFunction = interp1d(
        slice_xs,
        offset_border,
        fill_value=(offset_border[0], offset_border[-1]),  # type: ignore
        bounds_error=False,
    )  # now it's a function!
    return border_fn


def _generate_new_window_bottom_border(
    slice_fit_df: pd.DataFrame, energy_bin_left_edges: pd.Series, sigma: float
) -> VectorLikeFunction:
    neutron_mu_series = get_df_col(slice_fit_df, SliceFitDataframeColumn.NEUTRON_MU)
    neutron_sigma_series = get_df_col(
        slice_fit_df, SliceFitDataframeColumn.NEUTRON_SIGMA
    )
    border_psd_values = neutron_mu_series - sigma * neutron_sigma_series
    fill_values = (border_psd_values.iloc[0], border_psd_values.iloc[-1])
    border: VectorLikeFunction = interp1d(
        energy_bin_left_edges,
        border_psd_values,
        kind="previous",
        fill_value=fill_values,  # type: ignore
        bounds_error=False,
    )
    return border


def _generate_new_window_top_border(
    slice_fit_df: pd.DataFrame, energy_bin_left_edges: pd.Series, sigma: float
) -> VectorLikeFunction:
    neutron_mu_series = get_df_col(slice_fit_df, SliceFitDataframeColumn.NEUTRON_MU)
    neutron_sigma_series = get_df_col(
        slice_fit_df, SliceFitDataframeColumn.NEUTRON_SIGMA
    )
    border_psd_values = neutron_mu_series + sigma * neutron_sigma_series
    fill_values = (border_psd_values.iloc[0], border_psd_values.iloc[-1])
    border: VectorLikeFunction = interp1d(
        energy_bin_left_edges,
        border_psd_values,
        kind="previous",
        fill_value=fill_values,  # type: ignore
        bounds_error=False,
    )
    return border


def _get_energy_midpoints(df: pd.DataFrame) -> pd.Series:
    slice_energy_min = get_df_col(df, SliceFitDataframeColumn.SLICE_ENERGY_MINIMUM)
    slice_energy_max = get_df_col(df, SliceFitDataframeColumn.SLICE_ENERGY_MAXIMUM)
    intervals = pd.Series(
        pd.arrays.IntervalArray.from_arrays(slice_energy_min, slice_energy_max),
        index=df.index,
    )
    midpoints = pd.Series(intervals.array.mid, index=df.index)  # type: ignore
    return midpoints
