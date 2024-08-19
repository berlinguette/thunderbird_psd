"""This module is responsible for generating neutron windows."""
import pandas as pd
from data_processing.dataframe_validation import SliceFitDataframeColumn, get_df_col
from data_processing.types import VectorLikeFunction, WindowBorders
from data_processing.helpers import get_midpoints_from_min_max_series
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
    """Generates a neutron window based on the NASA method. 
    See Baramsai, B., Park, B., Becks, M. D., Chait, A. & Hendricks, R. Fast neutron spectroscopy with organic scintillation detectors in a high-radiation environment. NASA (2020).
    The lower border is made from lower border points calculated per slice.
    These points are found by calculating mu_gamma + n * sigma_gamma.
    The upper border is made by adding a vertical offset to the lower border.

    :param slice_fit_df: Dataframe with bimodal fit data for each energy slice of a PSD/energy histogram
    :type slice_fit_df: pd.DataFrame
    :param window_offset: Offset (in PSD units) between lower and upper window borders, defaults to 0.2
    :type window_offset: float, optional
    :param sigma: Multiplier value to use when finding lower bounds points, defaults to 5
    :type sigma: float, optional
    :param lower_energy_bound: Energy value to use as the left window border, defaults to 0.1966
    :type lower_energy_bound: float, optional
    :param recalculate_lower_energy_bound: Whether to calculate the lower energy bound based on slice FOM values, defaults to False
    :type recalculate_lower_energy_bound: bool, optional
    :return: Window borders using the NASA method
    :rtype: WindowBorders
    """
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
    """Generates a neutron window based on the neutron Gaussian distribution.
    The lower border points are calculated using mu_n - n * sigma_n.
    The upper border points are calculated using mu_n + n * sigma_n.

    :param slice_fit_df: Dataframe with bimodal fit data for each energy slice of a PSD/energy histogram
    :type slice_fit_df: pd.DataFrame
    :param sigma: Multiplier value to use when finding upper and lower bounds points, defaults to 3
    :type sigma: float, optional
    :param fom_energy_range: Energy range (in MeVee) where the FOM threshold (and left border) should be found, defaults to (0.10, 0.35)
    :type fom_energy_range: tuple[float, float], optional
    :return: Window borders using the neutron distribution method
    :rtype: WindowBorders
    """
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
    """Generate a simple rectangular neutron window.

    :param left: Energy value of left border (in MeVee)
    :type left: float
    :param bottom: PSD value of bottom border
    :type bottom: float
    :param width: Width of neutron window (in MeVee)
    :type width: float
    :param height: Height of neutron window (in PSD units)
    :type height: float
    :return: Window borders as simple rectangles
    :rtype: WindowBorders
    """
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
    slice_fit_df: pd.DataFrame, sigma: float, offset: float = 0, use_filter: bool = False
) -> VectorLikeFunction:
    gamma_mu_series = get_df_col(slice_fit_df, SliceFitDataframeColumn.GAMMA_MU)
    gamma_sigma_series = get_df_col(slice_fit_df, SliceFitDataframeColumn.GAMMA_SIGMA)

    gamma_border = (gamma_mu_series + sigma * gamma_sigma_series).to_numpy(copy=True)
    if use_filter:
        gamma_border = savgol_filter(
            gamma_border,
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
    index = df.index
    return get_midpoints_from_min_max_series(slice_energy_min, slice_energy_max, index)
