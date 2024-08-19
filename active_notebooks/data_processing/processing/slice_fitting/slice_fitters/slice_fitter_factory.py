from data_processing.types import SliceFitStyle
from numpy import ndarray
from data_processing.types import BimodalBounds, BoundsSequence

from .bounds_slice_fitter import BoundsSliceFitter
from .peak_finder_slice_fitter import PeakFinderSliceFitter
from .slice_fitter import SliceFitter


class SliceFitterFactory:
    """Factory for generating slice fitters.
    """
    def make_slice_fitter(
        self,
        style: SliceFitStyle,
        psd_bin_midpoints: ndarray,
        energy_bin_edges: ndarray,
        default_bounds: BimodalBounds | None = None,
        bounds: BoundsSequence | None = None
    ) -> SliceFitter:
        """_summary_

        :param style: Style of slice fitter
        :type style: SliceFitStyle
        :param psd_bin_midpoints: Array of all PSD bin midpoints
        :type psd_bin_midpoints: np.ndarray
        :param energy_bin_edges: Array of all energy bin edges (n+1 edges for n energy bins)
        :type energy_bin_edges: np.ndarray
        :param default_bounds: Bounds to use for any slice without defined bounds, defaults to None
        :type default_bounds: BimodalBounds | None, optional
        :param bounds: Information on bounds to use for specified ranges of slices, defaults to None
        :type bounds: BoundsSequence | None, optional
        :raises ValueError: if a given style is invalid
        :return: Desired slice fitter
        :rtype: SliceFitter
        """
        if style == "bounds":
            return BoundsSliceFitter(
                psd_bin_midpoints, energy_bin_edges, default_bounds, bounds
            )
        elif style == "peak_finder":
            return PeakFinderSliceFitter(
                psd_bin_midpoints, energy_bin_edges, default_bounds, bounds
            )
        else:
            raise ValueError(f"Invalid style {style}")
