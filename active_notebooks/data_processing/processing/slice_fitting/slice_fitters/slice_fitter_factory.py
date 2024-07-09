from data_processing.types import SliceFitStyle
from numpy import ndarray
from data_processing.types import BimodalBounds, BoundsSequence

from .bounds_slice_fitter import BoundsSliceFitter
from .peak_finder_slice_fitter import PeakFinderSliceFitter
from .slice_fitter import SliceFitter


class SliceFitterFactory:
    def make_slice_fitter(
        self,
        style: SliceFitStyle,
        psd_bin_midpoints: ndarray,
        energy_bin_edges: ndarray,
        default_bounds: BimodalBounds | None = None,
        bounds: BoundsSequence | None = None
    ) -> SliceFitter:
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
