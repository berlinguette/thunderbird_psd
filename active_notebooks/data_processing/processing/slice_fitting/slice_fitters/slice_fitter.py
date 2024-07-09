from abc import ABC, abstractmethod

import numpy as np
from data_processing.types import BimodalBounds, NumberedSlice, SliceFitResult, BoundsSequence


class SliceFitter(ABC):
    # based on work by Steven EngelHardt
    # https://www.stevenengelhardt.com/2013/01/16/python-multiprocessing-module-and-closures/

    def __init__(
        self,
        psd_bin_midpoints: np.ndarray,
        energy_bin_edges: np.ndarray,
        default_bounds: BimodalBounds | None = None,
        bounds: BoundsSequence | None = None,
    ):
        self.psd_bin_midpoints = psd_bin_midpoints
        self.energy_bin_edges = energy_bin_edges
        self.default_bounds = default_bounds
        self.bounds = bounds

    @abstractmethod
    def __call__(self, numbered_slice: NumberedSlice) -> SliceFitResult: ...
