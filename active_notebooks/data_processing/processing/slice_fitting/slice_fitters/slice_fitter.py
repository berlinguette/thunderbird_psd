from abc import ABC, abstractmethod

import numpy as np
from data_processing.types import BimodalBounds, NumberedSlice, SliceFitResult, BoundsSequence


class SliceFitter(ABC):
    """Base class for slice fitters.
    Slice fitters take a PSD histogram vertical slice (i.e. for a defined energy range) and fit it to a bimodal fit function.
    Since slice fitters are objects, they can be pickled, and can therefore act as closures that can be called by multiprocessing pools.
    This concept is based on work by Steven Engelhardt (https://www.stevenengelhardt.com/2013/01/16/python-multiprocessing-module-and-closures/)

    :param psd_bin_midpoints: Array of all PSD bin midpoints
    :type psd_bin_midpoints: np.ndarray
    :param energy_bin_edges: Array of all energy bin edges (n+1 edges for n energy bins)
    :type energy_bin_edges: np.ndarray
    :param default_bounds: Bounds to use for any slice without defined bounds, defaults to None
    :type default_bounds: BimodalBounds | None, optional
    :param bounds: Information on bounds to use for specified ranges of slices, defaults to None
    :type bounds: BoundsSequence | None, optional
    """
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
    def __call__(self, numbered_slice: NumberedSlice) -> SliceFitResult: 
        """Call method for this slice fitter, allowing the slice fitter to be called like any function.

        :param numbered_slice: Slice data with associated slice index
        :type numbered_slice: NumberedSlice
        :return: Result of slice fitting
        :rtype: SliceFitResult
        """
        ...
