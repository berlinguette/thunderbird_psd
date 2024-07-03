import numpy as np
from data_processing.processing.figure_of_merit import FOM
from data_processing.processing.slice_fitting.bimodal_fitting import (
    get_bimodal_fit,
    get_bimodal_fit_guess,
)
from data_processing.processing.slice_fitting.helpers import split_params
from data_processing.types import (
    BimodalParams,
    FitErrorResult,
    FitResult,
    NumberedSlice,
    SliceFitResult,
)

from .slice_fitter import SliceFitter


class PeakFinderSliceFitter(SliceFitter):
    def __call__(self, numbered_slice: NumberedSlice) -> SliceFitResult:
        i, slice = numbered_slice
        slice_left_edge = self.energy_bin_edges[i]
        slice_right_edge = self.energy_bin_edges[i + 1]

        guess = get_bimodal_fit_guess(self.psd_bin_midpoints, slice)
        try:
            gamma_params, neutron_params, cov = get_bimodal_fit(
                self.psd_bin_midpoints, slice, guess=guess
            )
        except RuntimeError:
            fit_result = FitResult(
                i, None, None, slice_left_edge, slice_right_edge, None
            )
            fit_error_result = FitErrorResult(
                i, None, None, slice_left_edge, slice_right_edge
            )
            return fit_result, fit_error_result

        fom = FOM(*gamma_params[:-1], *neutron_params[:-1])
        perr = BimodalParams(*np.sqrt(np.diag(cov)))
        fit_result = FitResult(
            i, gamma_params, neutron_params, slice_left_edge, slice_right_edge, fom
        )
        fit_error_result = FitErrorResult(
            i, *split_params(perr), slice_left_edge, slice_right_edge
        )

        return fit_result, fit_error_result
