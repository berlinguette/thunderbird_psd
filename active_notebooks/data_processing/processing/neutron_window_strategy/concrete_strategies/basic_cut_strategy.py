from data_processing.processing.neutron_window_strategy.abstract_strategy import AbstractNeutronStrategy
from data_processing.types import BasicCutSettings, WindowBorders, VectorLikeFunction
from scipy.interpolate import interp1d
import numpy as np


class BasicCutStrategy(AbstractNeutronStrategy):
    # just cuts at given PSD value
    def get_neutron_window(self) -> WindowBorders:
        bottom_val = self._settings.bottom
        top_val = self._settings.top
        bottom_border_fn: VectorLikeFunction = interp1d(
            np.linspace(0, 1, 10),
            np.full((10,), bottom_val),
            fill_value=(bottom_val, bottom_val),
            bounds_error=False
        )
        if top_val is not None:
            top_border_fn: VectorLikeFunction = interp1d(
                np.linspace(0, 1, 10),
                np.full((10,), top_val),
                fill_value=(top_val, top_val),
                bounds_error=False
            )
        else:
            top_border_fn = None
        return WindowBorders(0, None, bottom_border_fn, top_border_fn)

    def _validate_settings(self):
        return isinstance(self._settings, BasicCutSettings)