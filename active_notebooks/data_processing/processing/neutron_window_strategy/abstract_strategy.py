from abc import ABC, abstractmethod

from data_processing.types import (
    NeutronWindowSettings,
    SpecificNeutronWindowSettings,
    WindowBorders,
)
from pandas import DataFrame


class AbstractNeutronStrategy(ABC):
    def __init__(self, settings: NeutronWindowSettings):
        self._slice_fit_df: DataFrame | None = None
        self._settings = settings
        valid_settings = self._validate_settings()
        if not valid_settings:
            raise ValueError("Settings are not valid")

    @abstractmethod
    def get_neutron_window(self) -> WindowBorders: ...

    @abstractmethod
    def _validate_settings(self) -> bool: ...

    def _get_settings(
        self, settings_type: type[SpecificNeutronWindowSettings]
    ) -> SpecificNeutronWindowSettings:
        if isinstance(self._settings, settings_type):
            return self._settings
        else:
            raise ValueError("Settings are not valid")

    def set_slice_fit_dataframe(self, df: DataFrame):
        self._slice_fit_df = df

    def _get_slice_fit_dataframe(self) -> DataFrame:
        if self._slice_fit_df is not None:
            return self._slice_fit_df
        else:
            raise ValueError("Slice fit dataframe was not set")
