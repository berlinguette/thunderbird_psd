from data_processing.processing.neutron_window_generation import (
    generate_nasa_neutron_window,
)
from data_processing.processing.neutron_window_strategy.abstract_strategy import (
    AbstractNeutronWindowStrategy,
)
from data_processing.types import NasaGenerationSettings, WindowBorders


class NasaGenerationStrategy(AbstractNeutronWindowStrategy):
    def get_neutron_window(self) -> WindowBorders:
        window_settings = self._get_settings(NasaGenerationSettings)
        df = self._get_slice_fit_dataframe()
        return generate_nasa_neutron_window(
            df,
            window_offset=window_settings.window_offset,
            sigma=window_settings.sigma,
            lower_energy_bound=window_settings.lower_energy_bound,
            recalculate_lower_energy_bound=window_settings.recalculate_lower_energy_bound,
        )

    def _validate_settings(self) -> bool:
        return isinstance(self._settings, NasaGenerationSettings)

    # def _get_settings(self) -> NasaGenerationSettings:
    #     return self._get_settings_generic(NasaGenerationSettings)
