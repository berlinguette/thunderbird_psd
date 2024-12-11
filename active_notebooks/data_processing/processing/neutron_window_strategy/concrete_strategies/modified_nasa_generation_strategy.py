from data_processing.processing.neutron_window_generation import generate_modified_nasa_neutron_window
from data_processing.processing.neutron_window_strategy.abstract_strategy import AbstractNeutronStrategy
from data_processing.types import ModifiedNasaGenerationSettings, WindowBorders


class ModifiedNasaGenerationStrategy(AbstractNeutronStrategy):
    def get_neutron_window(self) -> WindowBorders:
        window_settings = self._get_settings(ModifiedNasaGenerationSettings)
        df = self._get_slice_fit_dataframe()
        return generate_modified_nasa_neutron_window(
            df,
            gamma_sigma=window_settings.gamma_sigma,
            neutron_sigma=window_settings.neutron_sigma,
            lower_energy_bound=window_settings.lower_energy_bound
        )
    
    def _validate_settings(self) -> bool:
        return isinstance(self._settings, ModifiedNasaGenerationSettings)
