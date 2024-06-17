from data_processing.processing.neutron_window_generation import (
    generate_n_distro_neutron_window,
)
from data_processing.processing.neutron_window_strategy.abstract_strategy import (
    AbstractNeutronStrategy,
)
from data_processing.types import (
    NeutronDistributionGenerationSettings,
    WindowBorders
)


class NeutronDistributionGenerationStrategy(AbstractNeutronStrategy):
    def get_neutron_window(self) -> WindowBorders:
        window_settings = self._get_settings(NeutronDistributionGenerationSettings)
        df = self._get_slice_fit_dataframe()
        return generate_n_distro_neutron_window(
            df,
            sigma=window_settings.sigma,
            fom_energy_range=window_settings.fom_energy_range,
        )

    def _validate_settings(self) -> bool:
        return isinstance(self._settings, NeutronDistributionGenerationSettings)
