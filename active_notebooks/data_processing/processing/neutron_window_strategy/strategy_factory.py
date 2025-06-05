from data_processing.processing.neutron_window_strategy.abstract_strategy import (
    AbstractNeutronStrategy,
)
from data_processing.processing.neutron_window_strategy.concrete_strategies import (
    LoadingStrategy,
    NasaGenerationStrategy,
    NeutronDistributionGenerationStrategy,
    SquarishGenerationStrategy,
    BasicCutStrategy
)
from data_processing.types import NeutronWindowSettings, WindowType


class NeutronStrategyFactory:
    def make_neutron_window_strategy(
        self,
        window_type: WindowType,
        loading: bool,
        settings: NeutronWindowSettings,
    ) -> AbstractNeutronStrategy:
        if loading:
            return self._make_loading_strategy(settings)
        else:
            return self._make_generator_strategy(window_type, settings)

    def _make_loading_strategy(
        self, settings: NeutronWindowSettings
    ) -> AbstractNeutronStrategy:
        return LoadingStrategy(settings)

    def _make_generator_strategy(
        self, window_type: WindowType, settings: NeutronWindowSettings
    ) -> AbstractNeutronStrategy:
        if window_type == "nasa":
            return NasaGenerationStrategy(settings)
        elif window_type == "n_distro":
            return NeutronDistributionGenerationStrategy(settings)
        elif window_type == "squarish":
            return SquarishGenerationStrategy(settings)
        elif window_type == "basic_cut":
            return BasicCutStrategy(settings)
        else:
            raise ValueError(f"Unsupported window type: {window_type}")
