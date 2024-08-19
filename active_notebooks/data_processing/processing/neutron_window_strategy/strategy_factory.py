"""This module is responsible for creating neutron window strategies."""
from data_processing.processing.neutron_window_strategy.abstract_strategy import (
    AbstractNeutronStrategy,
)
from data_processing.processing.neutron_window_strategy.concrete_strategies import (
    LoadingStrategy,
    NasaGenerationStrategy,
    NeutronDistributionGenerationStrategy,
    SquarishGenerationStrategy
)
from data_processing.types import NeutronWindowSettings, WindowType


class NeutronStrategyFactory:
    """Factory to create neutron window strategies.
    """
    def make_neutron_window_strategy(
        self,
        window_type: WindowType,
        loading: bool,
        settings: NeutronWindowSettings,
    ) -> AbstractNeutronStrategy:
        """Create desired neutron window strategy.

        :param window_type: Type of neutron window strategy to make
        :type window_type: WindowType
        :param loading: Whether the strategy should be loaded from save files
        :type loading: bool
        :param settings: Neutron window generation settings
        :type settings: NeutronWindowSettings
        :return: Desired neutron window strategy
        :rtype: AbstractNeutronStrategy
        """
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
        else:
            raise ValueError(f"Unsupported window type: {window_type}")
