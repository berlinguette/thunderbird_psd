from data_processing.processing.neutron_window_generation import \
    generate_rectangle_neutron_window
from data_processing.processing.neutron_window_strategy.abstract_strategy import \
    AbstractNeutronWindowStrategy
from data_processing.types import (
    SquarishGenerationSettings,
    WindowBorders
)


class SquarishGenerationStrategy(AbstractNeutronWindowStrategy):
    def get_neutron_window(self) -> WindowBorders:
        window_settings = self._get_settings(SquarishGenerationSettings)
        left = window_settings.left
        bottom = window_settings.bottom
        width = window_settings.width
        height = width / window_settings.aspect_ratio
        return generate_rectangle_neutron_window(
            left=left,
            bottom=bottom,
            width=width,
            height=height
        )
    
    def _validate_settings(self) -> bool:
        return isinstance(self._settings, SquarishGenerationSettings)
    