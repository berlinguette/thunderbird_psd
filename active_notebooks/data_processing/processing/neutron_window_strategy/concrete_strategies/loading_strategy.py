"""This module is responsible for the Loading neutron window strategy."""
from data_processing.loading.window_loading import (
    get_neutron_window_paths,
    load_neutron_window,
)
from data_processing.processing.neutron_window_strategy.abstract_strategy import (
    AbstractNeutronStrategy,
)
from data_processing.types import WindowBorders


class LoadingStrategy(AbstractNeutronStrategy):
    """Strategy to generate a neutron window by loading from window files.
    """
    def get_neutron_window(self) -> WindowBorders:
        """Generate the neutron window for this strategy

        :return: Borders of the generated neutron window
        :rtype: WindowBorders
        """
        file_name_prefix = self._get_settings(str)
        return load_neutron_window(file_name_prefix)

    def _validate_settings(self) -> bool:
        if isinstance(self._settings, str):
            paths = get_neutron_window_paths(self._settings)
            paths_exist = {path.exists() for path in paths}
            return all(paths_exist)
        else:
            return False
