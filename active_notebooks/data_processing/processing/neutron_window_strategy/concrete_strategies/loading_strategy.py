from data_processing.loading.window_loading import (
    get_neutron_window_paths,
    load_neutron_window,
)
from data_processing.processing.neutron_window_strategy.abstract_strategy import (
    AbstractNeutronWindowStrategy,
)
from data_processing.types import WindowBorders


class LoadingStrategy(AbstractNeutronWindowStrategy):
    def get_neutron_window(self) -> WindowBorders:
        file_name_prefix = self._get_settings(str)
        return load_neutron_window(file_name_prefix)

    def _validate_settings(self) -> bool:
        if isinstance(self._settings, str):
            paths = get_neutron_window_paths(self._settings)
            paths_exist = {path.exists() for path in paths}
            return all(paths_exist)
        else:
            return False

    # def _get_settings(self) -> str:
    #     return self._get_settings_generic(str)
