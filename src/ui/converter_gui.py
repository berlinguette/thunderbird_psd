from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog,
                               QFileDialog, QFileSystemModel, QListView,
                               QListWidget, QMainWindow, QPushButton,
                               QTreeView, QVBoxLayout, QWidget)

from ui.settings_window import SettingsWindow

WINDOW_TITLE = 'Select Experiment Folder'


class ConverterGui(QMainWindow):

    def __init__(self, config: Dict, config_setup: Dict[str, Any]):
        super().__init__()
        self._set_window_params()

        # Models
        self._start_experiment = False
        self._config = config
        self._config_setup = config_setup

        # UI Elements
        self.settings_button = QPushButton(
            text="Settings"
        )
        self.folder_picker_button = QPushButton(
            text="Choose Experiment Folder(s)"
        )
        self.start_button = QPushButton(
            text="Start Experiment"
            # TODO change size and text color
        )
        self.folder_list = QListWidget()
        self.settings_dialog = SettingsWindow(self._config, self._config_setup)

        self._connect_signals()
        self._layout_window()

    def _connect_signals(self):
        self.settings_button.clicked.connect(  # type: ignore
            self.handle_button_clicked_settings
        )
        self.folder_picker_button.clicked.connect(  # type: ignore
            self.handle_button_clicked_folder_picker
        )
        self.start_button.clicked.connect(  # type: ignore
            self.handle_button_clicked_start
        )
        self.settings_dialog.finished.connect(  # type: ignore
            self.handle_settings_closed)

    def _set_window_params(self):
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumWidth(800)

    def _layout_window(self):
        layout = QVBoxLayout()
        layout.addWidget(self.settings_button)
        layout.addWidget(self.folder_picker_button)
        layout.addWidget(self.folder_list)
        layout.addWidget(self.start_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    @property
    def start_experiment(self):
        return self._start_experiment

    @property
    def config(self) -> Dict:
        return self._config

    @Slot()
    def handle_button_clicked_settings(self):
        self.settings_dialog.open()

    @Slot(int)
    def handle_settings_closed(self, result: int):
        self._config = self.settings_dialog.config

    @Slot()
    def handle_button_clicked_folder_picker(self):
        dialog = QFileDialog(self)
        dialog.setWindowTitle('Choose Experiment Folder(s)')
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.Directory)

        list_views = dialog.findChildren(QListView)
        list_views.extend(dialog.findChildren(QTreeView))
        for view in list_views:
            if isinstance(view.model(), QFileSystemModel):
                view.setSelectionMode(
                    QAbstractItemView.ExtendedSelection)

        if dialog.exec_() == QDialog.Accepted:
            folders = [folder for folder in dialog.selectedFiles()
                       if Path(folder).is_dir()]
            self.folder_list.clear()
            self.folder_list.addItems(folders)
        dialog.deleteLater()

    @Slot()
    def handle_button_clicked_start(self):
        self._start_experiment = True
        self.close()


def converter_gui(
    config: Dict,
    config_setup: Dict[str, Any]
) -> Tuple[Dict, Optional[List[Path]]]:
    """Opens a GUI window for user input, including settings changes and folder selection

    Returns
    -------
    Tuple[Dict, Optional[List[Path]]]
        Tuple of:
            - updated settings (or original if no updates)
            - the chosen path, or None if the converter window is closed
    """
    # window = _layout_window()
    # folder, config = _event_handling_loop(window, config, config_setup)

    # window.close()
    # if folder is not None:
    #     folder = Path(folder)
    # return config, folder

    app = QApplication([])
    converter_gui = ConverterGui(config, config_setup)
    converter_gui.show()

    app.exec_()

    if converter_gui.start_experiment:
        final_config = converter_gui.config
        folders = [
            Path(converter_gui.folder_list.item(folder).text())
            for folder in range(converter_gui.folder_list.count())
        ]
    else:
        final_config = config
        folders = None

    return final_config, folders  # stub TODO finish this


if __name__ == "__main__":
    from configuration.configuration import (get_configuration,
                                             load_config_setup)

    config_setup = load_config_setup()
    config = get_configuration({}, config_setup)
    config, folder = converter_gui(config, config_setup)
    print(config)
    print(folder)
