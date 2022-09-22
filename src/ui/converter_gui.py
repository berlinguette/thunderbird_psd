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
    """The main GUI for the converter, 
    used to change settings and choose experiment folders

    Parameters
    ----------
    config : Dict
        Converter configuration data
    config_setup : Dict[str, Any]
        Config setup data
    """

    def __init__(self, config: Dict, config_setup: Dict[str, Any]):
        super().__init__()
        self._set_window_params()

        # Models
        self._start_conversion = False
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
            text="Start Conversion"
            # TODO change size and text color
        )
        self.start_button.setStyleSheet(
            "QPushButton {"
            "color: green;"
            "background-color: white;"
            "font: bold 14px;"
            "border-style: outset;"
            "border-width: 1px;"
            "border-radius: 5px;"
            "border-color: grey;"
            "min-width: 10em;"
            "padding: 6px;"
            "}"
            "QPushButton:pressed {"
            "border-style: inset"
            "}"
        )
        self.start_button.setMinimumHeight(50)
        self.folder_list = QListWidget()
        self.settings_dialog = SettingsWindow(self._config, self._config_setup)

        self._connect_signals()
        self._layout_window()

    def _connect_signals(self):
        """Connects all UI element signals to appropriate slots
        """
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
        """Sets up all UI window parameters
        """
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumWidth(800)

    def _layout_window(self):
        """Generates window layout
        """
        layout = QVBoxLayout()
        layout.addWidget(self.settings_button)
        layout.addWidget(self.folder_picker_button)
        layout.addWidget(self.folder_list)
        layout.addWidget(self.start_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    @property
    def start_conversion(self) -> bool:
        """This property indicates whether the "Start" button has been pressed, 
        and should not be changed
        """
        return self._start_conversion

    @property
    def config(self) -> Dict:
        """This property gives the current conversion settings, and should not 
        be changed.
        """
        return self._config

    @Slot()
    def handle_button_clicked_settings(self):
        """Slot handling click events on the "Settings" button
        """
        self.settings_dialog.open()

    @Slot(int)
    def handle_settings_closed(self, result: int):
        """Slot handling dialog close events on the "Settings" dialog

        Parameters
        ----------
        result : int
            Dialog status code, indicating how it was closed (i.e. cancel/OK)
        """
        if result == QDialog.Accepted:
            self._config = self.settings_dialog.config

    @Slot()
    def handle_button_clicked_folder_picker(self):
        """Slot handling click events on the experiment folder picker button
        """
        dialog = QFileDialog(self)
        dialog.setWindowTitle('Choose Experiment Folder(s)')
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.Directory)

        # hack to allow multi-folder selection from https://stackoverflow.com/q/28544425
        # native Windows folder picker doesn't support multiple folders
        # must use Qt version, but can't just set ExtendedSelection because
        # it has multiple views, and each must be set separately
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
        """Slot handling click events on the "Start Conversion" button
        """
        self._start_conversion = True
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
    app = QApplication([])
    converter_gui = ConverterGui(config, config_setup)
    converter_gui.show()

    app.exec_()

    if converter_gui.start_conversion:
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
