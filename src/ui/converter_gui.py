# import PySimpleGUI as sg
import sys
from email.charset import QP
from lib2to3.pytree import convert
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QStringListModel, Slot
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog,
                               QFileDialog, QFileSystemModel, QListView,
                               QListWidget, QMainWindow, QPushButton,
                               QTreeView, QVBoxLayout, QWidget)

from ui.settings_window import settings_window

WINDOW_TITLE = 'Select Experiment Folder'
FOLDER_KEY = '-FOLDER-'
SETTINGS_KEY = 'Settings'


# def _handle_event(
#     event: str,
#     values: Dict[str, Any],
#     state: Dict[str, Any]
# ) -> Dict[str, Any]:
#     """Handles button events on GUI window, immutably updating state

#     Parameters
#     ----------
#     event : str
#         Event key
#     values : Dict[str, Any]
#         All window control values
#     state : Dict[str, Any]
#         Current state of the GUI window data. Includes any data needed to 
#         handle any event

#     Returns
#     -------
#     Dict[str, Any]
#         Updated window state. Since state is updated immutably, this is a new
#         dictionary object.
#     """
#     new_state = state
#     if event in (sg.WIN_CLOSED, 'Exit'):
#         new_state = {**state, 'done': True}

#     if event == FOLDER_KEY:
#         new_state = {**state, 'done': True, 'folder': values[FOLDER_KEY]}
    
#     if event == SETTINGS_KEY:
#         window: sg.Window = state['window']
#         window.hide()
#         new_config = settings_window(state['config'], state['config_setup'])
#         new_state = {**state, 'config': new_config}
#         window.un_hide()
#     return new_state


# def _event_handling_loop(
#     window: sg.Window,
#     config: Dict,
#     config_setup: Dict[str, Any]
# ) -> Tuple[Optional[str], Dict]:
#     """Repeatedly checks for window events and handles them.
#     Closes when a terminating event is handled

#     Parameters
#     ----------
#     window : sg.Window
#         Window to be checked
#     config : Dict
#         Current conversion settings
#     config_setup : Dict[str, Any]
#         Configuration setup data

#     Returns
#     -------
#     Tuple[Optional[str], Dict]
#         Chosen folder, conversion settings with any updates applied
#     """
#     state = {
#         'done': False,
#         'folder': None,
#         'window': window,
#         'config': config,
#         'config_setup': config_setup
#     }
#     while not state['done']:
#         read_result = window.read()
#         if not isinstance(read_result, tuple):
#             continue
#         event: str
#         values: Dict[str, Any]
#         event, values = read_result
#         state = _handle_event(
#             event, values, state)

#     return state['folder'], state['config']


class ConverterGui(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self._set_window_params()
        
        # models
        self._start_experiment = False
        
        # controls
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
        
        # layout
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
        
    def _set_window_params(self):
        self.setWindowTitle(WINDOW_TITLE)


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
    converter_gui = ConverterGui()
    converter_gui.show()
    
    app.exec_()
    
    return config, None  # stub TODO finish this

if __name__ == "__main__":
    from configuration.configuration import (get_configuration,
                                             load_config_setup)
    
    config_setup = load_config_setup()
    config = get_configuration({}, config_setup)
    config, folder = converter_gui(config, config_setup)
    print(config)
    print(folder)
