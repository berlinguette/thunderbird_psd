import PySimpleGUI as sg
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from ui.settings_window import settings_window

WINDOW_TITLE = 'Select Raw Data Folder'
FOLDER_KEY = '-FOLDER-'
SETTINGS_KEY = 'Settings'


def _layout_window() -> sg.Window:
    """Generates the window layout for the main converter GUI

    Returns
    -------
    sg.Window
        converter GUI window
    """
    left_col = [
        [
            sg.Text('Folder'),
            sg.In(size=(25, 1), enable_events=True, key=FOLDER_KEY),
            sg.FolderBrowse()
        ],
        [
            sg.Button(SETTINGS_KEY)
        ]
    ]
    layout = [[sg.Column(left_col, element_justification='c')]]
    window = sg.Window(WINDOW_TITLE, layout, resizable=True)
    return window


def _handle_event(
    event: str,
    values: Dict[str, Any],
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Handles button events on GUI window, immutably updating state

    Parameters
    ----------
    event : str
        Event key
    values : Dict[str, Any]
        All window control values
    state : Dict[str, Any]
        Current state of the GUI window data. Includes any data needed to 
        handle any event

    Returns
    -------
    Dict[str, Any]
        Updated window state. Since state is updated immutably, this is a new
        dictionary object.
    """
    new_state = state
    if event in (sg.WIN_CLOSED, 'Exit'):
        new_state = {**state, 'done': True}

    if event == FOLDER_KEY:
        new_state = {**state, 'done': True, 'folder': values[FOLDER_KEY]}
    
    if event == SETTINGS_KEY:
        window: sg.Window = state['window']
        window.hide()
        new_config = settings_window(state['config'], state['config_setup'])
        new_state = {**state, 'config': new_config}
        window.un_hide()
    return new_state


def _event_handling_loop(
    window: sg.Window,
    config: Dict,
    config_setup: Dict[str, Any]
) -> Tuple[Optional[str], Dict]:
    """Repeatedly checks for window events and handles them.
    Closes when a terminating event is handled

    Parameters
    ----------
    window : sg.Window
        Window to be checked
    config : Dict
        Current conversion settings
    config_setup : Dict[str, Any]
        Configuration setup data

    Returns
    -------
    Tuple[Optional[str], Dict]
        Chosen folder, conversion settings with any updates applied
    """
    state = {
        'done': False,
        'folder': None,
        'window': window,
        'config': config,
        'config_setup': config_setup
    }
    while not state['done']:
        read_result = window.read()
        if not isinstance(read_result, tuple):
            continue
        event: str
        values: Dict[str, Any]
        event, values = read_result
        state = _handle_event(
            event, values, state)

    return state['folder'], state['config']


def converter_gui(
    config: Dict,
    config_setup: Dict[str, Any]
) -> Tuple[Dict, Optional[Path]]:
    """Opens a GUI window for user input, including settings changes and folder selection

    Returns
    -------
    Tuple[Dict, Optional[Path]]
        Tuple of:
            - updated settings (or original if no updates)
            - the chosen path, or None if the converter window is closed
    """
    window = _layout_window()
    folder, config = _event_handling_loop(window, config, config_setup)

    window.close()
    if folder is not None:
        folder = Path(folder)
    return config, folder
