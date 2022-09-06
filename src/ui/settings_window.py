from pathlib import Path
from typing import Any, Dict, List, Union

import PySimpleGUI as sg
from configuration.configuration import (load_config, override_config,
                                         save_config)

HIDDEN_SAVE_KEY = '-SAVE-'
HIDDEN_LOAD_KEY = '-LOAD-'
SAVE_KEY = 'Save'
LOAD_KEY = 'Load'


def _make_control(
    control_name: str,
    current_value: Any,
    control_key: str,
    min_int: int = 0,
    max_int: int = 20
) -> sg.Element:
    control_width = 10

    if control_name == 'checkbox':
        return sg.Checkbox('', default=current_value, key=control_key)
    if control_name == 'input':
        return sg.Input(
            default_text=str(current_value),
            size=control_width,
            justification='left',
            key=control_key
        )
    if control_name == 'spin':
        return sg.Spin(
            [x for x in range(min_int, max_int + 1)],
            initial_value=current_value,
            size=control_width,
            key=control_key
        )
    raise ValueError(f"Unsupported control name {control_name}")


def _make_controls_row(
    controls_data: Dict[str, Any],
    text_width: int
) -> List[sg.Element]:
    control_name: str = controls_data['control']
    title: str = controls_data['title']
    current_value = controls_data['value']
    control_key: str = controls_data['key']

    text_element = sg.Text(
        text=title, size=text_width, justification='right')
    control_element = _make_control(
        control_name, current_value, control_key)
    return [text_element, control_element]


def _generate_layout_data(
    config: Dict,
    config_setup: Dict[str, Any]
) -> List[Dict[str, Any]]:
    gui_setup: Dict[str, Dict[str, Any]] = {
        k: v['config']['gui'] for k, v in config_setup.items()
        if 'gui' in v.get('config', {})
    }
    for key, value in gui_setup.items():
        value['value'] = config[key]
    layout_data: List[Dict[str, Any]] = sorted(
        [{'key': k, **v} for k, v in gui_setup.items()],
        key=lambda x: x['order'])
    return layout_data


def _layout_window(
    config: Dict,
    config_setup: Dict[str, Any]
) -> sg.Window:
    layout_data = _generate_layout_data(config, config_setup)

    avg_width_per_char = 0.8  # works for default font
    text_width = round(max([
        len(v.get('title', ''))*avg_width_per_char for v in layout_data
    ]))
    layout = [_make_controls_row(controls_data, text_width)
              for controls_data in layout_data]

    hidden_row: List[sg.Element] = [
        sg.Input(key=HIDDEN_SAVE_KEY, enable_events=True, visible=False),
        sg.Input(key=HIDDEN_LOAD_KEY, enable_events=True, visible=False)
    ]
    layout.append(hidden_row)

    yaml_type_filter = (('YAML settings file', '*.yaml'),)

    save_button = sg.FileSaveAs(
        button_text=SAVE_KEY,
        file_types=yaml_type_filter,
        default_extension='.yaml',
        target=HIDDEN_SAVE_KEY
    )
    load_button = sg.FileBrowse(
        button_text=LOAD_KEY,
        file_types=yaml_type_filter,
        target=HIDDEN_LOAD_KEY
    )
    button_row = [save_button, load_button,
                  sg.Push(), sg.Submit(), sg.Cancel()]
    layout.append(button_row)

    window = sg.Window('Settings', layout)
    return window


def _extract_config_from_values(values: Dict[str, Any]) -> Dict[str, Any]:
    non_config_keys = (SAVE_KEY, LOAD_KEY, HIDDEN_SAVE_KEY, HIDDEN_LOAD_KEY)
    config_values = {k: v for k, v in values.items()
                     if k not in non_config_keys}
    return config_values


def _update_window_controls(
    window: sg.Window,
    control_values: Dict
):
    elements: List[sg.Element] = window.element_list()
    for element in elements:
        if element.key in control_values:
            new_value = control_values[element.key]
            # Spinner values are picked from list, so must exist in list
            if (isinstance(element, sg.Spin) 
                and new_value not in element.Values):
                new_min = min(min(element.Values), new_value)
                new_max = max(max(element.Values), new_value)
                element.update(values = [x for x in range(new_min, new_max+1)])
            element.update(value = new_value)
    window.refresh()


def _handle_event(
    event: str,
    values: Dict[str, Any],
    state: Dict[str, Any]
) -> Dict[str, Any]:
    new_state = state

    if event in (sg.WIN_CLOSED, 'Cancel'):
        new_state = {**state, 'done': True}
    if event == 'Submit':
        config_values = _extract_config_from_values(values)
        new_config = override_config(state['config'], config_values)
        new_state = {**state, 'config': new_config}
    if event == HIDDEN_SAVE_KEY:
        save_path = values.get(HIDDEN_SAVE_KEY)
        if save_path is not None:
            config_values = _extract_config_from_values(values)
            save_config(config_values, Path(save_path))
    if event == HIDDEN_LOAD_KEY:
        load_path = values.get(HIDDEN_LOAD_KEY)
        if load_path is not None:
            new_settings = load_config(Path(load_path))
            _update_window_controls(state['window'], new_settings)
            new_config = override_config(state['config'], new_settings)
            new_state = {**state, 'config': new_config}

    return new_state


def _event_handling_loop(
    window: sg.Window,
    config: Dict,
):
    state = {
        'done': False,
        'config': config,
        'window': window
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

    return state['config']


def settings_window(config: Dict, config_setup: Dict[str, Any]) -> Dict:
    """Launch GUI window to change settings

    Parameters
    ----------
    config : Dict
        configuration data
    config_setup : Dict[str, Any]
        config setup data

    Returns
    -------
    Optional[Dict]
        new settings data dictionary with updated values

    Raises
    ------
    ValueError
        when config setup data has an unsupported UI control type
    """

    window = _layout_window(config, config_setup)

    new_config = _event_handling_loop(window, config)

    # new_config = {k: v for k, v in config.items()}
    # done = False
    # while not done:
    #     read_result = window.read()
    #     if not isinstance(read_result, tuple):
    #         continue
    #     event: str
    #     values: Dict[str, Any]
    #     event, values = read_result
    #     non_config_keys = (save_key, load_key,
    #                        hidden_save_key, hidden_load_key)
    #     config_values = {k: v for k, v in values.items()
    #                      if k not in non_config_keys}

    #     if event in (sg.WIN_CLOSED, 'Cancel', 'Submit'):
    #         done = True
    #         if event == 'Submit':
    #             new_config = override_config(config, config_values)
    #     elif event == hidden_save_key:
    #         save_path = values.get(hidden_save_key)
    #         message_debug("Save detected\n" +
    #                       f"File: {save_path}\n" +
    #                       f"Config: {config_values}",
    #                       logger, in_log=False)
    #         if save_path is not None:
    #             save_config(config_values, Path(save_path))
    #             message_debug(f"Config saved at {save_path}",
    #                           logger, in_log=False)
    #         else:
    #             message_debug("Save path could not be found",
    #                           logger, in_log=False)
    #     elif event == hidden_load_key:
    #         load_path = values.get(hidden_load_key)
    #         message_debug("Load detected\n" +
    #                       f"File: {load_path}\n" +
    #                       f"Config: {config_values}",
    #                       logger, in_log=False)
    #         if load_path is not None:
    #             new_config =
    #             # use config values to update controls
    #             pass
    #         else:
    #             message_debug("Save path could not be found",
    #                           logger, in_log=False)

    window.close()
    return new_config
