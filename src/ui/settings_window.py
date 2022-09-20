from typing import Any, Dict, List, Tuple

# import PySimpleGUI as sg
from configuration.configuration import (load_config, override_config,
                                         save_config, validate_config)
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                               QLineEdit, QVBoxLayout, QPushButton, QWidget, QCheckBox, QSpinBox)
from PySide6.QtCore import Slot

# HIDDEN_SAVE_KEY = '-SAVE-'
# HIDDEN_LOAD_KEY = '-LOAD-'
# SAVE_KEY = 'Save'
# LOAD_KEY = 'Load'


# def _make_control(
#     control_name: str,
#     current_value: Any,
#     control_key: str,
#     min_int: int = 0,
#     max_int: int = 20
# ) -> sg.Element:
#     """Creates a settings control of the desired type

#     Parameters
#     ----------
#     control_name : str
#         Control type to make. 
#         Currently supports:
#         - 'checkbox' -> Checkbox control
#         - 'spin' -> Spin control
#         - 'input' -> Input control
#     current_value : Any
#         Current value of the control
#     control_key : str
#         Key for the control
#     min_int : int, optional
#         lowest selectable value (for Spin controls), by default 0
#     max_int : int, optional
#         highest selectable value (for Spin controls), by default 20

#     Returns
#     -------
#     sg.Element
#         Settings control

#     Raises
#     ------
#     ValueError
#         when trying to make unsupported control type
#     """
#     control_width = 10

#     if control_name == 'checkbox':
#         return sg.Checkbox('', default=current_value, key=control_key)
#     if control_name == 'input':
#         return sg.Input(
#             default_text=str(current_value),
#             size=control_width,
#             justification='left',
#             key=control_key
#         )
#     if control_name == 'spin':
#         return sg.Spin(
#             [x for x in range(min_int, max_int + 1)],
#             initial_value=current_value,
#             size=control_width,
#             key=control_key
#         )
#     raise ValueError(f"Unsupported control name {control_name}")


# def _make_controls_row(
#     controls_data: Dict[str, Any],
#     text_width: int
# ) -> List[sg.Element]:
#     """Generate a row in the settings control form
    
#     The row consists of:
#     - text label, right justified
#     - settings control of type specified in controls_data (Checkbox, Spin, Input)

#     Parameters
#     ----------
#     controls_data : Dict[str, Any]
#         Data needed for one control form row
#     text_width : int
#         Width of the row's text label

#     Returns
#     -------
#     List[sg.Element]
#         _description_
#     """
#     control_name: str = controls_data['control']
#     title: str = controls_data['title']
#     current_value = controls_data['value']
#     control_key: str = controls_data['key']

#     text_element = sg.Text(
#         text=title, size=text_width, justification='right')
#     control_element = _make_control(
#         control_name, current_value, control_key)
#     return [text_element, control_element]


# def _layout_window(
#     config: Dict,
#     config_setup: Dict[str, Any]
# ) -> sg.Window:
#     """Generates the window layout for the main converter GUI

#     Returns
#     -------
#     sg.Window
#         converter GUI window
#     """
#     layout_data = _generate_layout_data(config, config_setup)

#     avg_width_per_char = 0.8  # works for default font
#     text_width = round(max([
#         len(v.get('title', ''))*avg_width_per_char for v in layout_data
#     ]))
#     layout = [_make_controls_row(controls_data, text_width)
#               for controls_data in layout_data]

#     files_limit_input = sg.Input(key='files_limit', visible=False)
#     hidden_row: List[sg.Element] = [
#         sg.Input(key=HIDDEN_SAVE_KEY, enable_events=True, visible=False),
#         sg.Input(key=HIDDEN_LOAD_KEY, enable_events=True, visible=False),
#         files_limit_input
#     ]
#     layout.append(hidden_row)

#     yaml_type_filter = (('YAML settings file', '*.yaml'),)

#     save_button = sg.FileSaveAs(
#         button_text=SAVE_KEY,
#         file_types=yaml_type_filter,
#         default_extension='.yaml',
#         target=HIDDEN_SAVE_KEY
#     )
#     load_button = sg.FileBrowse(
#         button_text=LOAD_KEY,
#         file_types=yaml_type_filter,
#         target=HIDDEN_LOAD_KEY
#     )
#     button_row = [save_button, load_button,
#                   sg.Push(), sg.Submit(), sg.Cancel()]
#     layout.append(button_row)

#     window = sg.Window('Settings', layout, finalize=True)
#     files_limit_input.update(value = str(config.get('files_limit')))
#     return window


# def _extract_config_from_values(values: Dict[str, Any]) -> Dict[str, Any]:
#     """Constructs settings data from current settings window control values

#     Parameters
#     ----------
#     values : Dict[str, Any]
#         Current settings window control values

#     Returns
#     -------
#     Dict[str, Any]
#         Conversion settings data
#     """
#     non_config_keys = (SAVE_KEY, LOAD_KEY, HIDDEN_SAVE_KEY, HIDDEN_LOAD_KEY)
#     config_values = {k: v for k, v in values.items()
#                      if k not in non_config_keys}
#     # files_limit must be None or int, Input values are str
#     if config_values['files_limit'] == 'None':
#         config_values['files_limit'] = None
#     else:
#         config_values['files_limit'] = int(config_values['files_limit'])
#     return config_values


# def _update_window_controls(
#     window: sg.Window,
#     control_values: Dict
# ):
#     """Updates settings window control values

#     Parameters
#     ----------
#     window : sg.Window
#         Window to update
#     control_values : Dict
#         New values to apply to controls
#     """
#     elements: List[sg.Element] = window.element_list()
#     for element in elements:
#         if element.key in control_values:
#             new_value = control_values[element.key]
#             # Spinner values are picked from list, so must exist in list
#             if (isinstance(element, sg.Spin)
#                     and new_value not in element.Values):
#                 new_min = min(min(element.Values), new_value)
#                 new_max = max(max(element.Values), new_value)
#                 element.update(values=[x for x in range(new_min, new_max+1)])
#             if element.key == 'files_limit' and new_value is None:
#                 new_value = 'None'
#             element.update(value=new_value)
#     window.refresh()


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

#     if event in (sg.WIN_CLOSED, 'Cancel'):
#         new_state = {**state, 'done': True}
#     if event == 'Submit':
#         config_values = _extract_config_from_values(values)
#         new_config = override_config(state['config'], config_values)
#         new_state = {**state, 'done': True, 'config': new_config}
#     if event == HIDDEN_SAVE_KEY:
#         save_path = values.get(HIDDEN_SAVE_KEY)
#         if save_path is not None:
#             config_values = _extract_config_from_values(values)
#             save_config(config_values, Path(save_path))
#     if event == HIDDEN_LOAD_KEY:
#         load_path = values.get(HIDDEN_LOAD_KEY)
#         if load_path is not None:
#             new_settings = load_config(Path(load_path))
#             if validate_config(new_settings, state['config_setup']):
#                 _update_window_controls(state['window'], new_settings)
#             else:
#                 sg.popup_ok("Invalid settings file")

#     return new_state


# def _event_handling_loop(
#     window: sg.Window,
#     config: Dict,
#     config_setup: Dict[str, Any]
# ) -> Dict:
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
#     _type_
#         Updated conversion settings
#     """
#     state = {
#         'done': False,
#         'config': config,
#         'config_setup': config_setup,
#         'window': window
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

#     return state['config']


# def settings_window(config: Dict, config_setup: Dict[str, Any]) -> Dict:
#     """Launch GUI window to change settings

#     Parameters
#     ----------
#     config : Dict
#         configuration data
#     config_setup : Dict[str, Any]
#         config setup data

#     Returns
#     -------
#     Optional[Dict]
#         new settings data dictionary with updated values

#     Raises
#     ------
#     ValueError
#         when config setup data has an unsupported UI control type
#     """

#     window = _layout_window(config, config_setup)

#     new_config = _event_handling_loop(window, config, config_setup)

#     window.close()
#     return new_config
        
    
class CheckboxProxy():
    
    def __init__(self, checkbox: QCheckBox):
        self._checkbox = checkbox
        
    @property
    def control_widget(self) -> QCheckBox:
        return self._checkbox
    
    @property
    def value(self) -> bool:
        return self._checkbox.isChecked()
    
    @value.setter
    def value(self, new_value: bool):
        self._checkbox.setChecked(new_value)
        
    @property
    def valid_type(self) -> type:
        return bool


class TextInputProxy():
    
    def __init__(self, text_input: QLineEdit) -> None:
        self._textinput = text_input
        
    @property
    def control_widget(self) -> QLineEdit:
        return self._textinput
    
    @property
    def value(self) -> str:
        return self._textinput.text()
    
    @value.setter
    def value(self, new_value: str):
        self._textinput.setText(new_value)
        
    @property
    def valid_type(self) -> type:
        return str
    
    
class SpinboxProxy():
    
    def __init__(self, spinbox: QSpinBox) -> None:
        self._spinbox = spinbox
        
    @property
    def control_widget(self) -> QSpinBox:
        return self._spinbox
    
    @property
    def value(self) -> int:
        return self._spinbox.value()
    
    @value.setter
    def value(self, new_value: int):
        self._spinbox.setValue(new_value)
        
    @property
    def valid_type(self) -> type:
        return int
    
    
class SettingControlProxy():
    
    def __init__(self, setting_control: QCheckBox | QLineEdit | QSpinBox):
        if isinstance(setting_control, QCheckBox):
            self._strategy = CheckboxProxy(setting_control)
        elif isinstance(setting_control, QLineEdit):
            self._strategy = TextInputProxy(setting_control)
        elif isinstance(setting_control, QSpinBox):
            self._strategy = SpinboxProxy(setting_control)
        else:
            raise ValueError('Unsupported setting control type')
                
    @property
    def control_widget(self) -> QCheckBox | QLineEdit | QSpinBox:
        return self._strategy.control_widget
    
    @property
    def value(self) -> bool | str | int:
        return self._strategy.value
    
    @value.setter
    def value(self, new_value: Any):
        self._strategy.value = new_value


class SettingsWindow(QDialog):
    
    def __init__(self, config: Dict, config_setup: Dict[str, Any]):
        super().__init__()
        self._set_window_params()
        
        # models
        self._config = config
        self._original_config = config
        
        # controls
        self._test_line_edit = QLineEdit()
        self.save_button = QPushButton('Save')
        self.load_button = QPushButton('Load')
        self.button_box = QDialogButtonBox()
        
        self._form_controls: Dict[str, Tuple[str, SettingControlProxy]] = {}
        form_controls = self._generate_layout_data(config, config_setup)
        for control_data in form_controls:
            control_name: str = control_data['control']
            title: str = control_data['title']
            current_value = control_data['value']
            control_key: str = control_data['key']
            
            if control_name == 'checkbox':
                control = SettingControlProxy(QCheckBox())
            elif control_name == 'input':
                control = SettingControlProxy(QLineEdit())
            elif control_name == 'spin':
                control = SettingControlProxy(QSpinBox())
            else:
                raise ValueError(f"Unsupported control name {control_name}")
            control.value = current_value
            self._form_controls[control_key] = (title, control)
            
        # model/view connections
        self._connect_signals()
        self._layout_window()
    
    def _set_window_params(self):
        self.setWindowTitle("Converter Settings")
    
    def _connect_signals(self):
        self.button_box.accepted.connect(self._handle_accepted) # type: ignore
        self.button_box.rejected.connect(self._handle_rejected) # type: ignore
        self.save_button.clicked.connect( # type: ignore
            self._handle_save_button_clicked)
        self.load_button.clicked.connect( # type: ignore
            self._handle_load_button_clicked)
    
    def _layout_window(self):
        form_layout = QFormLayout()
        for form_row in self._form_controls.values():
            label, control = form_row
            form_layout.addRow(label, control.control_widget)
        
        self.button_box.addButton(self.save_button, 
                                  QDialogButtonBox.ApplyRole)
        self.button_box.addButton(self.load_button, 
                                  QDialogButtonBox.ActionRole)
        self.button_box.addButton(QDialogButtonBox.Ok)
        self.button_box.addButton(QDialogButtonBox.Cancel)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)
    
    @property
    def config(self) -> Dict:
        return {**self._config}
    
    @Slot()
    def _handle_save_button_clicked(self):
        print("Save button clicked")
    
    @Slot()
    def _handle_load_button_clicked(self):
        print("Load button clicked")
    
    @Slot()
    def _handle_accepted(self):
        controls_values = self._get_control_values()
        self._config = {**self._config, **controls_values}
        self._original_config = {**self._config}  # maintain independence
        self.accept()
    
    @Slot()
    def _handle_rejected(self):
        self._config = {**self._original_config}  # maintain independence
        self._update_control_values(self._original_config)
        self.reject()
    
    def _get_control_values(self) -> Dict:
        controls_values = {}
        for key, control_row in self._form_controls.items():
            control = control_row[1]
            value = control.value
            controls_values[key] = value
        return controls_values
    
    def _update_control_values(self, config: Dict):
        for key, control_row in self._form_controls.items():
            value = config[key]
            control = control_row[1]
            control.value = value
    
    def _generate_layout_data(
        self,
        config: Dict,
        config_setup: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generates all data needed to create settings window

        Parameters
        ----------
        config : Dict
            Current settings data
        config_setup : Dict[str, Any]
            Configuration setup data

        Returns
        -------
        List[Dict[str, Any]]
            List of required data for each control row in the settings window
        """
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

        
        
