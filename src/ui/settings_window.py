from pathlib import Path
from typing import Any, Dict, List, Tuple

from configuration.configuration import (is_config_valid, load_config,
                                         override_config, save_config)
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox,
                               QVBoxLayout)


class CheckboxProxy():
    """Provides a simplified interface to the QCheckBox widget
    
    CheckboxProxy follows the interface of SettingControlProxy, allowing it
    to be used as a strategy in that class

    Parameters
    ----------
    checkbox : QCheckBox
        checkbox widget
    """

    def __init__(self, checkbox: QCheckBox):
        self._checkbox = checkbox

    @property
    def control_widget(self) -> QCheckBox:
        """This property gives the underlying checkbox widget used here, 
        which should not be changed.
        """
        return self._checkbox

    @property
    def value(self) -> bool:
        """This property represents the current value of the checkbox widget, 
        and can be set.
        """
        return self._checkbox.isChecked()

    @value.setter
    def value(self, new_value: bool):
        self._checkbox.setChecked(new_value)


class TextInputProxy():
    """Provides a simplified interface to the QLineEdit widget
    
    TextInputProxy follows the interface of SettingControlProxy, allowing it
    to be used as a strategy in that class

    Parameters
    ----------
    checkbox : QLineEdit
        text input widget
    """

    def __init__(self, text_input: QLineEdit) -> None:
        self._textinput = text_input

    @property
    def control_widget(self) -> QLineEdit:
        """This property gives the underlying text input widget used here, 
        which should not be changed.
        """
        return self._textinput

    @property
    def value(self) -> str:
        """This property represents the current value of the text input widget, 
        and can be set.
        """
        return self._textinput.text()

    @value.setter
    def value(self, new_value: str):
        self._textinput.setText(new_value)


class SpinboxProxy():
    """Provides a simplified interface to the QSpinBox widget
    
    SpinboxProxy follows the interface of SettingControlProxy, allowing it
    to be used as a strategy in that class

    Parameters
    ----------
    checkbox : QSpinBox
        spinbox widget
    """

    def __init__(self, spinbox: QSpinBox) -> None:
        self._spinbox = spinbox

    @property
    def control_widget(self) -> QSpinBox:
        """This property gives the underlying spinbox widget used here, 
        which should not be changed.
        """
        return self._spinbox

    @property
    def value(self) -> int:
        """This property represents the current value of the spinbox widget, 
        and can be set.
        """
        return self._spinbox.value()

    @value.setter
    def value(self, new_value: int):
        self._spinbox.setValue(new_value)


class SettingControlProxy():

    def __init__(self, setting_control: QCheckBox | QLineEdit | QSpinBox):
        """Provides a simpler interface to a setting control widget, allowing 
        values to be read and changed in the same way regardless of data type

        Parameters
        ----------
        setting_control : QCheckBox | QLineEdit | QSpinBox
            setting control widget to be proxied

        Raises
        ------
        ValueError
            when an unsupported widget type is provided
        """
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
        """This property gives the underlying setting control widget used here, 
        which should not be changed.
        """
        return self._strategy.control_widget

    @property
    def value(self) -> bool | str | int:
        """This property represents the current value of the widget, 
        and can be set.
        """
        return self._strategy.value

    @value.setter
    def value(self, new_value: Any):
        self._strategy.value = new_value


class SettingsWindow(QDialog):
    """The GUI dialog window for changing conversion settings

    Parameters
    ----------
    config : Dict
        current converter settings data
    config_setup : Dict[str, Any]
        config setup data
    """

    def __init__(self, config: Dict, config_setup: Dict[str, Any]):
        super().__init__()
        self._set_window_params()

        # models
        self._config = config
        self._original_config = config
        self._config_setup = config_setup
        self._settings_active_folder = None

        # controls
        self._test_line_edit = QLineEdit()
        self.save_button = QPushButton('Save')
        self.load_button = QPushButton('Load')
        self.button_box = QDialogButtonBox()
        self._save_dialog = QFileDialog(self)
        self._load_dialog = QFileDialog(self)

        self._form_controls: Dict[str, Tuple[str, SettingControlProxy]] = {}
        self._set_up_form_controls()
        self._set_up_dialogs()

        self._connect_signals()
        self._layout_window()

    def _set_window_params(self):
        """Sets up all dialog parameters"""
        self.setWindowTitle("Converter Settings")

    def _set_up_form_controls(self):
        """Sets up all form controls, allowing layout and data access

        Raises
        ------
        ValueError
            when control type given in config setup data is not supported
        """
        form_controls = self._generate_layout_data(
            self._config, self._config_setup)
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

    def _set_up_dialogs(self):
        """Sets up save and load dialogs
        """
        settings_file_filter = "Configuration Files (*.yaml)"
        self._save_dialog.setWindowTitle('Save Settings')
        self._save_dialog.setNameFilter(settings_file_filter)
        self._save_dialog.setFileMode(QFileDialog.AnyFile)
        self._save_dialog.setAcceptMode(QFileDialog.AcceptSave)
        self._load_dialog.setWindowTitle('Load Settings')
        self._load_dialog.setNameFilter(settings_file_filter)

    def _connect_signals(self):
        """Connects all UI element signals to appropriate slots
        """
        self.button_box.accepted.connect(self._handle_accepted)  # type: ignore
        self.button_box.rejected.connect(self._handle_rejected)  # type: ignore
        self.save_button.clicked.connect(  # type: ignore
            self._handle_save_button_clicked)
        self.load_button.clicked.connect(  # type: ignore
            self._handle_load_button_clicked)
        self._save_dialog.finished.connect(  # type: ignore
            self._handle_save_dialog_finished)
        self._load_dialog.finished.connect(  # type: ignore
            self._handle_load_dialog_finished)
        self._save_dialog.fileSelected.connect(  # type: ignore
            self._handle_save_file_picked)
        self._load_dialog.fileSelected.connect(  # type: ignore
            self._handle_load_file_picked)

    def _layout_window(self):
        """Generates window layout
        """
        form_layout = QFormLayout()
        for form_row in self._form_controls.values():
            label, control = form_row
            form_layout.addRow(label, control.control_widget)

        self.button_box.addButton(QDialogButtonBox.Ok)
        self.button_box.addButton(QDialogButtonBox.Cancel)
        
        button_row = QHBoxLayout()
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.load_button)
        button_row.addStretch()
        button_row.addWidget(self.button_box)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addLayout(button_row)
        self.setLayout(main_layout)

    @property
    def config(self) -> Dict:
        """This property gives the current conversion settings, and should not 
        be changed.
        """
        return {**self._config}

    @Slot()
    def _handle_save_button_clicked(self):
        """Slot handling click events on the "Save" button
        """
        if self._settings_active_folder is not None:
            self._save_dialog.setDirectory(self._settings_active_folder)
        self._save_dialog.open()

    @Slot()
    def _handle_load_button_clicked(self):
        """Slot handling click events on the "Load" button"""
        if self._settings_active_folder is not None:
            self._load_dialog.setDirectory(self._settings_active_folder)
        self._load_dialog.open()

    @Slot()
    def _handle_save_dialog_finished(self):
        """Slot handling dialog close events on the 'Save' dialog
        """
        self._settings_active_folder = self._save_dialog.directory()

    @Slot()
    def _handle_load_dialog_finished(self):
        """Slot handling dialog close events on the 'Load' dialog
        """
        self._settings_active_folder = self._load_dialog.directory()

    @Slot(str)
    def _handle_save_file_picked(self, filename: str):
        """Slot handling file selected events on the "Save" dialog

        Parameters
        ----------
        filename : str
            Path to the selected file (to save)
        """
        filepath = Path(filename)
        if not filepath.is_dir():
            control_values = override_config(
                self._config, self._get_control_values())
            if is_config_valid(control_values, self._config_setup):
                save_config(control_values, filepath)
            else:
                QMessageBox.warning(
                    self,
                    "Invalid settings",
                    "Some settings were not valid and could not be saved.")
        else:
            QMessageBox.warning(
                self,
                "Invalid file",
                "The chosen save file was not a valid file."
            )

    @Slot(str)
    def _handle_load_file_picked(self, filename: str):
        """Slot handling file selected events on the "Load" dialog

        Parameters
        ----------
        filename : str
            Path to the selected file (to load)
        """
        filepath = Path(filename)
        if filepath.is_file():
            new_config = override_config(
                self._config, load_config(filepath))
            if is_config_valid(new_config, self._config_setup):
                self._config = new_config
                self._update_control_values(self._config)
            else:
                QMessageBox.warning(
                    self,
                    "Invalid settings file",
                    "The chosen file is not a valid settings file.")
        else:
            QMessageBox.warning(
                self,
                "Not a file",
                "You did not choose a file.")

    @Slot()
    def _handle_accepted(self):
        """Slot handling 'accepted' events from this dialog
        """
        self._config = override_config(
            self._config, self._get_control_values())
        self._original_config = {**self._config}  # maintain independence
        self.accept()

    @Slot()
    def _handle_rejected(self):
        """Slot handling 'rejected' events from this dialog
        """
        self._config = {**self._original_config}  # maintain independence
        self._update_control_values(self._original_config)
        self.reject()

    def _get_control_values(self) -> Dict:
        """Gets values from all setting controls.
        These values are keyed based on their original config keys, so this 
        data can be used to generate an updated configuration

        Returns
        -------
        Dict
            values from all settings controls (keyed as specified above)
        """
        controls_values = {}
        for key, control_row in self._form_controls.items():
            control = control_row[1]
            value = control.value
            controls_values[key] = value
        return controls_values

    def _update_control_values(self, config: Dict):
        """Updates all visible control values to match the given config data

        Parameters
        ----------
        config : Dict
            Configuration data to be used for control updates
        """
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
