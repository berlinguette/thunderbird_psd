from pathlib import Path
from typing import Any, Dict, List, Tuple

from configuration.configuration import (is_config_valid, load_config,
                                         override_config, save_config)
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QLineEdit,
                               QMessageBox, QPushButton, QSpinBox, QVBoxLayout)


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
        self.setWindowTitle("Converter Settings")

    def _set_up_form_controls(self):
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
        settings_file_filter = "Configuration Files (*.yaml)"
        self._save_dialog.setWindowTitle('Save Settings')
        self._save_dialog.setNameFilter(settings_file_filter)
        self._save_dialog.setFileMode(QFileDialog.AnyFile)
        self._save_dialog.setAcceptMode(QFileDialog.AcceptSave)
        self._load_dialog.setWindowTitle('Load Settings')
        self._load_dialog.setNameFilter(settings_file_filter)

    def _connect_signals(self):
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
        if self._settings_active_folder is not None:
            self._save_dialog.setDirectory(self._settings_active_folder)
        self._save_dialog.open()

    @Slot()
    def _handle_load_button_clicked(self):
        if self._settings_active_folder is not None:
            self._load_dialog.setDirectory(self._settings_active_folder)
        self._load_dialog.open()

    @Slot()
    def _handle_save_dialog_finished(self):
        self._settings_active_folder = self._save_dialog.directory()

    @Slot()
    def _handle_load_dialog_finished(self):
        self._settings_active_folder = self._load_dialog.directory()

    @Slot(str)
    def _handle_save_file_picked(self, filename: str):
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
        self._config = override_config(
            self._config, self._get_control_values())
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
