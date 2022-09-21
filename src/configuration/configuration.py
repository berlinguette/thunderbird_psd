from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import yaml

ValueType = Union[None, bool, int, float, str]

if TYPE_CHECKING:
    from argparse import ArgumentParser


def override_config(old_config: Dict, new_config: Dict) -> Dict:
    """Combines two configuration data dictionaries, such that if any keys 
       match between old_config and new_config, the values from new_config are 
       kept

    Parameters
    ----------
    old_config : Dict
        Original configuration data
    new_config : Dict
        New configuration data, which will override any values from matching
        keys

    Returns
    -------
    Dict
        Resulting configuration data
    """
    return {**old_config, **new_config}


def _load_yaml_dict_file(file_path: Path) -> Dict[str, Any]:
    """Loads the given YAML file

    Parameters
    ----------
    file_path : Path
        path to YAML file to be loaded

    Returns
    -------
    Dict
        file data
    """
    with open(file_path, 'r') as setup_file:
        file_data = yaml.safe_load(setup_file)

    if isinstance(file_data, dict) and all(isinstance(key, str) for key in file_data.keys()):
        return file_data
    else:
        raise ValueError(
            f"YAML file at {file_path} did not have dictionary data")


def load_config_setup() -> Dict[str, Any]:
    """Loads the configuration setup file

    Returns
    -------
    Dict
        configuration setup data
    """
    # assumes setup file is in same dir as this .py file
    setup_file_path = Path(__file__).parent / 'config_fields_setup.yaml'
    return _load_yaml_dict_file(setup_file_path)


def _generate_default_config(config_setup: Dict[str, Any]) -> Dict:
    """Generates default configuration data from the configuration setup

    Args:
        config_setup (Dict): _description_

    Returns:
        Dict: _description_
    """
    config = {k: v['config']['default'] for k, v in config_setup.items()
              if 'config' in v.keys() and isinstance(v['config'], dict)}
    return config


def is_config_valid(config: Dict, config_setup: Dict[str, Any]) -> bool:
    """Validates that config data is valid

    Config data is valid if:
    - config key exists in setup
    - config value has correct type
    - config value is >= 0 if it's an integer

    Args:
        config (Dict): user-generated config file data to be validated
        config_setup (Dict): configuration setup data

    Returns:
        bool: True if config file data has valid types and values, False if not
    """

    def test_int_valid(value: ValueType) -> bool:
        if isinstance(value, int):
            return value >= 0
        else:
            return False

    def is_valid_type(value: ValueType, type_string: str) -> bool:
        tests: Dict[str, Callable[[ValueType], bool]] = {
            'none': lambda x: x is None,
            'bool': lambda x: isinstance(x, bool),
            'int': test_int_valid,
            'float': lambda x: isinstance(x, float),
            'str': lambda x: isinstance(x, str)
        }
        type_test = tests[type_string]
        return type_test(value)

    def check_config_item(key: Any, value: Any) -> bool:
        setup_data = config_setup.get(key)
        if setup_data is None:
            return False

        try:
            allowed_types: List[str] = setup_data['config']['allowed_types']
        except KeyError:
            # given config may have non-config keys (and is thus invalid)
            # those can be found in setup data, but don't have 'config' entries
            return False
        return any([
            is_valid_type(value, type_string) for type_string in allowed_types
        ])

    return all([check_config_item(k, v) for k, v in config.items()])


def populate_args_parser(
    parser: ArgumentParser,
    config_setup: Dict[str, Any]
) -> ArgumentParser:
    """Add arguments to ArgumentParser based on configuration setup data

    Configuration setup YAML file should have a 'cli' section for each setting,
    with all needed data to add an argument. At minimum, this includes a list 
    of flags under the 'flags' key, but can also include any other keyword 
    argument from add_arguments().  (See the argparse documentation for more 
    details.)

    Parameters
    ----------
    parser : ArgumentParser
        the parser to be populated with arguments
    config_setup : Dict[str, Any]
        configuration setup data

    Returns
    -------
    ArgumentParser
        populated parser

    Raises
    ------
    ValueError
        when configuration setup data does not have the correct format (i.e. 
        missing 'cli' and 'flags' keys)
    """
    type_str_to_type: Dict[str, type] = {
        'int': int,
        'float': float,
    }

    setting_data: Dict[str, Any]
    for setting, setting_data in config_setup.items():
        try:
            cli_data: Dict[str, Any] = setting_data['cli']
            flags: Union[List[str], str] = cli_data['flags']
            # YAML parse: single item list != list. Thanks YAML parse...
            if not isinstance(flags, list):
                flags = [flags]
            kwargs = {k: v for k, v in cli_data.items() if k != 'flags'}
            real_type: Optional[type] = type_str_to_type.get(
                kwargs.get('type', 'unknown'), None)
            if real_type is not None:
                kwargs['type'] = real_type
            parser.add_argument(*flags, **kwargs)
        except KeyError as e:
            raise ValueError(
                "Config setup does not have correct format ",
                f"for setting {setting}\n",
                f"Config setup: {config_setup}\n",
                f"Original exception:\n{e}"
            )

    return parser


def save_config(config: Dict, save_path: Path):
    """Saves configuration data as YAML file

    Parameters
    ----------
    config : Dict
        configuration data to be saved
    save_path : Path
        path to destination file
    """
    with open(save_path, 'w') as config_file:
        yaml.safe_dump(config, config_file)


def load_config(load_path: Path) -> Dict:
    """_summary_

    Parameters
    ----------
    load_path : Path
        _description_
    """
    return _load_yaml_dict_file(load_path)
        

def get_configuration(
    cli_args: Dict,
    config_setup: Dict,
    user_config_path: Optional[Path] = None
) -> Dict:
    """Gets the complete configuration data from any config files and command
       line arguments

    Matching items between configuration sources are overridden as follows:
    - User-defined configuration file overrides default configuration file
    - Command line arguments override both configuration files

    User-defined configuration files are in YAML format as key: value pairs.
    Check example_config.yaml for an example of the config file format.
    Check config_fields_setup.yaml to see default values. 

    Parameters
    ----------
    cli_args : Dict
        command line arguments, as {setting_name: value}
    config_setup: Dict
        configuration setup data
    user_config_path : Optional[Path], optional
        path to the user-generated configuration file. If not given, only the 
        default configuration file is used

    Returns
    -------
    Dict
        Configuration data, as {setting_name: value}
    """
    default_config = _generate_default_config(config_setup)

    config = default_config
    if user_config_path is not None:
        new_config = load_config(user_config_path)
        if is_config_valid(new_config, config_setup):
            config = override_config(config, new_config)

    # omitted args get None value, which would override config
    # so we must remove them
    cli_args = {k: v for k, v in cli_args.items() if v is not None}
    config = override_config(config, cli_args)

    return config


if __name__ == "__main__":
    from argparse import ArgumentParser

    config_setup = load_config_setup()
    default_config = _generate_default_config(config_setup)
    print(f"Default config:\n{default_config}")
    print("Saving default config...")
    default_config_path = Path(__file__).parent / 'example_config.yaml'
    save_config(default_config, default_config_path)

    new_config_data = {'files_limit': 10}
    config_valid = is_config_valid(new_config_data, config_setup)
    print(f"config valid? {config_valid}")
    if config_valid:
        config = override_config(default_config, new_config_data)
        print(f"Merged config:\n{config}")

    parser = populate_args_parser(ArgumentParser(), config_setup)
    print("Parser help:")
    parser.print_help()
