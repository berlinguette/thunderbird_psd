from pathlib import Path
from typing import Dict, Optional

import yaml


def _override_config(old_config: Dict, new_config: Dict) -> Dict:
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


def _load_configuration(config_path: Optional[Path] = None) -> Dict:
    """Loads the default configuration file, as well as any user-generated 
       configuration files, producing one set of configuration data

    Any items in the user-generated configuration file will override any 
    matching items from the default configuration file. 

    Parameters
    ----------
    config_path : Optional[Path], optional
        path to the user-generated configuration file. If not given, only the 
        default configuration file is used

    Returns
    -------
    Dict
        Resulting configuration data
    """
    # this assumes default_config is in same dir as this py file
    default_path = Path(__file__).parent / 'default_config.yaml'
    with open(default_path, 'r') as conf_file:
        config = yaml.safe_load(conf_file)

    # given config overrides any default config values
    if config_path is not None:
        with open(config_path, 'r') as conf_file:
            new_config = yaml.safe_load(conf_file)
            config = _override_config(config, new_config)

    return config


def get_configuration(cli_args: Dict, config_path: Optional[Path] = None) -> Dict:
    """Gets the complete configuration data from any config files and command
       line arguments

    Matching items between configuration sources are overridden as follows:
    - User-defined configuration file overrides default configuration file
    - Command line arguments override both configuration files

    User-defined configuration files are in YAML format as key: value pairs.
    Check default_config.yaml, or the command line arguments with `--help`, 
    for all current configuration keys and default values. 

    Parameters
    ----------
    cli_args : Dict
        command line arguments, as {setting_name: value}
    config_path : Optional[Path], optional
        path to the user-generated configuration file. If not given, only the 
        default configuration file is used

    Returns
    -------
    Dict
        Configuration data, as {setting_name: value}
    """
    config = _load_configuration(config_path)
    # omitted args get None value, which would override config
    # so we must remove them
    cli_args = {k: v for k, v in cli_args.items() if v is not None}
    config = _override_config(config, cli_args)
    return config
