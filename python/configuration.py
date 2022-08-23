from pathlib import Path
from typing import Dict

import yaml


def override_config(old_config: Dict, new_config: Dict) -> Dict:
    return {**old_config, **new_config}


def load_configuration(config_path: Path = None):
    default_path = Path('python/default_config.yaml')
    with open(default_path, 'r') as conf_file:
        config = yaml.safe_load(conf_file)
    
    # given config overrides any default config values
    if config_path is not None:
        with open(config_path, 'r') as conf_file:
            new_config = yaml.safe_load(conf_file)
            config = override_config(config, new_config)

    return config

def get_configuration(cli_args: Dict, config_path: Path = None) -> Dict:
    config = load_configuration(config_path)
    # omitted args get None value, which would override config
    # so we must remove them
    cli_args = {k: v for k, v in cli_args.items() if v is not None}
    config = override_config(config, cli_args)
    return config
