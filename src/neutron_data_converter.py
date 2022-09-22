import logging
from argparse import ArgumentParser
from importlib.util import find_spec
from multiprocessing import freeze_support
from os import environ
from pathlib import Path
from shutil import rmtree
from typing import Any, Dict, Optional

from configuration.configuration import (get_configuration, load_config_setup,
                                         populate_args_parser)
from logging_helpers.setup_logger import (CONVERSION_LOG_FILENAME,
                                          cleanup_logger, message_debug,
                                          message_info, setup_logger)
from parquetizer import parquetize_directory
from psdata_to_matlab import convert_psdata_directory
from ui.converter_gui import converter_gui

logger = logging.getLogger('main')

WINDOW_TITLE = 'Select Experiment Folder'
RAW_DATA_FOLDER_NAME = 'raw_data'
PSDATA_FOLDER_NAME = 'psdata'
MATLAB_FOLDER_NAME = 'mat'
PARQUET_FOLDER_NAME = 'parquet'


def _setup_parser(config_setup: Dict[str, Any]) -> ArgumentParser:
    """Produces ArgumentParser with all needed arguments

    Returns
    -------
    ArgumentParser
        ArgumentParser configured with all supported command line arguments
    """
    parser = ArgumentParser(
        prog="PSData to Parquet Converter",
        description=("Converts PSData files from experiment"
                     " to Matlab and Parquet files"))
    parser = populate_args_parser(parser, config_setup)

    return parser


def _find_experiment_root(folder_path: Path, found_psdata: bool = False, found_rawdata: bool = False) -> Optional[Path]:
    RAW_DATA_FOLDERS = (PSDATA_FOLDER_NAME,
                        PARQUET_FOLDER_NAME, MATLAB_FOLDER_NAME)
    RAW_DATA_METADATA_FILE = 'exp_times.csv'
    ROOT_METADATA_FILE = 'exp_info.txt'

    root_path = None
    if folder_path.name in RAW_DATA_FOLDERS:
        is_psdata_folder = folder_path.name == PSDATA_FOLDER_NAME
        root_path = _find_experiment_root(
            folder_path.parent, found_psdata=is_psdata_folder)
    elif folder_path.name == RAW_DATA_FOLDER_NAME:
        checks = [
            (folder_path / RAW_DATA_METADATA_FILE).exists(),
            found_psdata or (folder_path / PSDATA_FOLDER_NAME).exists()
        ]
        if all(checks):
            root_path = _find_experiment_root(
                folder_path.parent,
                found_psdata=found_psdata,
                found_rawdata=True
            )
        pass
    else:
        checks = [
            (folder_path / ROOT_METADATA_FILE).exists(),
            found_rawdata or (folder_path / RAW_DATA_FOLDER_NAME).exists(),
            found_psdata or (folder_path / RAW_DATA_FOLDER_NAME /
                             PSDATA_FOLDER_NAME).exists(),
        ]
        if all(checks):
            root_path = folder_path
    return root_path  # TODO test this


def _prepare_destination(destination_path: Path, fresh_destination: bool):
    """Ensures that the destination path exists, and is empty if needed

    Parameters
    ----------
    destination_path : Path
        Destination path to prepare
    fresh_destination : bool
        If true, deletes any files or folders at the destination folder
    """
    if fresh_destination and destination_path.is_dir():
        message_debug(
            f'Deleting destination {destination_path}',
            logger, on_screen=False)
        rmtree(destination_path)
    # destinations must exist for converters to work properly
    if not destination_path.exists():
        message_debug(
            f'Making destination {destination_path}',
            logger, on_screen=False)
        destination_path.mkdir()


def main(
    config: Dict,
    config_setup: Dict[str, Any],
    folder_str: Optional[str] = None
):
    """Runs the neutron data conversion process:
    - Running the folder picker GUI if needed
    - Preparing destination folders
    - Running conversion steps
      - PSData to Matlab
      - Matlab to Parquet
    - Deleting Matlab files if needed

    Parameters
    ----------
    config : Dict
        Configuration data. See configuration.py for more info
    config_setup: Dict[str, Any]
        Configuration setup data
    folder_str : Optional[str], optional
        location of the experiment folder from command line arguments.
        (This can also be the `raw_data` folder, or any of its subfolders.)
        If not provided, the UI window will be launched.
    """
    if folder_str is None:
        config, folder_path = converter_gui(config, config_setup)
    else:
        folder_path = Path(folder_str)

    if folder_path is not None:
        experiment_root = _find_experiment_root(folder_path)
        if experiment_root is not None:
            raw_data_folder = experiment_root / RAW_DATA_FOLDER_NAME
            psdata_folder = raw_data_folder / PSDATA_FOLDER_NAME
            matlab_folder = raw_data_folder / MATLAB_FOLDER_NAME
            parquet_folder = raw_data_folder / PARQUET_FOLDER_NAME
            setup_logger(logger, experiment_root / CONVERSION_LOG_FILENAME)
            message_info(f'Converting files at {experiment_root}', logger)
            message_info("", logger, in_log=False)
            message_debug(
                f'Final configuration: {config}', logger, on_screen=False)

            fresh_destination = config.get('fresh_destination', False)
            message_info('Preparing destination folders', logger)
            _prepare_destination(matlab_folder, fresh_destination)
            message_debug(' - Matlab destination done', logger)
            _prepare_destination(parquet_folder, fresh_destination)
            message_debug(' - Parquet destination done', logger)
            message_info("", logger, in_log=False)

            message_info("Converting PSData to Matlab", logger)
            message_info(
                "You might see other windows pop up quickly."
                + " This is normal. Don't panic!",
                logger,
                in_log=False
            )
            convert_psdata_directory(psdata_folder, matlab_folder, config)
            message_info("", logger, in_log=False)
            message_info("Converting Matlab to Parquet", logger)
            parquetize_directory(matlab_folder, parquet_folder, config)

            if not config.get('keep_matlab'):
                message_info("", logger, in_log=False)
                message_info("Removing Matlab files", logger)
                rmtree(matlab_folder)
            message_info('Conversion complete!', logger)
            cleanup_logger(logger)
            input("Press Enter to close window")
        else:
            print('Selected folder is not a valid experiment folder')
    else:
        print('Closing...')


if __name__ == "__main__":
    # taken from https://stackoverflow.com/a/68666505
    if '_PYIBoot_SPLASH' in environ and find_spec("pyi_splash"):
        # splash module only exists when packaged
        from time import sleep

        import pyi_splash  # type: ignore
        pyi_splash.update_text('Loading complete')
        sleep(1)
        pyi_splash.close()

    freeze_support()  # needed for Windows multiprocessing/processpool

    config_setup = load_config_setup()
    parser = _setup_parser(config_setup)

    args = parser.parse_args()
    args_dict = vars(args)
    # source and config are only needed here, not in config
    source_path = args_dict.pop('source', None)
    config_path = args_dict.pop('config', None)
    config = get_configuration(args_dict, config_setup, config_path)

    main(config, config_setup, folder_str=source_path)
