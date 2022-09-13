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
from logging_helpers.setup_logger import (cleanup_logger, message_debug,
                                          message_info, setup_logger)
from parquetizer import parquetize_directory
from psdata_to_matlab import convert_psdata_directory
from ui.converter_gui import converter_gui

logger = logging.getLogger('main')


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
    psdata_folder_str: Optional[str] = None
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
    psdata_folder_str : Optional[str], optional
        location of the PSData folder from command line arguments.
        If not provided, the folder picker window will be launched.
    """
    if psdata_folder_str is None:
        config, psdata_folder_path = converter_gui(config, config_setup)
    else:
        psdata_folder_path = Path(psdata_folder_str)

    if psdata_folder_path is not None:
        setup_logger(logger, psdata_folder_path.parent.parent)
        message_info(f'Converting files at {psdata_folder_path}', logger)
        message_info("", logger, in_log=False)
        message_debug(
            f'Final configuration: {config}', logger, on_screen=False)

        matlab_directory = psdata_folder_path.parent / 'mat'
        parquet_directory = psdata_folder_path.parent.parent / 'raw_data' / 'parquet'
        fresh_destination = config.get('fresh_destination', False)
        message_info('Preparing destination folders', logger)
        _prepare_destination(matlab_directory, fresh_destination)
        message_debug(' - Matlab destination done', logger)
        _prepare_destination(parquet_directory, fresh_destination)
        message_debug(' - Parquet destination done', logger)
        message_info("", logger, in_log=False)

        message_info("Converting PSData to Matlab", logger)
        message_info(
            "You might see other windows pop up quickly."
            + " This is normal. Don't panic!",
            logger,
            in_log=False
        )
        convert_psdata_directory(psdata_folder_path, matlab_directory, config)
        message_info("", logger, in_log=False)
        message_info("Converting Matlab to Parquet", logger)
        parquetize_directory(matlab_directory, parquet_directory, config)

        if not config.get('keep_matlab'):
            message_info("", logger, in_log=False)
            message_info("Removing Matlab files", logger)
            rmtree(matlab_directory)
        message_info('Conversion complete!', logger)
        cleanup_logger(logger)
        input("Press Enter to close window")
    else:
        print('Closing...')


if __name__ == "__main__":
    # taken from https://stackoverflow.com/a/68666505
    if '_PYIBoot_SPLASH' in environ and find_spec("pyi_splash"):
        # splash module only exists when packaged
        import pyi_splash  # type: ignore
        from time import sleep
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

    main(config, config_setup, psdata_folder_str=source_path)
