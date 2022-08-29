import logging
from argparse import ArgumentParser
from multiprocessing import freeze_support
from pathlib import Path
from shutil import rmtree
from typing import Dict, Optional

import PySimpleGUI as sg

from configuration.configuration import get_configuration
from logging_helpers.setup_logger import (cleanup_logger, message_debug,
                                          message_info, setup_logger)
from parquetizer import parquetize_directory
from psdata_to_matlab import convert_psdata_directory

logger = logging.getLogger('main')

WINDOW_TITLE = 'Select Raw Data Folder'


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


def _select_folder() -> Optional[Path]:
    """Opens a folder picker GUI for user input

    Returns
    -------
    Optional[Path]
        The chosen path, or None if the GUI window is closed
    """
    left_col = [[sg.Text('Folder'), sg.In(
        size=(25, 1), enable_events=True, key='-FOLDER-'), sg.FolderBrowse()]]
    layout = [[sg.Column(left_col, element_justification='c')]]
    window = sg.Window(WINDOW_TITLE, layout, resizable=True)

    done = False
    folder = None
    while not done:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit', '-FOLDER-'):
            done = True
            if event == '-FOLDER-':
                folder = values['-FOLDER-']

    window.close()
    if folder is not None:
        return Path(folder)
    else:
        return folder


def main(config: Dict, psdata_folder_str: Optional[str] = None):
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
        psdata_folder_path = _select_folder()
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

        if config.get('delete_matlab'):
            message_info("", logger, in_log=False)
            message_info("Removing Matlab files", logger)
            rmtree(matlab_directory)
        message_info('Done', logger)
        cleanup_logger(logger)
    else:
        print('Closing...')


def setup_parser() -> ArgumentParser:
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

    parser.add_argument(
        '--source', '-s',
        help='location of PSData source folder')
    parser.add_argument(
        '--config', '-c',
        help=('Location of YAML configuration file. '
              'This file will override any default configuration file, '
              'and will be overridden by any arguments given here'))
    parser.add_argument(
        '--delete-matlab', '-d',
        action='store_true',
        help='delete generated Matlab files when conversion is done')
    parser.add_argument(
        '--fresh-destination', '-f',
        action='store_true',
        help='delete any previous Matlab and Parquet files if they exist')
    parser.add_argument(
        '--files-limit', '-l',
        type=int,
        help=('limit number of PSData files processed. '
              'Omit this (or use 0) for no limit (process all files in folder)'))
    parser.add_argument(
        '--psdata-tasks',
        type=int,
        help=('max number of simultaneous PSData to Matlab conversions. '
              'Use 0 for no limit (process all given files at the same time)'))
    parser.add_argument(
        '--parquet-tasks',
        type=int,
        help=('max number of simultaneous Matlab to Parquet conversions. '
              'Use 0 to do as many tasks as possible on this machine'))
    parser.add_argument(
        '--parquet-files',
        type=int,
        help=('max number of simultaneous Matlab files loaded per conversion. '
              'Use 0 to load as many files as possible'))

    return parser


if __name__ == "__main__":
    freeze_support()
    parser = setup_parser()

    args = parser.parse_args()
    args_dict = vars(args)
    # source and config are only needed here, not in config
    source_path = args_dict.pop('source', None)
    config_path = args_dict.pop('config', None)
    config = get_configuration(args_dict, config_path)

    main(config, psdata_folder_str=source_path)
