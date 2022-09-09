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

WINDOW_TITLE = 'Select Experiment Folder'
RAW_DATA_FOLDER_NAME = 'raw_data'
PSDATA_FOLDER_NAME = 'psdata'
MATLAB_FOLDER_NAME = 'mat'
PARQUET_FOLDER_NAME = 'parquet'


def _setup_parser() -> ArgumentParser:
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
        '--keep-matlab', '-k',
        action='store_true',
        help='keep generated Matlab files when conversion is done')
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


def _find_experiment_root(folder_path: Path, found_psdata: bool = False, found_rawdata: bool = False) -> Optional[Path]:
    # check folder name
    # if "psdata","parquet" or "mat", do "raw_data" check on parent
    # if "raw_data", check for "exp_times.csv" and "psdata" subfolder; if so, do "other folder" check on parent
    # if any other folder, check if it has "exp_info.txt" and correct subfolders; if so, return this folder
    RAW_DATA_FOLDERS = (PSDATA_FOLDER_NAME, PARQUET_FOLDER_NAME, MATLAB_FOLDER_NAME)
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
        # check for "exp_info.txt"
        # check for raw_data (and raw_data/psdata) if needed
        # if all okay, return this path
        checks = [
            (folder_path / ROOT_METADATA_FILE).exists(),
            found_rawdata or (folder_path / RAW_DATA_FOLDER_NAME).exists(),
            found_psdata or (folder_path / RAW_DATA_FOLDER_NAME / PSDATA_FOLDER_NAME).exists(),
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


def main(config: Dict, folder_str: Optional[str] = None):
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
    if folder_str is None:
        folder_path = _select_folder()
    else:
        folder_path = Path(folder_str)

    if folder_path is not None:
        # get experiment root
        # get all needed subfolders from root
        experiment_root = _find_experiment_root(folder_path)
        if experiment_root is not None:
            raw_data_folder = experiment_root / RAW_DATA_FOLDER_NAME
            psdata_folder = raw_data_folder / PSDATA_FOLDER_NAME
            matlab_folder = raw_data_folder / MATLAB_FOLDER_NAME
            parquet_folder = raw_data_folder / PARQUET_FOLDER_NAME
            setup_logger(logger, experiment_root)
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
            message_info('Done', logger)
            cleanup_logger(logger)
        else:
            print('Selected folder is not a valid experiment folder')
    else:
        print('Closing...')


if __name__ == "__main__":
    freeze_support()  # needed for Windows multiprocessing/processpool
    parser = _setup_parser()

    args = parser.parse_args()
    args_dict = vars(args)
    # source and config are only needed here, not in config
    source_path = args_dict.pop('source', None)
    config_path = args_dict.pop('config', None)
    config = get_configuration(args_dict, config_path)

    main(config, folder_str=source_path)
