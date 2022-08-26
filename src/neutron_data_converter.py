import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree
from typing import Dict, Optional

import PySimpleGUI as sg

from configuration import get_configuration
from parquetizer import parquetize_directory
from psdata_to_matlab import convert_psdata_directory
from setup_logger import (cleanup_logger, setup_logger, tqdm_log_debug,
                          tqdm_log_info)

logger = logging.getLogger('main')

WINDOW_TITLE = 'Select Raw Data Folder'


def prepare_destination(destination_path: Path, fresh_destination: bool):
    if fresh_destination and destination_path.is_dir():
        tqdm_log_debug(
            f'Deleting destination {destination_path}',
            logger, on_screen=True)
        rmtree(destination_path)
    # destinations must exist for converters to work properly
    if not destination_path.exists():
        tqdm_log_debug(
            f'Making destination {destination_path}',
            logger, on_screen=True)
        destination_path.mkdir()


def select_folder() -> Optional[Path]:
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


def main(config: Dict, psdata_folder_path: Path = None):
    if psdata_folder_path is None:
        psdata_folder_path = select_folder()

    if psdata_folder_path is not None:
        setup_logger(logger, psdata_folder_path.parent.parent)
        tqdm_log_info(f'Converting files at {psdata_folder_path}', logger)
        tqdm_log_debug(
            f'Final configuration: {config}', logger, on_screen=False)

        matlab_directory = psdata_folder_path.parent / 'mat'
        parquet_directory = psdata_folder_path.parent.parent / 'processed_data' / 'parquets'
        fresh_destination = config.get('fresh_destination', False)
        tqdm_log_info('Preparing destination folders', logger)
        prepare_destination(matlab_directory, fresh_destination)
        tqdm_log_debug('Matlab destination done', logger)
        prepare_destination(parquet_directory, fresh_destination)
        tqdm_log_debug('Parquet destination done', logger)

        tqdm_log_info("Converting PSData to Matlab", logger)
        convert_psdata_directory(psdata_folder_path, matlab_directory, config)
        tqdm_log_info("Converting Matlab to Parquet", logger)
        parquetize_directory(matlab_directory, parquet_directory, config)

        if config.get('delete_matlab'):
            tqdm_log_info("Removing Matlab files", logger)
            rmtree(matlab_directory)
        cleanup_logger(logger)
    else:
        print('Closing...')


def setup_parser() -> ArgumentParser:
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
    parser = setup_parser()

    args = parser.parse_args()
    args_dict = vars(args)
    # source and config are only needed here, not in config
    source_path = args_dict.pop('source', None)
    config_path = args_dict.pop('config', None)
    config = get_configuration(args_dict, config_path)

    main(config, psdata_folder_path=Path(source_path))
