from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree
from typing import Dict, Optional

import PySimpleGUI as sg

from configuration import get_configuration
from psdata_converter import convert_psdata_directory
from parquetizer import parquetize_directory
from setup_logger import (
    cleanup_logger,
    setup_logger
)

WINDOW_TITLE = 'Select Raw Data Folder'


def prepare_destination(destination_path: Path):
    if destination_path.is_dir():
        rmtree(destination_path)
    # destinations must exist for converters to work properly
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
        logger = setup_logger('main', psdata_folder_path.parent.parent)
        matlab_directory = psdata_folder_path.parent / 'mat'
        parquet_directory = psdata_folder_path.parent.parent / 'processed_data' / 'parquets'
        logger.info('Preparing destination folders')
        prepare_destination(matlab_directory)
        prepare_destination(parquet_directory)

        logger.info("Converting PSData to Matlab")
        convert_psdata_directory(psdata_folder_path, matlab_directory, config)
        logger.info("Converting Matlab to Parquet")
        parquetize_directory(matlab_directory, parquet_directory, config)

        if config.get('delete_matlab'):
            logger.info("Removing Matlab files")
            rmtree(matlab_directory)
    else:
        logger.info('Closing...')


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
    # source and config are only needed here, so strip them out
    source_path = args_dict.pop('source', None)
    config_path = args_dict.pop('config', None)
    # get config
    config = get_configuration(args_dict, config_path)

    main(config, psdata_folder_path=source_path)
