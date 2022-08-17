from pathlib import Path
from shutil import rmtree
from typing import Optional
import logging

import PySimpleGUI as sg

from main import parquetize_directory
from psdata_converter import MAX_CONCURRENT_TASKS, convert_psdata_directory

WINDOW_TITLE = 'Select Raw Data Folder'

logger = logging.getLogger('main')
def prepare_destination(destination_path: Path):
    if destination_path.is_dir():
        rmtree(destination_path)
    else:
        destination_path.mkdir()

def select_folder() -> Optional[Path]:
    left_col = [[sg.Text('Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()]]
    layout = [[sg.Column(left_col, element_justification='c')]]
    window = sg.Window(WINDOW_TITLE, layout,resizable=True)
    
    done = False
    folder = None
    while not done:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit', '-FOLDER-'):
            done = True
            if event == '-FOLDER-':
                folder = values['-FOLDER-']
    
    window.close()
    return Path(folder)

if __name__ == "__main__":
    psdata_folder_path = None
    while psdata_folder_path is None:
        psdata_folder_path = select_folder()
    matlab_directory = psdata_folder_path.parent / 'mat'
    parquet_directory = psdata_folder_path.parent.parent / 'processed_data' / 'parquets'
    prepare_destination(matlab_directory)
    prepare_destination(parquet_directory)
    logger.info("Converting PSData to Matlab")
    convert_psdata_directory(psdata_folder_path, matlab_directory)
    logger.info("Converting Matlab to Parquet")
    parquetize_directory(matlab_directory, parquet_directory)

    