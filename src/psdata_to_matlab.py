import logging
import subprocess
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

from tqdm import tqdm

from configuration.configuration import load_config_setup
from logging_helpers.setup_logger import (Messenger, cleanup_logger,
                                          get_conversion_logfile_path,
                                          setup_logger)
from utilities.constants import BAR_FORMAT
from utilities.get_limited_files import get_limited_files
from utilities.timing import Timer

FORMAT = 'mat'
T = TypeVar('T')
logger = logging.getLogger('psdata_converter')
log_only_messenger = Messenger(logger, on_screen=False)


def _subprocess_results_printer(
    command_text: str,
    returncode: int,
    stdout: Optional[str],
    stderr: Optional[str],
    on_screen: bool = True
):
    """Print subprocess result message to log and (optionally) to console

    Parameters
    ----------
    command_text : str
        text describing the subprocess command attempted
    returncode : int
        subprocess return code
    stdout : Optional[str]
        text returned by subprocess on stdout, or None to ignore entirely
    stderr : Optional[str]
        text returned by subprocess on stderr, or None to ignore entirely
    on_screen : bool, optional
        whether to display subprocess messages on console screen, default True.
        Messages are always written to log
    """
    subprocess_messenger = Messenger(logger, on_screen=on_screen)
    subprocess_messenger.debug(f'[{command_text} exited with {returncode}]')
    if stdout:
        subprocess_messenger.debug(f'[stdout]\n{stdout}')
    if stderr:
        subprocess_messenger.debug(f'[stderr]\n{stderr}')


def _generate_shell_command(file_path: Path, destination: Path) -> List[str]:
    """Generates shell command to convert PSData file to Matlab folder

    Parameters
    ----------
    file_path : Path
        path to PSData file
    destination : Path
        path to save Matlab folder

    Returns
    -------
    List[str]
        shell command in list form
    """
    return [
        'picoscope',
        '/c', f'{file_path.resolve()}',
        '/d', f'{destination.resolve()}',
        '/f', f'{FORMAT}',
        '/q',
        '/b', 'all'
    ]


def _make_chunks(list_to_chunk: List[T], chunk_size: int) -> Iterable[List[T]]:
    """Generates a series of smaller lists of designated length (AKA chunks)
       from a longer list

    If the list length is not divisible by the chunk size, every chunk except
    the last will be equal length. The last chunk will receive the remainder.

    This function returns a generator allowing lazy chunk production.

    Taken from https://stackoverflow.com/a/312464

    Parameters
    ----------
    list_to_chunk : List[T]
        original list to be chunked
    chunk_size : int
        Desired chunk size

    Returns
    -------
    Iterable[List[T]]
        chunk iterator

    Raises
    ------
    ValueError
        when chunk_size is not positive integer
    """
    if chunk_size <= 0:
        raise ValueError(
            f"Given chunk_size {chunk_size} must be positive integer")
    for i in range(0, len(list_to_chunk), chunk_size):
        yield list_to_chunk[i: i + chunk_size]


def _get_number_of_chunks(list_length: int, chunk_size: int) -> int:
    """Determine the number of chunks produced when chunking a list

    Since the make_chunks function is a generator, this function helps avoid
    turning the produced iterator into a list to determine the length.

    Parameters
    ----------
    list_length : int
        length of original list to be chunked
    chunk_size : int
        Desired chunk size

    Returns
    -------
    int
        number of chunks that will be produced

    Raises
    ------
    ValueError
        when chunk_size is not positive integer
    """
    if chunk_size <= 0:
        raise ValueError(
            f"Given chunk_size {chunk_size} must be positive integer")
    return ceil(list_length / chunk_size)


def _chunkify(
    list_to_chunk: List[T],
    chunk_size: int
) -> Tuple[Iterable[List[T]], int]:
    """Makes a chunk iterator for this list, as well as providing the eventual 
       number of chunks that will be produced

    Parameters
    ----------
    list_to_chunk : List[T]
        original list to be chunked
    chunk_size : int
        Desired chunk size. Chunk size of 0 produces 1 chunk consisting of the
        entire list

    Returns
    -------
    Tuple[Iterable[List[T]], int]
        tuple of (chunk iterator, number of chunks that will be produced)

    Raises
    ------
    ValueError
        when chunk_size is not positive integer or 0
    """
    if chunk_size == 0:
        chunks = [list_to_chunk]
        chunk_count = 1
    else:
        chunks = _make_chunks(list_to_chunk, chunk_size)
        chunk_count = _get_number_of_chunks(len(list_to_chunk), chunk_size)
    return (chunks, chunk_count)


def _process_files(
    folder_path: Path,
    destination: Path,
    num_files: Optional[int],
    chunk_size: int
):
    """Processes PSData files into Matlab files concurrently

    Parameters
    ----------
    folder_path : Path
        path to folder containing the PSData files
    destination : Path
        path to destination folder for the Matlab files
    num_files : Optional[int]
        Number of files to process. If None or not given, all files are processed
    chunk_size : int
        Maximum number of files to process concurrently
    """
    timer = Timer(start_now=True)

    file_paths = [file_path for file_path in get_limited_files(
        folder_path, num_files)]
    chunks, chunks_count = _chunkify(file_paths, chunk_size)

    with tqdm(
        desc='PSData Files',
        unit='file',
        total=len(file_paths),
        bar_format=BAR_FORMAT
    ) as progress_bar:
        for chunk_index, chunk in enumerate(chunks):
            log_only_messenger.debug(f'Beginning work on chunk {chunk_index+1}/{chunks_count}')
            procs = [(file_path, subprocess.Popen(_generate_shell_command(
                file_path, destination))) for file_path in chunk]
            for proc_data in procs:
                file_path, proc = proc_data
                returncode = proc.wait()
                progress_bar.update()
                _subprocess_results_printer(
                    f'Converting {file_path.name}', returncode, None, None,
                    on_screen=False)
            log_only_messenger.debug(f'Completed work on chunk {chunk_index+1}/{chunks_count}')

    exec_time = timer.stop_timer()
    log_only_messenger.debug(f"Method executed in {timer.format_elapsed_time(exec_time)}")


def convert_psdata_directory(
    folder_path: Path,
    destination: Path,
    config: Dict
):
    """Converts PSData files in given directory to Matlab files

    Processing involves running PicoScope software's command line interface, 
    which momentarily launches a PicoScope software window. This causes many 
    blank windows to pop-up momentarily (though some can stay for an 
    extended amount of time).

    Parameters
    ----------
    folder_path : Path
        path to folder containing the PSData files
    destination : Path
        path to destination folder for the Matlab files
    config : Dict
        Configuration data. See configuration.py for more info
    """
    log_file = get_conversion_logfile_path(folder_path.parent.parent)
    setup_logger(logger, log_file)
    log_only_messenger.debug(f'PSData source: {folder_path}')
    log_only_messenger.debug(f'Matlab destination: {destination}')
    num_files = config.get('files_limit')
    chunk_size = config.get('psdata_tasks', 0)
    _process_files(folder_path, destination,
                   num_files, chunk_size)
    cleanup_logger(logger)


if __name__ == "__main__":
    from configuration.configuration import get_configuration

    psdata_directory = Path(
        "sample_datasets/20220824_CERC_background/raw_data/psdata")
    destination = psdata_directory.parent / 'mat'

    config_setup = load_config_setup()
    config = get_configuration(
        {'files_limit': 5, 'psdata_tasks': 5}, config_setup)
    convert_psdata_directory(psdata_directory, destination, config)
