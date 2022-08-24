import asyncio
import logging
import subprocess
import sys
import timeit
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

from tqdm import tqdm

from logging_helpers.setup_logger import (cleanup_logger, setup_logger,
                                          tqdm_log_debug)
from utilities.get_limited_files import get_limited_files

FORMAT = 'mat'
T = TypeVar('T')
logger = logging.getLogger('psdata_converter')


def subprocess_results_printer(
    command_text: str,
    returncode: int,
    stdout: Optional[str],
    stderr: Optional[str],
    on_screen: bool = True
):
    tqdm_log_debug(f'[{command_text} exited with {returncode}]',
                   logger, on_screen=on_screen)
    if stdout:
        tqdm_log_debug(f'[stdout]\n{stdout}', logger, on_screen=on_screen)
    if stderr:
        tqdm_log_debug(f'[stderr]\n{stderr}', logger, on_screen=on_screen)


def generate_shell_command(file_path: Path, destination: Path):
    return [
        'picoscope',
        '/c', f'{file_path.resolve()}',
        '/d', f'{destination.resolve()}',
        '/f', f'{FORMAT}',
        '/q',
        '/b', 'all'
    ]


def make_chunks(list_to_chunk: List[T], chunk_size: int) -> Iterable[List[T]]:
    # Taken from https://stackoverflow.com/a/312464
    for i in range(0, len(list_to_chunk), chunk_size):
        yield list_to_chunk[i: i + chunk_size]


def get_number_of_chunks(list_length: int, chunk_size: int) -> int:
    return ceil(list_length / chunk_size)


def chunkify(list_to_chunk: List[T], chunk_size: int) -> Tuple[Iterable[List[T]], int]:
    if chunk_size == 0:
        chunks = [list_to_chunk]
        chunk_count = len(chunks)
    else:
        chunks = make_chunks(list_to_chunk, chunk_size)
        chunk_count = get_number_of_chunks(len(list_to_chunk), chunk_size)
    return (chunks, chunk_count)


async def process_file_async(file_path: Path, destination: Path):
    proc = await asyncio.create_subprocess_exec(
        *generate_shell_command(file_path, destination))
    await proc.communicate()
    if proc.returncode is not None:
        returncode = proc.returncode
    else:  # only happens if process not done, which should never happen with await
        returncode = 420  # computer must be high
    subprocess_results_printer(
        f'Converting {file_path.name}', returncode, None, None,
        on_screen=False
    )


async def process_files_async(
    folder_path: Path,
    destination: Path,
    num_files: Optional[int],
    chunk_size: int
):
    start = timeit.default_timer()

    cors = [process_file_async(file_path, destination, logger)
            for file_path in get_limited_files(folder_path, num_files)]
    # Chunking code from https://fredrikaverpil.github.io/2017/06/20/async-and-await-with-subprocesses/
    chunks, chunks_count = chunkify(cors, chunk_size)
    for chunk_index, chunk in enumerate(chunks):
        tqdm_log_debug(
            f'Beginning work on chunk {chunk_index+1}/{chunks_count}',
            logger, on_screen=False)
        await asyncio.gather(*chunk)
        tqdm_log_debug(
            f'Completed work on chunk {chunk_index+1}/{chunks_count}',
            logger, on_screen=False)

    stop = timeit.default_timer()
    exec_time = stop - start
    tqdm_log_debug(f"Method executed in {exec_time:.4f} seconds",
                   logger, on_screen=False)


def process_files_popen(
    folder_path: Path,
    destination: Path,
    num_files: Optional[int],
    chunk_size: int
):
    start = timeit.default_timer()

    file_paths = [file_path for file_path in get_limited_files(
        folder_path, num_files)]
    chunks, chunks_count = chunkify(file_paths, chunk_size)

    with tqdm(desc='PSData Files', unit='file', total=len(file_paths)) as progress_bar:
        for chunk_index, chunk in enumerate(chunks):
            tqdm_log_debug(
                f'Beginning work on chunk {chunk_index+1}/{chunks_count}',
                logger, on_screen=False)
            procs = [(file_path, subprocess.Popen(generate_shell_command(
                file_path, destination))) for file_path in chunk]
            for proc_data in procs:
                file_path, proc = proc_data
                returncode = proc.wait()
                progress_bar.update()
                subprocess_results_printer(
                    f'Converting {file_path.name}', returncode, None, None,
                    on_screen=False)
            tqdm_log_debug(
                f'Completed work on chunk {chunk_index+1}/{chunks_count}',
                logger, on_screen=False)

    stop = timeit.default_timer()
    exec_time = stop - start
    tqdm_log_debug(f"Method executed in {exec_time:.4f} seconds",
                   logger, on_screen=False)


def convert_psdata_directory(
    folder_path: Path,
    destination: Path,
    config: Dict
):
    logging_folder = folder_path.parent.parent
    setup_logger(logger, logging_folder)
    num_files = config.get('files_limit')
    chunk_size = config.get('psdata_tasks', 0)
    process_files_popen(folder_path, destination,
                        num_files, chunk_size)
    # asyncio.run(
    #     process_files_async(
    #         folder_path, destination, num_files, chunk_size
    #     )
    # )
    cleanup_logger(logger)


if __name__ == "__main__":
    from configuration import get_configuration

    if 'win32' in sys.platform:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    psdata_directory = Path("sample_dataset/raw_data/psdata")
    destination = psdata_directory.parent / 'mat'

    config = get_configuration({})
    convert_psdata_directory(psdata_directory, destination, config)
