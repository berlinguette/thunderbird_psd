import asyncio
import subprocess
import sys
import timeit
from itertools import islice
from math import ceil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, TypeVar

from setup_logger import setup_logger

MAX_CONCURRENT_TASKS = 5
FORMAT = 'mat'
FILES_LIMIT = 10

logger = setup_logger('psdata_converter')


def subprocess_results_printer(
    command_text: str,
    returncode: int,
    stdout: Optional[str],
    stderr: Optional[str]
):
    logger.info(f'[{command_text} exited with {returncode}]')
    if stdout:
        logger.debug(f'[stdout]\n{stdout}')
    if stderr:
        logger.debug(f'[stderr]\n{stderr}')


def generate_shell_command(file_path: Path, destination: Path):
    return [
        'picoscope',
        '/c', f'{file_path.resolve()}',
        '/d', f'{destination.resolve()}',
        '/f', f'{FORMAT}',
        '/q',
        '/b', 'all'
    ]


def get_limited_files(folder_path: Path, limit: Optional[int]):
    if limit is None:
        files = folder_path.iterdir()
    else:
        files = islice(folder_path.iterdir(), limit)
    return files


T = TypeVar('T')
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
        f'Converting {file_path.name}', returncode, None, None)


async def process_files_async(
    folder_path: Path,
    destination: Path,
    num_files=FILES_LIMIT,
    chunk_size=MAX_CONCURRENT_TASKS
):
    start = timeit.default_timer()

    cors = [process_file_async(file_path, destination)
            for file_path in get_limited_files(folder_path, num_files)]
    # Chunking code from https://fredrikaverpil.github.io/2017/06/20/async-and-await-with-subprocesses/
    chunks, chunks_count = chunkify(cors, chunk_size)
    for chunk_index, chunk in enumerate(chunks):
        logger.info(f'Beginning work on chunk {chunk_index+1}/{chunks_count}')
        await asyncio.gather(*chunk)
        logger.info(f'Completed work on chunk {chunk_index+1}/{chunks_count}')

    stop = timeit.default_timer()
    exec_time = stop - start
    logger.debug(f"Method executed in {exec_time:.4f} seconds")


def process_files_popen(
    folder_path: Path,
    destination: Path,
    num_files=FILES_LIMIT,
    chunk_size=MAX_CONCURRENT_TASKS
):
    start = timeit.default_timer()

    file_paths = [file_path for file_path in get_limited_files(
        folder_path, num_files)]
    chunks, chunks_count = chunkify(file_paths, chunk_size)

    for chunk_index, chunk in enumerate(chunks):
        logger.info(f'Beginning work on chunk {chunk_index+1}/{chunks_count}')
        procs = [(file_path, subprocess.Popen(generate_shell_command(
            file_path, destination))) for file_path in chunk]
        for file_path, proc in procs:
            returncode = proc.wait()
            subprocess_results_printer(
                f'Converting {file_path.name}', returncode, None, None)
        logger.info(f'Completed work on chunk {chunk_index+1}/{chunks_count}')

    stop = timeit.default_timer()
    exec_time = stop - start
    logger.debug(f"Method executed in {exec_time:.4f} seconds")


def convert_psdata_directory(
    folder_path: Path,
    destination: Path,
    num_files=FILES_LIMIT,
    chunk_size=MAX_CONCURRENT_TASKS
):
    # asyncio.run(process_files_async(folder_path, destination, num_files, chunk_size))
    process_files_popen(folder_path, destination, num_files, chunk_size)


if __name__ == "__main__":
    if 'win32' in sys.platform:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    psdata_directory = Path("sample_dataset/raw_data/psdata")
    destination = psdata_directory.parent / 'mat'
    convert_psdata_directory(psdata_directory, destination)
