from pathlib import Path
import asyncio
from typing import Optional, TypeVar, List, Iterable
from itertools import islice
from math import ceil
import timeit
import sys
import subprocess

MAX_CONCURRENT_TASKS = 5
FORMAT = 'mat'
FILES_LIMIT = 10

def subprocess_results_printer(command_text: str, returncode: int, stdout: Optional[str], stderr: Optional[str]):
    print(f'[{command_text} exited with {returncode}]')
    if stdout:
        print(f'[stdout]\n{stdout}')
    if stderr:
        print(f'[stderr]\n{stderr}')
        
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
        yield list_to_chunk[i : i + chunk_size]

def get_number_of_chunks(list_length: int, chunk_size: int) -> int:
    return ceil(list_length / chunk_size)

async def process_file_async(file_path: Path, destination: Path):
    proc = await asyncio.create_subprocess_exec(*generate_shell_command(file_path, destination))
    await proc.communicate()
    if proc.returncode is not None:
        returncode = proc.returncode
    else:  # only happens if process not done, which should never happen with await
        returncode = 420  # computer must be high
    subprocess_results_printer(f'Converting {file_path.name}', returncode, None, None)
    
async def process_files_async(folder_path: Path, destination: Path, num_files = FILES_LIMIT):
    start = timeit.default_timer()
    # movefile = wrap(move)
    
    cors = [process_file_async(file_path, destination) for file_path in get_limited_files(folder_path, num_files)]
    # Chunking code from https://fredrikaverpil.github.io/2017/06/20/async-and-await-with-subprocesses/
    if MAX_CONCURRENT_TASKS == 0:
        chunks = [cors]
        num_chunks = len(chunks)
    else:
        chunks = make_chunks(cors, MAX_CONCURRENT_TASKS)
        num_chunks = get_number_of_chunks(len(cors), MAX_CONCURRENT_TASKS)
    for chunk_index, chunk in enumerate(chunks):
        print(f'Beginning work on chunk {chunk_index+1}/{num_chunks}')
        await asyncio.gather(*chunk)
        print(f'Completed work on chunk {chunk_index+1}/{num_chunks}')
    
    # files_cors = [movefile(str(x), destination) for x in folder_path.iterdir() if x.is_dir()]
    # await asyncio.gather(*files_cors)

    stop = timeit.default_timer()
    exec_time = stop - start
    print(f"Method executed in {exec_time:.4f} seconds")

def process_files_popen(folder_path: Path, destination: Path, num_files=FILES_LIMIT):
    start = timeit.default_timer()

    file_paths = [file_path for file_path in get_limited_files(folder_path, num_files)]
    if MAX_CONCURRENT_TASKS == 0:
        chunks = [file_paths]
        num_chunks = len(chunks)
    else:
        chunks = make_chunks(file_paths, MAX_CONCURRENT_TASKS)
        num_chunks = get_number_of_chunks(len(file_paths), MAX_CONCURRENT_TASKS)

    for chunk_index, chunk in enumerate(chunks):
        print(f'Beginning work on chunk {chunk_index+1}/{num_chunks}')
        procs = [(file_path, subprocess.Popen(generate_shell_command(file_path, destination))) for file_path in chunk]
        for file_path, proc in procs:
            returncode = proc.wait()
            subprocess_results_printer(f'Converting {file_path.name}', returncode, None, None)
        print(f'Completed work on chunk {chunk_index+1}/{num_chunks}')

    stop = timeit.default_timer()
    exec_time = stop - start
    print(f"Method executed in {exec_time:.4f} seconds")

def convert_psdata_directory(folder_path: Path, destination: Path, num_files=FILES_LIMIT):
    asyncio.run(process_files_async(folder_path, destination))
    # process_files_popen(folder_path, destination, num_files)

if __name__ == "__main__":
    if 'win32' in sys.platform:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    psdata_directory = Path("sample_dataset/raw_data/psdata")
    destination = psdata_directory.parent / 'mat'
    convert_psdata_directory(psdata_directory, destination)
