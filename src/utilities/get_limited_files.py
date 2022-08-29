from itertools import islice
from pathlib import Path
from typing import Iterable, Optional


def get_limited_files(
    folder_path: Path,
    limit: Optional[int]
) -> Iterable[Path]:
    """Iterates over the files in the given folder, limited to the 
       desired number

    Parameters
    ----------
    folder_path : Path
        source folder. Files in this folder will be iterated over. 
    limit : Optional[int]
        maximum number of files to iterate over

    Returns
    -------
    Iterable[Path]
        limited file iterator
    """
    if limit is None or limit == 0:
        files = folder_path.iterdir()
    else:
        files = islice(folder_path.iterdir(), limit)
    return files
