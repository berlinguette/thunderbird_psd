from itertools import islice
from pathlib import Path
from typing import Optional


def get_limited_files(folder_path: Path, limit: Optional[int]):
    if limit is None or limit == 0:
        files = folder_path.iterdir()
    else:
        files = islice(folder_path.iterdir(), limit)
    return files
