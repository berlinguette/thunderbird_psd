import re
import pickle as pkl
from pathlib import Path

from data_processing.types import WindowBorders, VectorLikeFunction, WindowBorderProducer
from data_processing.arc_paths import INPUT_DATA_FOLDER

def load_neutron_window(file_name_prefix: str) -> WindowBorders:
    side_borders_path, bottom_border_path, top_border_path = get_neutron_window_paths(
        file_name_prefix
    )
    left_border, right_border = _load_side_borders(side_borders_path)
    bottom_border = _load_pickled_border(bottom_border_path)
    top_border = _load_pickled_border(top_border_path)
    return WindowBorders(
        left=left_border, right=right_border, bottom=bottom_border, top=top_border
    )


def get_neutron_window_paths(file_name_prefix: str) -> tuple[Path, Path, Path]:
    window_save_folder = INPUT_DATA_FOLDER / "ReferenceWindow"
    side_borders_path = window_save_folder / f"{file_name_prefix}_side_borders.txt"
    bottom_border_path = window_save_folder / f"{file_name_prefix}_bottom_border.pkl"
    top_border_path = window_save_folder / f"{file_name_prefix}_top_border.pkl"
    return side_borders_path, bottom_border_path, top_border_path


def _load_side_borders(
    side_borders_path: Path
) -> tuple[float|None, float|None]:
    with side_borders_path.open("r") as side_file:
        side_file_lines = side_file.readlines()
    if len(side_file_lines) < 2:
        raise ValueError("Side borders path did not lines for left and right borders")
    left_line, right_line, *_ = side_file_lines

    left_match = re.match(r"left: (.+)")
    if left_match is None:
        raise ValueError("Left line does not match accepted format")
    left_value_str = left_match.group(1)
    left_value = None if left_value_str == "None" else float(left_value_str)

    right_match = re.match(r"right: (.+)")
    if right_match is None:
        raise ValueError("Right line does not match accepted format")
    right_value_str = right_match.group(1)
    right_value = None if right_value_str == "None" else float(right_value_str)

    return left_value, right_value

def _load_pickled_border(border_path: Path) -> VectorLikeFunction:
    with border_path.open("rb") as border_file:
        border_fn = pkl.load(border_file)
    return border_fn
