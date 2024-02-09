import datetime
from pathlib import Path

from matplotlib.figure import Figure


def generate_report(
    num_initial_signals: int,
    num_missing_signals: int,
    num_multipeaks: int,
    num_incomplete: int,
    num_low_snr: int,
    final_size: int,
) -> dict:
    report = {
        "last_updated": datetime.datetime.now(),
        "initial": num_initial_signals,
        "missing_or_nan": num_missing_signals,
        "multipeaks": num_multipeaks,
        "incomplete": num_incomplete,
        "low_snr": num_low_snr,
        "final": final_size,
    }
    return report


def save_plot(destination: Path, fig: Figure, label: str | Path, 
              extension: str | None = None) -> None:
    destination.mkdir(exist_ok=True)
    save_path = destination / label
    if extension is not None:
        save_path = Path(str(save_path) + extension)

    fig.savefig(str(save_path))
