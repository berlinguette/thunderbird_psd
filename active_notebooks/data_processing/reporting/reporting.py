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
    """[DEPRECATED]Generate signal analysis report

    :param num_initial_signals: Number of initial signals processed
    :type num_initial_signals: int
    :param num_missing_signals: Number of signals missing
    :type num_missing_signals: int
    :param num_multipeaks: Number of multipeak (i.e. pileup) signals
    :type num_multipeaks: int
    :param num_incomplete: Number of incomplete signals
    :type num_incomplete: int
    :param num_low_snr: Number of signals with low signal-noise ratio
    :type num_low_snr: int
    :param final_size: Final dataset size
    :type final_size: int
    :return: Dictionary of report data
    :rtype: dict
    """
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
    """Save plot as file.

    :param destination: Path for plot save folder
    :type destination: Path
    :param fig: Plot figure
    :type fig: Figure
    :param label: Plot file name (without suffix)
    :type label: str | Path
    :param extension: file extension, defaults to None
    :type extension: str | None, optional
    """
    destination.mkdir(exist_ok=True)
    save_path = destination / label
    if extension is not None:
        save_path = Path(str(save_path) + extension)

    fig.savefig(str(save_path))
