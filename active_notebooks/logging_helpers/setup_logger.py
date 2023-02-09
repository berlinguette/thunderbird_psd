import logging
from pathlib import Path

from tqdm import tqdm

from logging_helpers.custom_formatter import CustomFormatter


def setup_logger(logger: logging.Logger, log_file: Path):
    """Sets up logger:
    - sets level to Debug
    - applies CustomFormatter
    - sets logger to log to given log file

    Parameters
    ----------
    logger : logging.Logger
        logger to set up
    log_file : Path
        path to log file
    """
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(CustomFormatter())

    cleanup_logger(logger)
    logger.addHandler(handler)
    

def get_conversion_logfile_path(experiment_folder: Path) -> Path:
    """Gives correct conversion log file path for given experiment folder

    Parameters
    ----------
    experiment_folder : Path
        path to root folder for this experiment

    Returns
    -------
    Path
        path to experiment's conversion log file
    """
    log_filename = 'conversion.log'
    return experiment_folder / log_filename


def cleanup_logger(logger: logging.Logger):
    """Cleans up file loggers, releasing all log file resources

    Parameters
    ----------
    logger : logging.Logger
        logger to clean up if needed
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


class Messenger:
    """Handles sending log messages that are compatible with tqdm

    Parameters
    ----------
    logger : logging.Logger
        logger used when sending log messages
    in_log : bool, optional
        whether messages should be saved to log file, by default True
    on_screen : bool, optional
        whether messages should be displayed in console, by default True
    """

    def __init__(
        self,
        logger: logging.Logger,
        in_log: bool = True,
        on_screen: bool = True
    ):
        self.logger = logger
        self.in_log = in_log
        self.on_screen = on_screen

    def message(self, msg: str, level: int):
        """Send a message that can be logged and/or displayed on screen

        These messages are compatible with tqdm. 

        Parameters
        ----------
        msg : str
            message to log/display
        level : int
            desired log level
        """
        if self.in_log:
            self.logger.log(level, msg)
        if self.on_screen:
            tqdm.write(msg)

    def debug(self, msg: str):
        """Send a message that logs at the debug log level

        Parameters
        ----------
        msg : str
            message to log/display
        """
        self.message(msg, logging.DEBUG)

    def info(self, msg: str):
        """Send a message that logs at the debug log level

        Parameters
        ----------
        msg : str
            message to log/display
        """
        self.message(msg, logging.INFO)
