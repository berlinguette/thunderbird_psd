import logging
from pathlib import Path

from tqdm import tqdm

from logging_helpers.custom_formatter import CustomFormatter


def setup_logger(logger: logging.Logger, log_file: Path):
    """Sets up logger:
    - sets level to Debug
    - applies CustomFormatter
    - sets logger to log to given log file

    :param logger: Logger to set up
    :type logger: logging.Logger
    :param log_file: Path to log file
    :type log_file: Path
    """
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(CustomFormatter())

    cleanup_logger(logger)
    logger.addHandler(handler)
    

def get_conversion_logfile_path(experiment_folder: Path) -> Path:
    """Gives correct conversion log file path for given experiment folder

    :param experiment_folder: path to root folder for this experiment
    :type experiment_folder: Path
    :return: path to experiment's conversion log file
    :rtype: Path
    """
    log_filename = 'conversion.log'
    return experiment_folder / log_filename


def cleanup_logger(logger: logging.Logger):
    """Cleans up file loggers, releasing all log file resources

    :param logger: logger to clean up if needed
    :type logger: logging.Logger
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


class Messenger:
    """Handles sending log messages that are compatible with tqdm

    :param logger: logger used when sending log messages
    :type logger: logging.Logger
    :param in_log: whether messages should be saved to log file, by default True, defaults to True
    :type in_log: bool, optional
    :param on_screen: whether messages should be displayed in console, by default True, defaults to True
    :type on_screen: bool, optional
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

        :param msg: message to log/display
        :type msg: str
        :param level: desired log level
        :type level: int
        """
        if self.in_log:
            self.logger.log(level, msg)
        if self.on_screen:
            tqdm.write(msg)

    def debug(self, msg: str):
        """Send a message that logs at the debug log level

        :param msg: message to log/display
        :type msg: str
        """
        self.message(msg, logging.DEBUG)

    def info(self, msg: str):
        """Send a message that logs at the debug log level

        :param msg: message to log/display
        :type msg: str
        """
        self.message(msg, logging.INFO)
