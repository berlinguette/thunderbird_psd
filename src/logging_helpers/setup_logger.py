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


def message(
    msg: str,
    level: int,
    logger: logging.Logger,
    on_screen: bool = True,
    in_log: bool = True
):
    """Send a message that can be logged and/or displayed on screen

    These messages are compatible with tqdm. 

    Parameters
    ----------
    msg : str
        message to log/display
    level : int
        desired log level
    logger : logging.Logger
        logger to use when logging this message
    on_screen : bool, optional
        whether to display the message on screen, by default True
    in_log : bool, optional
        whether to log the message, by default True
    """
    if in_log:
        logger.log(level, msg)
    if on_screen:
        tqdm.write(msg)


def message_debug(msg: str, logger: logging.Logger, **kwargs):
    """Send a message that logs at the debug log level

    See message() for info on possible keyword arguments

    Parameters
    ----------
    msg : str
        message to log/display
    logger : logging.Logger
        logger to use when logging this message
    """
    message(msg, logging.DEBUG, logger, **kwargs)


def message_info(msg: str, logger: logging.Logger, **kwargs):
    """Send a message that logs at the info log level

    See message() for info on possible keyword arguments

    Parameters
    ----------
    msg : str
        message to log/display
    logger : logging.Logger
        logger to use when logging this message
    """
    message(msg, logging.INFO, logger, **kwargs)
