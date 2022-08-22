import logging
from tqdm import tqdm
from pathlib import Path

from custom_formatter import CustomFormatter


def setup_logger(logger_name: str, experiment_folder: Path) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(experiment_folder / 'conversion.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)
    return logger


def cleanup_logger(logger: logging.Logger):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()


def tqdm_logging(message: str, level: int, logger: logging.Logger, on_screen: bool = True):
    logger.log(level, message)
    if on_screen:
        tqdm.write(message)


def tqdm_log_debug(message: str, logger: logging.Logger, on_screen: bool = True):
    tqdm_logging(message, logging.DEBUG, logger, on_screen=on_screen)


def tqdm_log_info(message: str, logger: logging.Logger, on_screen: bool = True):
    tqdm_logging(message, logging.INFO, logger, on_screen=on_screen)
