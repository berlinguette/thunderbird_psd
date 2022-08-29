import logging
from tqdm import tqdm
from pathlib import Path

from logging_helpers.custom_formatter import CustomFormatter


def setup_logger(logger: logging.Logger, experiment_folder: Path):
    logger.setLevel(logging.DEBUG)
    
    handler = logging.FileHandler(experiment_folder / 'conversion.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(CustomFormatter())
    
    logger.addHandler(handler)


def cleanup_logger(logger: logging.Logger):
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
    if in_log:
        logger.log(level, msg)
    if on_screen:
        tqdm.write(msg)


def message_debug(msg: str, logger: logging.Logger, **kwargs):
    message(msg, logging.DEBUG, logger, **kwargs)


def message_info(msg: str, logger: logging.Logger, **kwargs):
    message(msg, logging.INFO, logger, **kwargs)
