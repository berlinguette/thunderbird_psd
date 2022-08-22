import logging
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
