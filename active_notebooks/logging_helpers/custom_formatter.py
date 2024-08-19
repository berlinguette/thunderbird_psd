"""This module is responsible for creating custom logging formatters."""
import logging
from typing import Dict, Tuple, Union


class CustomFormatter(logging.Formatter):
    """Logging Formatter giving different format for more severe log levels"""

    basic_format = ("%(asctime)s - %(levelname)s/%(name)s - %(message)s",
                    "%Y-%m-%d %H:%M:%S")
    detailed_format = ("%(asctime)s - %(name)s - %(levelname)s - " +
                       "%(message)s (%(filename)s:%(lineno)d)",
                       None)

    FORMATS: Dict[int, Union[Tuple[str, str], Tuple[str, None]]] = {
        logging.DEBUG: basic_format,
        logging.INFO: basic_format,
        logging.WARNING: detailed_format,
        logging.ERROR: detailed_format,
        logging.CRITICAL: detailed_format
    }

    def format(self, record: logging.LogRecord) -> str:
        """Formats a given record according to its severity level

        :param record: Record to format
        :type record: LogRecord
        :return: Formatted log record text
        :rtype: str
        """
        log_fmt = self.FORMATS.get(record.levelno, self.detailed_format)
        formatter = logging.Formatter(*log_fmt)
        return formatter.format(record)
