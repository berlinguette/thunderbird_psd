import logging


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    basic_format = "%(asctime)s - %(levelname)s/%(name)s - %(message)s"
    detailed_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: basic_format,
        logging.INFO: basic_format,
        logging.WARNING: detailed_format,
        logging.ERROR: detailed_format,
        logging.CRITICAL: detailed_format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
