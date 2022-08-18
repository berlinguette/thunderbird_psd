import logging


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    debug_format = "%(levelname)s/%(name)s - %(message)s"
    info_format = "%(message)s"
    detailed_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: debug_format,
        logging.INFO: info_format,
        logging.WARNING: detailed_format,
        logging.ERROR: detailed_format,
        logging.CRITICAL: detailed_format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
