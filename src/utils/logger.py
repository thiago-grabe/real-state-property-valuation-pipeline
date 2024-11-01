import logging
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class CustomFormatter(logging.Formatter):
    """
    CustomFormatter is a logging formatter that applies color coding to log messages based on their severity level.

    Attributes:
        grey (str): ANSI escape code for grey color.
        blue (str): ANSI escape code for blue color.
        yellow (str): ANSI escape code for yellow color.
        red (str): ANSI escape code for red color.
        bold_red (str): ANSI escape code for bold red color.
        reset (str): ANSI escape code to reset color.
        format (str): Log message format string.
        FORMATS (dict): Mapping of log levels to their corresponding color-coded formats.
    """

    grey = "\x1b[38;21m"
    blue = "\x1b[34;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        """
        Formats a log record with the appropriate color coding.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            str: The formatted log message.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    

def get_logger(name: str = "Real-State-ML", level: str = "INFO") -> logging.Logger:

    """
    Gets a logger with the specified name and level.

    Args:
        name: The name of the logger. Defaults to "Real-State-ML".
        level: The logging level. Defaults to "INFO".

    Returns:
        A logger with the specified name and level, configured to log to the console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    logger.handlers = []

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())

    fh = logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "app.log"), mode="a")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"))

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


class EndpointFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter out log records that contain "/health" in their message.
        
        Used to avoid logging of healthcheck endpoint calls.
        """
        return record.getMessage().find("/health") == -1