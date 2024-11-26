import logging


def setup_logging():
    """
    Set up basic logging configuration.
    """
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def log_error(message):
    """
    Log errors to the log file.
    """
    logging.error(message)


def log_info(message):
    """
    Log informational messages to the log file.
    """
    logging.info(message)
