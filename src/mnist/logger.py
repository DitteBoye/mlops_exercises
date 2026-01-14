from loguru import logger
import sys

def setup_logger(
    log_file: str = "train.log",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
):
    logger.remove()

    # Terminal logger
    logger.add(
        sys.stdout,
        level=console_level,
        format="{time} | {level} | {message}",
    )

    # File logger (ALT gemmes her)
    logger.add(
        log_file,
        level=file_level,
        rotation="100 MB",
        format="{time} | {level} | {message}",
    )

    return logger