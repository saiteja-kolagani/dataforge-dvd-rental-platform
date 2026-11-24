
import os
import sys
import logging

from dotenv import load_dotenv
from pathlib import Path


# Load environment variables for local development
load_dotenv()

PROJECT_ROOT_DIRECTORY = Path(__file__).parent.parent.parent

ENVIRONMENT = os.getenv("ENVIRONMENT", "LOCAL").upper()


def setup_logger_handler(
    file_path: str | None = None,
    level=logging.INFO
) -> logging.Logger:

    logger = logging.getLogger("dataforge")
    logger.setLevel(level)

    # Prevent duplicate log propagation
    logger.propagate = False

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(funcName)s | %(levelname)s | %(message)s"
    )

    # Console logging: Local terminal and AWS CloudWatch
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File logging: Local environment only
    if ENVIRONMENT == "LOCAL":

        if file_path is None:
            file_path = os.getenv("LOG_FILE_PATH")

        if file_path:

            log_path = Path(file_path)

            if not log_path.is_absolute():
                log_path = PROJECT_ROOT_DIRECTORY / log_path

            log_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            file_handler = logging.FileHandler(
                log_path,
                encoding="utf-8"
            )

            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


logger = setup_logger_handler()
