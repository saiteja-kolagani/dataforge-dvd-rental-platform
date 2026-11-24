
import os

from pathlib import Path

from src.utils.setup_logger import logger


PROJECT_ROOT_DIRECTORY = Path(__file__).parent.parent.parent


def discover_csv_files(
    data_path: str | None = None
) -> list[Path]:

    logger.info("Started CSV file discovery.")
    logger.info(f"Initial data path: {data_path}")

    try:

        # Get data path from environment if not provided
        if data_path is None:
            data_path = os.getenv("RAW_DATA_PATH")

        if not data_path:
            raise ValueError(
                "RAW_DATA_PATH is not configured."
            )

        # Resolve local data directory
        raw_path = Path(data_path).expanduser()

        if not raw_path.is_absolute():
            raw_path = PROJECT_ROOT_DIRECTORY / raw_path

        raw_path = raw_path.resolve()

        logger.info(
            f"Resolved raw data directory: {raw_path}"
        )

        # Validate directory
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw data directory does not exist: {raw_path}"
            )

        if not raw_path.is_dir():
            raise NotADirectoryError(
                f"Raw data path is not a directory: {raw_path}"
            )

        # Discover CSV files
        csv_files = sorted(
            path
            for path in raw_path.glob("*.csv")
            if path.is_file()
        )

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {raw_path}"
            )

        logger.info(
            f"Successfully discovered {len(csv_files)} CSV files."
        )

        for file in csv_files:
            logger.info(f"Discovered file: {file.name}")

        return csv_files

    except Exception:
        logger.exception(
            "CSV file discovery failed."
        )
        raise
