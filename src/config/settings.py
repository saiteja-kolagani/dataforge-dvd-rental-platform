
import os

from dotenv import load_dotenv

from src.utils.setup_logger import logger


# Load local environment variables before reading configuration
load_dotenv()


# Environment Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "LOCAL").upper()
INGESTION_SOURCE = os.getenv("INGESTION_SOURCE", "LOCAL").upper()


# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION")
S3_BRONZE_BUCKET = os.getenv("S3_BRONZE_BUCKET")

S3_LANDING_BUCKET = os.getenv("S3_LANDING_BUCKET")
S3_LANDING_PREFIX = os.getenv(
    "S3_LANDING_PREFIX",
    "landing/incoming/"
)


# Local Configuration
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")


# Snowflake Configuration
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")


def validate_settings() -> None:
    """
    Validate configuration required for the ingestion process.
    Required settings depend on the configured ingestion source.
    """

    logger.info("Settings Validation Started...")

    if INGESTION_SOURCE not in ("LOCAL", "S3"):
        raise ValueError(
            f"Unsupported INGESTION_SOURCE: {INGESTION_SOURCE}. "
            "Supported values are LOCAL and S3."
        )

    required_settings = {
        "AWS_REGION": AWS_REGION,
        "S3_BRONZE_BUCKET": S3_BRONZE_BUCKET,
        "SNOWFLAKE_ACCOUNT": SNOWFLAKE_ACCOUNT,
        "SNOWFLAKE_USER": SNOWFLAKE_USER,
        "SNOWFLAKE_PASSWORD": SNOWFLAKE_PASSWORD,
        "SNOWFLAKE_WAREHOUSE": SNOWFLAKE_WAREHOUSE,
        "SNOWFLAKE_DATABASE": SNOWFLAKE_DATABASE,
        "SNOWFLAKE_SCHEMA": SNOWFLAKE_SCHEMA,
        "SNOWFLAKE_ROLE": SNOWFLAKE_ROLE,
    }

    if INGESTION_SOURCE == "LOCAL":
        required_settings["RAW_DATA_PATH"] = RAW_DATA_PATH

    elif INGESTION_SOURCE == "S3":
        required_settings["S3_LANDING_BUCKET"] = S3_LANDING_BUCKET

    missing_settings = [
        name
        for name, value in required_settings.items()
        if not value
    ]

    if missing_settings:
        error_message = (
            f"Missing required settings: {', '.join(missing_settings)}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    logger.info("No missing required settings.")
    logger.info("Settings Validation completed successfully.")
