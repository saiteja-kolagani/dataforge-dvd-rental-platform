
from src.utils.setup_logger import logger

from src.config.settings import (
    INGESTION_SOURCE,
    RAW_DATA_PATH,
    AWS_REGION,
    S3_LANDING_BUCKET,
)

from src.ingestion.local_source import LocalSource
from src.ingestion.s3_source import S3Source
from src.utils.s3_client import S3Client


def create_source(source_prefix: str | None = None):

    logger.info(
        f"Initializing ingestion source: {INGESTION_SOURCE}"
    )

    if INGESTION_SOURCE == "LOCAL":

        return LocalSource(
            data_path=RAW_DATA_PATH
        )

    elif INGESTION_SOURCE == "S3":

        if not source_prefix:
            raise ValueError(
                "A batch-specific S3 source prefix is required "
                "for S3 ingestion."
            )

        if not source_prefix.startswith(
            "landing/incoming/upload_batch_id="
        ):
            raise ValueError(
                "Invalid Landing source prefix."
            )

        if not source_prefix.endswith("/"):
            raise ValueError(
                "S3 source prefix must end with '/'."
            )

        s3_client = S3Client(
            region_name=AWS_REGION
        )

        logger.info(
            f"Using batch-specific Landing prefix: {source_prefix}"
        )

        return S3Source(
            s3_client=s3_client,
            bucket=S3_LANDING_BUCKET,
            prefix=source_prefix,
        )

    else:

        raise ValueError(
            f"Unsupported ingestion source: {INGESTION_SOURCE}"
        )
