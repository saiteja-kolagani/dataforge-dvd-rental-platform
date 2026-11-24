
from src.utils.setup_logger import logger

from src.config.settings import (
    AWS_REGION,
    S3_BRONZE_BUCKET,
    INGESTION_SOURCE,
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_SCHEMA,
    validate_settings,
)

from src.ingestion.batch_manager import (
    BatchContext,
    create_batch_context,
)

from src.ingestion.source_factory import create_source
from src.utils.s3_client import S3Client
from src.utils.snowflake_client import SnowflakeClient


METADATA_TABLE = (
    f"{SNOWFLAKE_DATABASE}."
    f"{SNOWFLAKE_SCHEMA}."
    "DVD_RENTAL_INGESTION_METADATA"
)


def build_s3_key(
    file,
    ingestion_date: str,
    batch_id: str
) -> str:

    return (
        f"bronze/{file.table_name}/"
        f"ingestion_date={ingestion_date}/"
        f"batch_id={batch_id}/"
        f"{file.file_name}"
    )


def insert_ingestion_metadata(
    snowflake_client: SnowflakeClient,
    batch: BatchContext,
    file,
    s3_key: str,
    status: str,
    error_message: str | None = None,
) -> None:

    query = f"""
        INSERT INTO {METADATA_TABLE} (
            BATCH_ID,
            INGESTION_TIMESTAMP,
            INGESTION_DATE,
            SOURCE_SYSTEM,
            SOURCE_FILE,
            SOURCE_TABLE,
            S3_BUCKET,
            S3_KEY,
            FILE_SIZE_BYTES,
            STATUS,
            ERROR_MESSAGE
        )
        VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
    """

    params = (
        batch.batch_id,
        batch.ingestion_timestamp,
        batch.ingestion_date,
        "DVD_RENTAL",
        file.file_name,
        file.table_name,
        S3_BRONZE_BUCKET,
        s3_key,
        file.file_size_bytes,
        status,
        error_message[:5000] if error_message else None,
    )

    snowflake_client.execute_query(
        query=query,
        params=params,
    )

    logger.info(
        f"{status} metadata recorded in Snowflake "
        f"for {file.file_name}"
    )


def run_ingestion(
    batch_context: BatchContext | None = None,
    source_prefix: str | None = None,
) -> dict:

    validate_settings()

    batch = batch_context or create_batch_context()

    logger.info("Starting DVD Rental ingestion...")
    logger.info(f"Batch ID: {batch.batch_id}")
    logger.info(
        f"Batch Ingestion Timestamp: "
        f"{batch.ingestion_timestamp.isoformat()}"
    )
    logger.info(f"Ingestion Date: {batch.ingestion_date}")
    logger.info(f"Ingestion Source: {INGESTION_SOURCE}")

    if INGESTION_SOURCE == "S3":
        logger.info(
            f"Landing source prefix: {source_prefix}"
        )

    # Discover only files belonging to the requested upload batch.
    source = create_source(
        source_prefix=source_prefix
    )

    files = source.discover_files()

    if not files:
        raise FileNotFoundError(
            "No CSV files found in the configured source."
        )

    total_files = len(files)

    logger.info(f"Discovered {total_files} CSV files")

    successful_files = 0
    failed_files = 0
    failed_file_details = []

    snowflake_client = None

    try:

        s3_client = S3Client(
            region_name=AWS_REGION
        )

        # Establish one Snowflake connection for this batch.
        snowflake_client = SnowflakeClient()
        snowflake_client.connect()

        for file in files:

            s3_key = build_s3_key(
                file=file,
                ingestion_date=batch.ingestion_date,
                batch_id=batch.batch_id,
            )

            logger.info(
                f"Processing file: {file.file_name}"
            )

            try:

                if INGESTION_SOURCE == "LOCAL":

                    s3_client.upload_file_to_s3(
                        local_path=file.local_path,
                        bucket=S3_BRONZE_BUCKET,
                        s3_key=s3_key,
                    )

                elif INGESTION_SOURCE == "S3":

                    s3_client.copy_s3_object(
                        source_bucket=file.source_bucket,
                        source_key=file.source_key,
                        destination_bucket=S3_BRONZE_BUCKET,
                        destination_key=s3_key,
                    )

                else:

                    raise ValueError(
                        f"Unsupported ingestion source: "
                        f"{INGESTION_SOURCE}"
                    )

                # Record successful ingestion in Snowflake.
                insert_ingestion_metadata(
                    snowflake_client=snowflake_client,
                    batch=batch,
                    file=file,
                    s3_key=s3_key,
                    status="SUCCESS",
                )

                successful_files += 1

                logger.info(
                    f"Successfully processed {file.file_name}"
                )

            except Exception as exc:

                failed_files += 1

                failed_file_details.append({
                    "file_name": file.file_name,
                    "error": str(exc),
                    "s3_key": s3_key,
                })

                logger.exception(
                    f"Failed to process {file.file_name}"
                )

                try:

                    # Record failed ingestion in Snowflake.
                    insert_ingestion_metadata(
                        snowflake_client=snowflake_client,
                        batch=batch,
                        file=file,
                        s3_key=s3_key,
                        status="FAILED",
                        error_message=str(exc),
                    )

                except Exception:

                    logger.exception(
                        f"Unable to record FAILED metadata "
                        f"for {file.file_name}"
                    )

    finally:

        if snowflake_client is not None:

            try:
                snowflake_client.close()

            except Exception:
                logger.exception(
                    "Failed to close Snowflake connection."
                )

        logger.info(
            f"Batch processing summary: "
            f"Successful={successful_files}, "
            f"Failed={failed_files}, "
            f"Total={total_files}"
        )

    if failed_files > 0:

        raise RuntimeError(
            f"Ingestion failed. "
            f"Successful: {successful_files}/{total_files}. "
            f"Failed: {failed_files}/{total_files}. "
            f"Batch ID: {batch.batch_id}. "
            f"Failed files: {failed_file_details}"
        )

    logger.info(
        f"Ingestion completed successfully. "
        f"Batch ID: {batch.batch_id}"
    )

    return {
        "batch_id": batch.batch_id,
        "total_files": total_files,
        "successful_files": successful_files,
        "failed_files": failed_files,
        "status": "SUCCESS",
    }
