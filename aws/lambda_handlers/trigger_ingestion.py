
import os
from datetime import datetime, date, timezone

import boto3

from botocore.exceptions import ClientError

from src.ingestion.batch_manager import BatchContext
from src.ingestion.ingestion_service import run_ingestion
from src.utils.setup_logger import logger


AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

IDEMPOTENCY_TABLE_NAME = os.getenv(
    "IDEMPOTENCY_TABLE_NAME",
    "DataForge-Ingestion-Idempotency",
)

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
idempotency_table = dynamodb.Table(IDEMPOTENCY_TABLE_NAME)


def update_batch_status(
    upload_batch_id: str,
    execution_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    """Update batch status only when this execution owns the record."""

    if status not in ("COMPLETED", "FAILED"):
        raise ValueError(f"Unsupported batch status: {status}")

    expression_values = {
        ":status": status,
        ":processing": "PROCESSING",
        ":execution_id": execution_id,
        ":timestamp": datetime.now(timezone.utc).isoformat(),
    }

    update_expression = (
        "SET #status = :status, updated_at = :timestamp"
    )

    if status == "FAILED":
        update_expression += ", error_message = :error_message"
        expression_values[":error_message"] = (
            (error_message or "Unknown ingestion error")[:1000]
        )
    else:
        update_expression += " REMOVE error_message"

    try:
        idempotency_table.update_item(
            Key={"upload_batch_id": upload_batch_id},
            UpdateExpression=update_expression,
            ConditionExpression=(
                "#status = :processing "
                "AND execution_id = :execution_id"
            ),
            ExpressionAttributeNames={
                "#status": "status",
            },
            ExpressionAttributeValues=expression_values,
        )

        logger.info(
            "DynamoDB batch status updated: %s -> %s",
            upload_batch_id,
            status,
        )

    except ClientError:
        logger.exception(
            "Unable to update DynamoDB status for upload batch: %s",
            upload_batch_id,
        )
        raise


def lambda_handler(event, context):

    logger.info("DataForge Trigger Ingestion Started")

    upload_batch_id = None
    execution_id = None

    try:

        if not isinstance(event, dict):
            raise ValueError(
                "Invalid Lambda event. Expected a dictionary."
            )

        # Retrieve batch context from InitializeBatch.
        batch_data = event.get("batch")

        if not isinstance(batch_data, dict):
            raise ValueError(
                "Missing or invalid batch context."
            )

        required_fields = [
            "batch_id",
            "ingestion_timestamp",
            "ingestion_date",
        ]

        missing_fields = [
            field
            for field in required_fields
            if not batch_data.get(field)
        ]

        if missing_fields:
            raise ValueError(
                f"Missing batch fields: {', '.join(missing_fields)}"
            )

        # Validate ingestion timestamp.
        try:
            ingestion_timestamp = datetime.fromisoformat(
                batch_data["ingestion_timestamp"]
            )
        except (TypeError, ValueError):
            raise ValueError(
                "Invalid ingestion_timestamp. "
                "Expected ISO 8601 format."
            )

        if ingestion_timestamp.tzinfo is None:
            raise ValueError(
                "ingestion_timestamp must include timezone information."
            )

        # Validate ingestion date.
        try:
            date.fromisoformat(batch_data["ingestion_date"])
        except (TypeError, ValueError):
            raise ValueError(
                "Invalid ingestion_date. Expected YYYY-MM-DD format."
            )

        # Retrieve Landing upload information.
        upload_batch_id = event.get("upload_batch_id")
        source_prefix = event.get("source_prefix")

        if not upload_batch_id or not source_prefix:
            raise ValueError(
                "Missing upload_batch_id or source_prefix."
            )

        expected_prefix = (
            f"landing/incoming/"
            f"upload_batch_id={upload_batch_id}/"
        )

        if source_prefix != expected_prefix:
            raise ValueError(
                "Source prefix does not match upload batch ID."
            )

        # Use the initialization Lambda's execution ID.
        # This must match the execution_id stored in DynamoDB.
        execution_id = event.get("execution_id")

        if not execution_id:
            raise ValueError(
                "Missing initialization execution_id."
            )

        # Reconstruct BatchContext.
        batch = BatchContext(
            batch_id=batch_data["batch_id"],
            ingestion_timestamp=ingestion_timestamp,
            ingestion_date=batch_data["ingestion_date"],
        )

        logger.info("Ingestion Batch ID: %s", batch.batch_id)
        logger.info("Upload Batch ID: %s", upload_batch_id)
        logger.info("Source Prefix: %s", source_prefix)
        logger.info("Initialization Execution ID: %s", execution_id)

        # Execute ingestion for this specific upload batch.
        ingestion_summary = run_ingestion(
            batch_context=batch,
            source_prefix=source_prefix,
        )

        # Mark the upload batch completed only after ingestion succeeds.
        update_batch_status(
            upload_batch_id=upload_batch_id,
            execution_id=execution_id,
            status="COMPLETED",
        )

        response = {
            "status": "SUCCESS",
            "batch_id": batch.batch_id,
            "upload_batch_id": upload_batch_id,
            "ingestion_summary": ingestion_summary,
            "execution_id": execution_id,
            "dynamodb_status": "COMPLETED",
        }

        logger.info(
            "DataForge ingestion completed successfully: %s",
            batch.batch_id,
        )

        return response

    except Exception as error:

        logger.exception(
            "DataForge Trigger Ingestion failed."
        )

        # Mark the batch failed only if we have its ownership details.
        if upload_batch_id and execution_id:
            try:
                update_batch_status(
                    upload_batch_id=upload_batch_id,
                    execution_id=execution_id,
                    status="FAILED",
                    error_message=str(error),
                )

            except Exception:
                logger.exception(
                    "Failed to update batch failure status in DynamoDB."
                )

        raise
