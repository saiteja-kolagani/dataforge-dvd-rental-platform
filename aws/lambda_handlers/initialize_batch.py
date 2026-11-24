
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

from src.config.settings import S3_LANDING_BUCKET
from src.ingestion.batch_manager import create_batch_context
from src.utils.setup_logger import logger


AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

IDEMPOTENCY_TABLE_NAME = os.getenv(
    "IDEMPOTENCY_TABLE_NAME",
    "DataForge-Ingestion-Idempotency",
)

s3_client = boto3.client("s3", region_name=AWS_REGION)

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
idempotency_table = dynamodb.Table(IDEMPOTENCY_TABLE_NAME)

READY_PREFIX = "landing/ready/"
INCOMING_PREFIX = "landing/incoming/"
EXPECTED_PROJECT = "DataForge"


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_ready_marker(event: dict) -> tuple[str, str, str]:
    """Extract and validate ready-marker event details."""

    if not isinstance(event, dict):
        raise ValueError("Expected an EventBridge event dictionary.")

    detail = event.get("detail", {})
    bucket = detail.get("bucket", {}).get("name")
    object_key = detail.get("object", {}).get("key")

    if not bucket or not object_key:
        raise ValueError("Missing S3 bucket or object key in event.")

    if bucket != S3_LANDING_BUCKET:
        raise ValueError("Event received from an unexpected S3 bucket.")

    object_key = unquote_plus(object_key)

    if not object_key.startswith(READY_PREFIX):
        raise ValueError("Event is not for a Landing ready marker.")

    marker_name = object_key.rsplit("/", 1)[-1]

    if not marker_name.endswith(".json"):
        raise ValueError("Ready marker must be a JSON file.")

    upload_batch_id = marker_name[:-5]

    if not re.fullmatch(r"[A-Za-z0-9_-]+", upload_batch_id):
        raise ValueError("Invalid upload batch ID.")

    source_prefix = (
        f"{INCOMING_PREFIX}upload_batch_id={upload_batch_id}/"
    )

    return upload_batch_id, source_prefix, object_key


def validate_ready_marker(
    marker_key: str,
    upload_batch_id: str,
    source_prefix: str,
) -> dict:
    """Validate the marker and all referenced S3 files."""

    logger.info("Reading ready marker: %s", marker_key)

    response = s3_client.get_object(
        Bucket=S3_LANDING_BUCKET,
        Key=marker_key,
    )

    marker_content = response["Body"].read().decode("utf-8")
    marker = json.loads(marker_content)

    if marker.get("project") != EXPECTED_PROJECT:
        raise ValueError("Invalid project name in ready marker.")

    if marker.get("status") != "READY":
        raise ValueError("Ready marker status is not READY.")

    if marker.get("upload_batch_id") != upload_batch_id:
        raise ValueError("Upload batch ID does not match marker filename.")

    if marker.get("bucket") != S3_LANDING_BUCKET:
        raise ValueError("Ready marker contains an unexpected bucket.")

    if marker.get("incoming_prefix") != source_prefix:
        raise ValueError("Incoming prefix does not match upload batch.")

    files = marker.get("files")

    if not isinstance(files, list) or not files:
        raise ValueError("Ready marker contains no valid file list.")

    file_count = marker.get("file_count")

    if (
        not isinstance(file_count, int)
        or isinstance(file_count, bool)
        or file_count != len(files)
    ):
        raise ValueError(
            "Ready marker file count does not match its file list."
        )

    expected_keys = set()

    for file_info in files:
        if not isinstance(file_info, dict):
            raise ValueError("Invalid file entry in ready marker.")

        file_name = file_info.get("file_name")
        s3_key = file_info.get("s3_key")
        expected_size = file_info.get("file_size_bytes")

        if not file_name or not s3_key:
            raise ValueError("File name or S3 key is missing.")

        if not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"Invalid file size for {file_name}.")

        if not file_name.lower().endswith(".csv"):
            raise ValueError(f"Non-CSV file found: {file_name}")

        if s3_key != f"{source_prefix}{file_name}":
            raise ValueError(f"Unexpected S3 key for file: {file_name}")

        if s3_key in expected_keys:
            raise ValueError(f"Duplicate file entry: {s3_key}")

        expected_keys.add(s3_key)

        head_response = s3_client.head_object(
            Bucket=S3_LANDING_BUCKET,
            Key=s3_key,
        )

        actual_size = head_response["ContentLength"]

        if actual_size != expected_size:
            raise ValueError(
                f"File size mismatch for {file_name}. "
                f"Marker: {expected_size}, S3: {actual_size}"
            )

    actual_keys = set()
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=S3_LANDING_BUCKET,
        Prefix=source_prefix,
    ):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".csv"):
                actual_keys.add(obj["Key"])

    if actual_keys != expected_keys:
        missing_files = expected_keys - actual_keys
        unexpected_files = actual_keys - expected_keys

        raise ValueError(
            "S3 files do not match the ready-marker manifest. "
            f"Missing: {sorted(missing_files)}. "
            f"Unexpected: {sorted(unexpected_files)}."
        )

    logger.info(
        "Ready marker validated successfully. Files: %s",
        file_count,
    )

    return marker


def claim_upload_batch(
    upload_batch_id: str,
    marker_key: str,
    file_count: int,
    source: str,
    execution_id: str,
) -> tuple[bool, str | None]:
    """
    Atomically claim an upload batch.

    New batches are inserted using a conditional PutItem.
    Previously failed batches can be claimed for a controlled retry.
    """

    timestamp = current_timestamp()

    item = {
        "upload_batch_id": upload_batch_id,
        "status": "PROCESSING",
        "created_at": timestamp,
        "updated_at": timestamp,
        "execution_id": execution_id,
        "marker_key": marker_key,
        "file_count": file_count,
        "source": source or "UNKNOWN",
    }

    try:
        idempotency_table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(upload_batch_id)",
        )

        logger.info(
            "Upload batch claimed successfully: %s",
            upload_batch_id,
        )

        return True, None

    except ClientError as error:
        error_code = error.response.get("Error", {}).get("Code")

        if error_code != "ConditionalCheckFailedException":
            raise

    existing_response = idempotency_table.get_item(
        Key={"upload_batch_id": upload_batch_id},
        ConsistentRead=True,
    )

    existing_item = existing_response.get("Item")

    if not existing_item:
        raise RuntimeError(
            "Batch record disappeared during duplicate-event handling."
        )

    existing_status = existing_item.get("status")

    if existing_status != "FAILED":
        logger.warning(
            "Duplicate upload batch detected: %s, existing status: %s",
            upload_batch_id,
            existing_status,
        )

        return False, existing_status

    # Allow a retry only when the existing status is FAILED.
    try:
        idempotency_table.update_item(
            Key={"upload_batch_id": upload_batch_id},
            UpdateExpression=(
                "SET #status = :processing, "
                "updated_at = :timestamp, "
                "execution_id = :execution_id, "
                "marker_key = :marker_key, "
                "file_count = :file_count, "
                "#source = :source"
            ),
            ConditionExpression="#status = :failed",
            ExpressionAttributeNames={
                "#status": "status",
                "#source": "source",
            },
            ExpressionAttributeValues={
                ":processing": "PROCESSING",
                ":failed": "FAILED",
                ":timestamp": timestamp,
                ":execution_id": execution_id,
                ":marker_key": marker_key,
                ":file_count": file_count,
                ":source": source or "UNKNOWN",
            },
        )

        logger.info(
            "Previously failed batch claimed for retry: %s",
            upload_batch_id,
        )

        return True, None

    except ClientError as error:
        error_code = error.response.get("Error", {}).get("Code")

        if error_code != "ConditionalCheckFailedException":
            raise

        latest_response = idempotency_table.get_item(
            Key={"upload_batch_id": upload_batch_id},
            ConsistentRead=True,
        )

        latest_item = latest_response.get("Item", {})

        logger.warning(
            "Batch retry claim was already taken: %s",
            upload_batch_id,
        )

        return False, latest_item.get("status", "UNKNOWN")


def mark_batch_failed(
    upload_batch_id: str,
    execution_id: str,
    error_message: str,
) -> None:
    """Mark an initialization failure in the idempotency table."""

    idempotency_table.update_item(
        Key={"upload_batch_id": upload_batch_id},
        UpdateExpression=(
            "SET #status = :failed, "
            "updated_at = :timestamp, "
            "error_message = :error_message"
        ),
        ConditionExpression=(
            "#status = :processing AND execution_id = :execution_id"
        ),
        ExpressionAttributeNames={
            "#status": "status",
        },
        ExpressionAttributeValues={
            ":failed": "FAILED",
            ":processing": "PROCESSING",
            ":timestamp": current_timestamp(),
            ":error_message": error_message[:1000],
            ":execution_id": execution_id,
        },
    )


def lambda_handler(event, context):
    logger.info("DataForge Batch Initialization Started")

    upload_batch_id = None
    execution_id = (
        getattr(context, "aws_request_id", None)
        if context is not None
        else "LOCAL-TEST"
    )
    batch_claimed = False

    try:
        upload_batch_id, source_prefix, marker_key = (
            extract_ready_marker(event)
        )

        marker = validate_ready_marker(
            marker_key=marker_key,
            upload_batch_id=upload_batch_id,
            source_prefix=source_prefix,
        )

        batch_claimed, existing_status = claim_upload_batch(
            upload_batch_id=upload_batch_id,
            marker_key=marker_key,
            file_count=marker["file_count"],
            source=marker.get("source"),
            execution_id=execution_id,
        )

        if not batch_claimed:
            return {
                "status": "DUPLICATE",
                "upload_batch_id": upload_batch_id,
                "existing_status": existing_status,
                "message": (
                    "This upload batch has already been claimed "
                    "or processed."
                ),
            }

        batch = create_batch_context()
        batch_data = asdict(batch)

        batch_data["ingestion_timestamp"] = (
            batch.ingestion_timestamp.isoformat()
        )

        response = {
            "status": "SUCCESS",
            "batch": batch_data,
            "upload_batch_id": upload_batch_id,
            "source_prefix": source_prefix,
            "marker_key": marker_key,
            "file_count": marker["file_count"],
            "source": marker.get("source"),
            "execution_id": execution_id,
        }

        logger.info(
            "Batch initialization completed. Batch ID: %s, "
            "Upload Batch ID: %s, Files: %s",
            batch.batch_id,
            upload_batch_id,
            marker["file_count"],
        )

        return response

    except Exception as error:
        logger.exception("DataForge Batch Initialization Failed")

        if batch_claimed and upload_batch_id:
            try:
                mark_batch_failed(
                    upload_batch_id=upload_batch_id,
                    execution_id=execution_id,
                    error_message=str(error),
                )
            except Exception:
                logger.exception(
                    "Failed to update DynamoDB batch status."
                )

        raise
