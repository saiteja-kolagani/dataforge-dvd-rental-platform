
import argparse
import json
import logging
import os
from datetime import datetime, timezone

import boto3
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("S3_LANDING_BUCKET")
REGION = os.getenv("AWS_REGION", "ap-south-1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--expected-files", type=int, default=17)
    parser.add_argument("--profile", default=os.getenv("AWS_PROFILE"))
    args = parser.parse_args()

    if not BUCKET:
        raise ValueError("S3_LANDING_BUCKET is missing from .env")

    batch_id = args.batch_id
    incoming_prefix = (
        f"landing/incoming/upload_batch_id={batch_id}/"
    )

    session = boto3.Session(
        profile_name=args.profile,
        region_name=REGION,
    )
    s3 = session.client("s3")

    paginator = s3.get_paginator("list_objects_v2")

    files = []

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=incoming_prefix,
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.lower().endswith(".csv"):
                files.append({
                    "file_name": key.rsplit("/", 1)[-1],
                    "s3_key": key,
                    "file_size_bytes": obj["Size"],
                })

    if len(files) != args.expected_files:
        raise RuntimeError(
            f"Expected {args.expected_files} CSV files, "
            f"but found {len(files)}. Marker not updated."
        )

    marker = {
        "project": "DataForge",
        "upload_batch_id": batch_id,
        "status": "READY",
        "source": "AWS_CONSOLE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket": BUCKET,
        "incoming_prefix": incoming_prefix,
        "file_count": len(files),
        "files": sorted(files, key=lambda item: item["file_name"]),
    }

    marker_key = f"landing/ready/{batch_id}.json"

    s3.put_object(
        Bucket=BUCKET,
        Key=marker_key,
        Body=json.dumps(marker, indent=4).encode("utf-8"),
        ContentType="application/json",
    )

    logging.info("Ready marker updated successfully")
    logging.info("Batch ID: %s", batch_id)
    logging.info("Files verified: %d", len(files))
    logging.info("Marker: s3://%s/%s", BUCKET, marker_key)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
