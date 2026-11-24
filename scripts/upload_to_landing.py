
"""
DataForge - Upload local CSV files to S3 Landing.

Workflow:
1. Discover CSV files in the local source directory.
2. Generate a unique upload batch ID.
3. Upload files to landing/incoming/upload_batch_id=<ID>/
4. Create landing/ready/<ID>.json only after all uploads succeed.

Usage:
    python scripts/upload_to_landing.py
    python scripts/upload_to_landing.py --profile dataforge-dev
    python scripts/upload_to_landing.py --source-dir data/raw
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from dotenv import load_dotenv


# Load environment variables from the project-root .env file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_upload_batch_id():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_dvdrental"


def discover_csv_files(source_dir: Path):
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    files = sorted(
        file for file in source_dir.iterdir()
        if file.is_file() and file.suffix.lower() == ".csv"
    )

    if not files:
        raise RuntimeError(f"No CSV files found in: {source_dir}")

    return files


def create_s3_client(profile_name=None, region_name=None):
    session = boto3.Session(
        profile_name=profile_name,
        region_name=region_name,
    )
    return session.client("s3")


def upload_files(s3_client, files, bucket, incoming_prefix, batch_id):
    batch_prefix = (
        f"{incoming_prefix.rstrip('/')}/"
        f"upload_batch_id={batch_id}/"
    )

    uploaded_files = []

    for file_path in files:
        s3_key = f"{batch_prefix}{file_path.name}"

        logging.info("Uploading %s", file_path.name)

        s3_client.upload_file(
            str(file_path),
            bucket,
            s3_key,
            ExtraArgs={"ContentType": "text/csv"},
        )

        uploaded_files.append({
            "file_name": file_path.name,
            "s3_key": s3_key,
            "file_size_bytes": file_path.stat().st_size,
        })

        logging.info("Uploaded successfully: s3://%s/%s", bucket, s3_key)

    return batch_prefix, uploaded_files


def create_ready_marker(
    s3_client,
    bucket,
    ready_prefix,
    batch_id,
    batch_prefix,
    uploaded_files,
):
    marker_key = f"{ready_prefix.rstrip('/')}/{batch_id}.json"

    marker = {
        "project": "DataForge",
        "upload_batch_id": batch_id,
        "status": "READY",
        "source": "LOCAL_PYTHON_UPLOADER",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "incoming_prefix": batch_prefix,
        "file_count": len(uploaded_files),
        "files": uploaded_files,
    }

    marker_body = json.dumps(marker, indent=4)

    s3_client.put_object(
        Bucket=bucket,
        Key=marker_key,
        Body=marker_body.encode("utf-8"),
        ContentType="application/json",
    )

    logging.info("Ready marker created: s3://%s/%s", bucket, marker_key)


def main():
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Upload DataForge CSV files to S3 Landing."
    )
    parser.add_argument(
        "--source-dir",
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Local directory containing CSV files.",
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("AWS_PROFILE"),
        help="AWS CLI profile name. Defaults to AWS_PROFILE.",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "ap-south-1"),
        help="AWS region.",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_LANDING_BUCKET"),
        help="S3 Landing bucket name.",
    )
    parser.add_argument(
        "--incoming-prefix",
        default="landing/incoming",
        help="S3 prefix for incoming upload batches.",
    )
    parser.add_argument(
        "--ready-prefix",
        default="landing/ready",
        help="S3 prefix for ready markers.",
    )

    args = parser.parse_args()

    if not args.bucket:
        parser.error(
            "S3 Landing bucket is missing. Set S3_LANDING_BUCKET in .env "
            "or provide --bucket."
        )

    source_dir = Path(args.source_dir).resolve()

    logging.info("DataForge Landing Upload Started")
    logging.info("Source directory: %s", source_dir)
    logging.info("S3 bucket: %s", args.bucket)
    logging.info("AWS region: %s", args.region)
    logging.info("AWS profile: %s", args.profile or "default credentials")

    try:
        files = discover_csv_files(source_dir)
        batch_id = generate_upload_batch_id()

        logging.info("Discovered %d CSV files", len(files))
        logging.info("Upload batch ID: %s", batch_id)

        s3_client = create_s3_client(
            profile_name=args.profile,
            region_name=args.region,
        )

        batch_prefix, uploaded_files = upload_files(
            s3_client=s3_client,
            files=files,
            bucket=args.bucket,
            incoming_prefix=args.incoming_prefix,
            batch_id=batch_id,
        )

        # This is deliberately the final S3 operation.
        # No marker is created if any file upload fails.
        create_ready_marker(
            s3_client=s3_client,
            bucket=args.bucket,
            ready_prefix=args.ready_prefix,
            batch_id=batch_id,
            batch_prefix=batch_prefix,
            uploaded_files=uploaded_files,
        )

        logging.info("Landing upload completed successfully")
        logging.info("Batch ID: %s", batch_id)
        logging.info(
            "Uploaded files: %d/%d",
            len(uploaded_files),
            len(files),
        )
        logging.info(
            "Incoming location: s3://%s/%s",
            args.bucket,
            batch_prefix,
        )
        logging.info(
            "Ready marker: s3://%s/%s/%s.json",
            args.bucket,
            args.ready_prefix.rstrip("/"),
            batch_id,
        )

        return 0

    except Exception:
        logging.exception("Landing upload failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
