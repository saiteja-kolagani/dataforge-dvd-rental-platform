
import os

from botocore.exceptions import ClientError

from src.ingestion.source_file import SourceFile
from src.utils.setup_logger import logger
from src.utils.s3_client import S3Client


class S3Source:

    def __init__(
        self,
        s3_client: S3Client,
        bucket: str,
        prefix: str,
    ):
        self.s3_client = s3_client
        self.bucket = bucket
        self.prefix = prefix

    def discover_files(self) -> list[SourceFile]:

        logger.info(
            f"Discovering CSV files from "
            f"s3://{self.bucket}/{self.prefix}"
        )

        paginator = self.s3_client.client.get_paginator(
            "list_objects_v2"
        )

        files = []

        try:

            for page in paginator.paginate(
                Bucket=self.bucket,
                Prefix=self.prefix,
            ):

                for obj in page.get("Contents", []):

                    key = obj["Key"]

                    # Ignore non-CSV objects
                    if not key.lower().endswith(".csv"):
                        continue

                    file_name = key.rsplit("/", 1)[-1]

                    # Ignore folder placeholders and empty files
                    if not file_name or obj["Size"] == 0:
                        continue

                    table_name = os.path.splitext(file_name)[0]

                    files.append(
                        SourceFile(
                            file_name=file_name,
                            table_name=table_name,
                            file_size_bytes=obj["Size"],
                            source_bucket=self.bucket,
                            source_key=key,
                        )
                    )

        except ClientError:
            logger.exception(
                f"Failed to discover files from "
                f"s3://{self.bucket}/{self.prefix}"
            )
            raise

        if not files:
            raise FileNotFoundError(
                f"No valid CSV files found in "
                f"s3://{self.bucket}/{self.prefix}"
            )

        files = sorted(
            files,
            key=lambda item: item.file_name
        )

        logger.info(
            f"Discovered {len(files)} CSV files in S3 Landing"
        )

        return files
