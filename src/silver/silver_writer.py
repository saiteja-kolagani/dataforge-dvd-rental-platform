
import re
import uuid
from datetime import date, datetime

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class SilverWriter:
    """
    Writes Silver and Quarantine DataFrames as partitioned Parquet.

    Features:
    - Batch-level idempotency.
    - Reprocessing replaces only the specified batch.
    - Other batch partitions remain untouched.
    - Staged writes before promotion.
    - Backup and rollback if promotion fails.
    - Supports local Hadoop-compatible filesystems.

    Important:
    This rename-based transaction strategy is intended for
    local filesystems. It is not an atomic transaction on Amazon S3.
    """

    def __init__(
        self,
        silver_base_path: str,
        quarantine_base_path: str,
    ):
        self.silver_base_path = silver_base_path.rstrip("/")
        self.quarantine_base_path = quarantine_base_path.rstrip("/")

    @staticmethod
    def _validate_path_segment(value: str, name: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid {name}: value cannot be empty.")

        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError(
                f"Invalid {name}: use only letters, numbers, "
                "underscores and hyphens."
            )

    @staticmethod
    def _normalize_ingestion_date(ingestion_date) -> str:
        """
        Normalize ingestion_date to YYYY-MM-DD.

        Supported inputs:
        - None
        - datetime.date
        - datetime.datetime
        - YYYY-MM-DD string
        """

        if ingestion_date is None:
            return date.today().isoformat()

        if isinstance(ingestion_date, datetime):
            return ingestion_date.date().isoformat()

        if isinstance(ingestion_date, date):
            return ingestion_date.isoformat()

        if isinstance(ingestion_date, str):
            try:
                return date.fromisoformat(
                    ingestion_date
                ).isoformat()
            except ValueError as exc:
                raise ValueError(
                    "ingestion_date must use YYYY-MM-DD format."
                ) from exc

        raise TypeError(
            "ingestion_date must be None, a date object, "
            "a datetime object, or a YYYY-MM-DD string."
        )

    @staticmethod
    def _add_batch_metadata(
        df: DataFrame,
        batch_id: str,
        ingestion_date: str,
    ) -> DataFrame:
        return (
            df
            .withColumn("batch_id", F.lit(batch_id))
            .withColumn(
                "ingestion_date",
                F.lit(ingestion_date).cast("date"),
            )
        )

    @staticmethod
    def _filesystem(spark, target_path: str):
        jvm = spark._jvm
        conf = spark._jsc.hadoopConfiguration()

        path = jvm.org.apache.hadoop.fs.Path(target_path)
        filesystem = path.getFileSystem(conf)

        return filesystem, path

    @classmethod
    def _delete_path(cls, spark, target_path: str) -> None:
        filesystem, path = cls._filesystem(spark, target_path)

        if filesystem.exists(path):
            deleted = filesystem.delete(path, True)

            if not deleted and filesystem.exists(path):
                raise RuntimeError(
                    f"Could not delete path: {target_path}"
                )

    @classmethod
    def _rename_path(
        cls,
        spark,
        source_path: str,
        target_path: str,
    ) -> None:
        source_fs, source = cls._filesystem(spark, source_path)
        target_fs, target = cls._filesystem(spark, target_path)

        if source_fs.getUri().toString() != target_fs.getUri().toString():
            raise RuntimeError(
                "Source and target must use the same filesystem."
            )

        if not source_fs.exists(source):
            raise RuntimeError(
                f"Rename source does not exist: {source_path}"
            )

        if target_fs.exists(target):
            raise RuntimeError(
                f"Rename target already exists: {target_path}"
            )

        parent = target.getParent()

        if not target_fs.exists(parent):
            if not target_fs.mkdirs(parent):
                raise RuntimeError(
                    f"Could not create directory: {parent}"
                )

        if not source_fs.rename(source, target):
            raise RuntimeError(
                f"Could not rename {source_path} to {target_path}"
            )

    @staticmethod
    def _write_partition(
        df: DataFrame,
        target_path: str,
        record_count: int,
    ) -> None:
        if record_count == 0:
            return

        data_to_write = df.drop(
            "ingestion_date",
            "batch_id",
        )

        (
            data_to_write.write
            .format("parquet")
            .mode("errorifexists")
            .save(target_path)
        )

    def write(
        self,
        valid_df: DataFrame,
        rejected_df: DataFrame,
        table_name: str,
        batch_id: str,
        ingestion_date=None,
    ) -> dict:

        # 1. Validate inputs.
        self._validate_path_segment(table_name, "table_name")
        self._validate_path_segment(batch_id, "batch_id")

        ingestion_date = self._normalize_ingestion_date(
            ingestion_date
        )

        # 2. Add batch metadata.
        valid_output = self._add_batch_metadata(
            valid_df,
            batch_id,
            ingestion_date,
        )

        rejected_output = self._add_batch_metadata(
            rejected_df,
            batch_id,
            ingestion_date,
        )

        # 3. Define table paths.
        silver_table_path = (
            f"{self.silver_base_path}/{table_name}"
        )

        quarantine_table_path = (
            f"{self.quarantine_base_path}/{table_name}"
        )

        partition_suffix = (
            f"/ingestion_date={ingestion_date}"
            f"/batch_id={batch_id}"
        )

        targets = {
            "silver": silver_table_path + partition_suffix,
            "quarantine": quarantine_table_path + partition_suffix,
        }

        # 4. Create transaction-specific staging and backup paths.
        transaction_id = uuid.uuid4().hex

        staging = {
            name: (
                f"{base}/_staging/{transaction_id}"
                + partition_suffix
            )
            for name, base in (
                ("silver", silver_table_path),
                ("quarantine", quarantine_table_path),
            )
        }

        backups = {
            name: (
                f"{base}/_backup/{transaction_id}"
                + partition_suffix
            )
            for name, base in (
                ("silver", silver_table_path),
                ("quarantine", quarantine_table_path),
            )
        }

        spark = valid_df.sparkSession

        # 5. Calculate output counts.
        valid_count = valid_output.count()
        rejected_count = rejected_output.count()

        outputs = {
            "silver": (valid_output, valid_count),
            "quarantine": (rejected_output, rejected_count),
        }

        backed_up = {
            "silver": False,
            "quarantine": False,
        }

        promoted = {
            "silver": False,
            "quarantine": False,
        }

        try:
            # 6. Write both outputs into staging.
            for name, (df, count) in outputs.items():
                self._write_partition(
                    df,
                    staging[name],
                    count,
                )

            # 7. Back up existing target partitions.
            for name in ("silver", "quarantine"):
                filesystem, path = self._filesystem(
                    spark,
                    targets[name],
                )

                if filesystem.exists(path):
                    self._rename_path(
                        spark,
                        targets[name],
                        backups[name],
                    )

                    backed_up[name] = True

            # 8. Promote staged outputs.
            for name in ("silver", "quarantine"):
                if outputs[name][1] > 0:
                    self._rename_path(
                        spark,
                        staging[name],
                        targets[name],
                    )

                    promoted[name] = True

        except Exception as write_error:
            rollback_errors = []

            # 9. Remove partially promoted replacements.
            for name in ("silver", "quarantine"):
                if promoted[name]:
                    try:
                        self._delete_path(
                            spark,
                            targets[name],
                        )
                    except Exception as exc:
                        rollback_errors.append(
                            f"Could not remove new {name}: {exc}"
                        )

            # 10. Restore original batch partitions.
            for name in ("silver", "quarantine"):
                if backed_up[name]:
                    try:
                        self._rename_path(
                            spark,
                            backups[name],
                            targets[name],
                        )
                    except Exception as exc:
                        rollback_errors.append(
                            f"Could not restore old {name}: {exc}"
                        )

            if rollback_errors:
                raise RuntimeError(
                    "Write failed and rollback was incomplete. "
                    "Preserve backup folders for recovery. "
                    + " | ".join(rollback_errors)
                ) from write_error

            raise RuntimeError(
                "Batch replacement failed. Previous outputs "
                "were restored."
            ) from write_error

        finally:
            # 11. Remove temporary staging folders.
            # Do not let cleanup errors hide an active write error.
            for name in ("silver", "quarantine"):
                try:
                    self._delete_path(
                        spark,
                        staging[name],
                    )
                except Exception:
                    pass

        # 12. Remove backups after successful promotion.
        for name in ("silver", "quarantine"):
            if backed_up[name]:
                self._delete_path(
                    spark,
                    backups[name],
                )

        # 13. Return write summary.
        return {
            "table_name": table_name,
            "batch_id": batch_id,
            "ingestion_date": ingestion_date,
            "silver_path": silver_table_path,
            "quarantine_path": quarantine_table_path,
            "silver_partition_path": targets["silver"],
            "quarantine_partition_path": targets["quarantine"],
            "valid_records_written": valid_count,
            "rejected_records_written": rejected_count,
            "write_mode": "staged_batch_replace_with_rollback",
            "transaction_id": transaction_id,
        }
