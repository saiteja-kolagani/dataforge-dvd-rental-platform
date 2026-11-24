
import os
import shutil
import tempfile

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.silver.silver_writer import SilverWriter


def main():
    test_root = tempfile.mkdtemp(
        prefix="silver_writer_rollback_"
    )

    spark = (
        SparkSession.builder
        .appName("SilverWriterRollbackTest")
        .master("local[2]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    schema = StructType([
        StructField("record_id", IntegerType(), False),
        StructField("description", StringType(), True),
    ])

    def make_df(rows):
        return spark.createDataFrame(rows, schema)

    try:
        silver_base = os.path.join(test_root, "silver")
        quarantine_base = os.path.join(test_root, "quarantine")

        writer = SilverWriter(silver_base, quarantine_base)

        print("\n========== TEST 1: ORIGINAL BATCH ==========")

        writer.write(
            make_df([(1, "original-valid")]),
            make_df([(2, "original-rejected")]),
            "test_table",
            "batch_failure",
            "2026-11-23",
        )

        print("Original batch written: PASS")

        print("\n========== TEST 2: SIMULATE PROMOTION FAILURE ==========")

        original_rename = writer._rename_path
        calls = {"count": 0}

        def fail_quarantine_promotion(
            spark_session,
            source_path,
            target_path,
        ):
            calls["count"] += 1

            # Call 1: back up Silver.
            # Call 2: back up Quarantine.
            # Call 3: promote Silver.
            # Call 4: promote Quarantine -> simulate failure.
            if calls["count"] == 4:
                raise RuntimeError(
                    "SIMULATED QUARANTINE PROMOTION FAILURE"
                )

            return original_rename(
                spark_session,
                source_path,
                target_path,
            )

        writer._rename_path = fail_quarantine_promotion

        try:
            writer.write(
                make_df([(10, "replacement-valid")]),
                make_df([(20, "replacement-rejected")]),
                "test_table",
                "batch_failure",
                "2026-11-23",
            )
        except RuntimeError as exc:
            print(f"Expected failure captured: {exc}")
        else:
            raise AssertionError(
                "Expected promotion failure did not occur."
            )

        print("\n========== TEST 3: VERIFY RESTORATION ==========")

        silver_path = os.path.join(
            silver_base,
            "test_table",
            "ingestion_date=2026-11-23",
            "batch_id=batch_failure",
        )
        quarantine_path = os.path.join(
            quarantine_base,
            "test_table",
            "ingestion_date=2026-11-23",
            "batch_id=batch_failure",
        )

        silver_df = spark.read.parquet(silver_path)
        quarantine_df = spark.read.parquet(quarantine_path)

        silver_rows = [
            row.asDict()
            for row in silver_df.collect()
        ]
        quarantine_rows = [
            row.asDict()
            for row in quarantine_df.collect()
        ]

        assert len(silver_rows) == 1
        assert silver_rows[0]["record_id"] == 1

        assert len(quarantine_rows) == 1
        assert quarantine_rows[0]["record_id"] == 2

        print("Original Silver record restored: PASS")
        print("Original Quarantine record restored: PASS")

        print("\nAll rollback assertions: PASS")

    finally:
        spark.stop()
        shutil.rmtree(test_root, ignore_errors=True)


if __name__ == "__main__":
    main()
