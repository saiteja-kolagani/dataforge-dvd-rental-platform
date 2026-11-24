
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
        prefix="silver_writer_idempotency_"
    )

    spark = (
        SparkSession.builder
        .appName("SilverWriterIdempotencyTest")
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
        writer = SilverWriter(
            silver_base_path=os.path.join(test_root, "silver"),
            quarantine_base_path=os.path.join(
                test_root, "quarantine"
            ),
        )

        print("\n========== TEST 1: INITIAL BATCH WRITE ==========")

        result = writer.write(
            valid_df=make_df([
                (1, "valid-one"),
                (2, "valid-two"),
            ]),
            rejected_df=make_df([
                (3, "rejected-one"),
            ]),
            table_name="test_table",
            batch_id="batch_A",
            ingestion_date="2026-11-23",
        )

        assert result["valid_records_written"] == 2
        assert result["rejected_records_written"] == 1

        print("Initial batch write: PASS")

        print("\n========== TEST 2: WRITE ANOTHER BATCH ==========")

        writer.write(
            valid_df=make_df([
                (4, "batch-b-valid"),
            ]),
            rejected_df=make_df([]),
            table_name="test_table",
            batch_id="batch_B",
            ingestion_date="2026-11-23",
        )

        print("Second batch write: PASS")

        print("\n========== TEST 3: REPROCESS BATCH A ==========")

        result = writer.write(
            valid_df=make_df([
                (10, "replacement-valid"),
            ]),
            rejected_df=make_df([
                (11, "replacement-rejected-one"),
                (12, "replacement-rejected-two"),
            ]),
            table_name="test_table",
            batch_id="batch_A",
            ingestion_date="2026-11-23",
        )

        assert result["valid_records_written"] == 1
        assert result["rejected_records_written"] == 2

        print("Batch A replacement: PASS")

        print("\n========== TEST 4: EMPTY SILVER OUTPUT ==========")

        writer.write(
            valid_df=make_df([]),
            rejected_df=make_df([
                (20, "batch-b-rejection"),
            ]),
            table_name="test_table",
            batch_id="batch_B",
            ingestion_date="2026-11-23",
        )

        print("Empty Silver output handling: PASS")

        print("\n========== TEST 5: VERIFY FINAL DATA ==========")

        silver_path = os.path.join(
            test_root, "silver", "test_table"
        )
        quarantine_path = os.path.join(
            test_root, "quarantine", "test_table"
        )

        silver_df = spark.read.parquet(silver_path)
        quarantine_df = spark.read.parquet(quarantine_path)

        silver_a = silver_df.filter(
            "batch_id = 'batch_A'"
        ).count()

        silver_b = silver_df.filter(
            "batch_id = 'batch_B'"
        ).count()

        quarantine_a = quarantine_df.filter(
            "batch_id = 'batch_A'"
        ).count()

        quarantine_b = quarantine_df.filter(
            "batch_id = 'batch_B'"
        ).count()

        print(f"Batch A Silver: {silver_a}")
        print(f"Batch B Silver: {silver_b}")
        print(f"Batch A Quarantine: {quarantine_a}")
        print(f"Batch B Quarantine: {quarantine_b}")

        assert silver_a == 1
        assert silver_b == 0
        assert quarantine_a == 2
        assert quarantine_b == 1

        print("\nAll idempotency assertions: PASS")

    finally:
        spark.stop()
        shutil.rmtree(test_root, ignore_errors=True)


if __name__ == "__main__":
    main()
