
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.silver.category_processor import CategoryProcessor
from src.silver.silver_writer import SilverWriter


def create_spark_session():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    return (
        SparkSession.builder
        .appName("DataForge-Category-Silver-Test")
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config(
            "spark.hadoop.fs.file.impl",
            "org.apache.hadoop.fs.RawLocalFileSystem",
        )
        .getOrCreate()
    )


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        source_path = PROJECT_ROOT / "data" / "raw" / "category.csv"

        source_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(str(source_path))
        )

        valid_df, rejected_df = CategoryProcessor().process(source_df)

        print("\n--- SOURCE SUMMARY ---")
        print(f"Source records: {source_df.count()}")

        print("\n--- PROCESSING SUMMARY ---")
        valid_count = valid_df.count()
        rejected_count = rejected_df.count()

        print(f"Valid Category records: {valid_count}")
        print(f"Rejected Category records: {rejected_count}")
        print(f"Total processed: {valid_count + rejected_count}")

        print("\n--- VALID CATEGORY RECORDS ---")
        valid_df.orderBy("category_id").show(30, truncate=False)

        print("\n--- REJECTION REASONS ---")
        (
            rejected_df
            .groupBy("rejection_reason")
            .count()
            .orderBy("rejection_reason")
            .show(truncate=False)
        )

        print("\n--- REJECTED RECORDS ---")
        rejected_df.orderBy("category_id").show(30, truncate=False)

        # Validate expected counts based on the agreed rules.
        assert source_df.count() == 19
        assert valid_count == 12
        assert rejected_count == 7
        assert valid_count + rejected_count == source_df.count()

        # Ensure Silver contains no duplicate category IDs.
        duplicate_ids = (
            valid_df
            .groupBy("category_id")
            .count()
            .filter("count > 1")
            .count()
        )
        assert duplicate_ids == 0

        # Ensure all valid IDs are positive.
        assert valid_df.filter(
            "category_id IS NULL OR category_id <= 0"
        ).count() == 0

        # Ensure conflicting IDs 3 and 12 are absent from Silver.
        assert valid_df.filter(
            "category_id IN (3, 12)"
        ).count() == 0

        # Ensure category ID 10 has exactly one survivor.
        assert valid_df.filter(
            "category_id = 10"
        ).count() == 1

        # Write Silver and Quarantine Parquet outputs.
        output_root = PROJECT_ROOT / "data" / "processed"

        writer = SilverWriter(
            silver_base_path=str(output_root / "silver"),
            quarantine_base_path=str(output_root / "quarantine"),
        )

        result = writer.write(
            valid_df=valid_df,
            rejected_df=rejected_df,
            table_name="category",
            batch_id="20261122_category_test_01",
            ingestion_date="2026-11-22",
        )

        print("\n--- WRITE SUMMARY ---")
        for key, value in result.items():
            print(f"{key}: {value}")

        print(
            "\nSUCCESS: Category Silver processing and "
            "Parquet verification passed."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
