
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.silver.country_processor import CountryProcessor
from src.silver.silver_writer import SilverWriter


def create_spark_session():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    return (
        SparkSession.builder
        .appName("DataForge-Country-Silver-Test")
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
        source_path = PROJECT_ROOT / "data" / "raw" / "country.csv"

        source_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(str(source_path))
        )

        valid_df, rejected_df = CountryProcessor().process(source_df)

        print("\n--- SOURCE SUMMARY ---")
        source_count = source_df.count()
        print(f"Source records: {source_count}")

        print("\n--- PROCESSING SUMMARY ---")
        valid_count = valid_df.count()
        rejected_count = rejected_df.count()

        print(f"Valid Country records: {valid_count}")
        print(f"Rejected Country records: {rejected_count}")
        print(f"Total processed: {valid_count + rejected_count}")

        print("\n--- VALID COUNTRY RECORDS ---")
        valid_df.orderBy("country_id").show(60, truncate=False)

        print("\n--- REJECTION REASONS ---")
        (
            rejected_df
            .groupBy("rejection_reason")
            .count()
            .orderBy("rejection_reason")
            .show(truncate=False)
        )

        print("\n--- REJECTED RECORDS ---")
        rejected_df.orderBy("country_id").show(60, truncate=False)

        # Validate expected processing counts.
        assert source_count == 54
        assert valid_count == 47
        assert rejected_count == 7
        assert valid_count + rejected_count == source_count

        # Ensure no duplicate IDs remain in Silver.
        duplicate_ids = (
            valid_df
            .groupBy("country_id")
            .count()
            .filter("count > 1")
            .count()
        )
        assert duplicate_ids == 0

        # Ensure country IDs are positive.
        assert valid_df.filter(
            "country_id IS NULL OR country_id <= 0"
        ).count() == 0

        # Ensure conflicting IDs 35 and 39 are rejected entirely.
        assert valid_df.filter(
            "country_id IN (35, 39)"
        ).count() == 0

        # Ensure exact duplicate IDs have one survivor.
        assert valid_df.filter("country_id = 8").count() == 1
        assert valid_df.filter("country_id = 41").count() == 1

        # Write Silver and Quarantine Parquet outputs.
        output_root = PROJECT_ROOT / "data" / "processed"

        writer = SilverWriter(
            silver_base_path=str(output_root / "silver"),
            quarantine_base_path=str(output_root / "quarantine"),
        )

        result = writer.write(
            valid_df=valid_df,
            rejected_df=rejected_df,
            table_name="country",
            batch_id="20261122_country_test_01",
            ingestion_date="2026-11-22",
        )

        print("\n--- WRITE SUMMARY ---")
        for key, value in result.items():
            print(f"{key}: {value}")

        print(
            "\nSUCCESS: Country Silver processing and "
            "Parquet verification passed."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
