
from pathlib import Path
from datetime import date

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.address_processor import AddressProcessor
from src.silver.silver_writer import SilverWriter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ADDRESS_PATH = PROJECT_ROOT / "data" / "raw" / "address.csv"
CITY_SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver" / "city"

SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver"
QUARANTINE_PATH = PROJECT_ROOT / "data" / "processed" / "quarantine"

BATCH_ID = "20261122_address_test_01"
INGESTION_DATE = date(2026, 11, 22)


def main():

    spark = (
        SparkSession.builder
        .appName("DataForge-Address-Silver-Test")
        .master("local[2]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        print("\n========== ADDRESS SILVER PROCESSING ==========")

        # ----------------------------------------------
        # Read source datasets
        # ----------------------------------------------

        raw_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(str(RAW_ADDRESS_PATH))
        )

        city_silver_df = spark.read.parquet(
            str(CITY_SILVER_PATH)
        )

        source_count = raw_df.count()
        city_count = city_silver_df.count()

        print(f"Raw Address records: {source_count}")
        print(f"City Silver records: {city_count}")

        # ----------------------------------------------
        # Process Address
        # ----------------------------------------------

        processor = AddressProcessor()

        valid_df, rejected_df = processor.process(
            raw_df=raw_df,
            city_silver_df=city_silver_df,
        )

        valid_count = valid_df.count()
        rejected_count = rejected_df.count()

        print("\n========== PROCESSING SUMMARY ==========")
        print(f"Source records: {source_count}")
        print(f"Silver records: {valid_count}")
        print(f"Quarantine records: {rejected_count}")
        print(f"Total processed: {valid_count + rejected_count}")

        # ----------------------------------------------
        # Rejection reason summary
        # ----------------------------------------------

        print("\n========== REJECTION REASONS ==========")

        rejected_df.groupBy(
            "rejection_reason"
        ).count().orderBy(
            F.desc("count")
        ).show(100, truncate=False)

        # ----------------------------------------------
        # Timestamp quality
        # ----------------------------------------------

        print("\n========== TIMESTAMP QUALITY ==========")

        valid_df.groupBy(
            "last_update_quality"
        ).count().show()

        # ----------------------------------------------
        # Optional field quality
        # ----------------------------------------------

        print("\n========== OPTIONAL FIELD QUALITY ==========")

        for column in [
            "postal_code_quality",
            "phone_quality",
        ]:
            valid_df.groupBy(column).count().show()

        # ----------------------------------------------
        # Silver sample
        # ----------------------------------------------

        print("\n========== ADDRESS SILVER SAMPLE ==========")

        valid_df.orderBy(
            "address_id"
        ).show(50, truncate=False)

        # ----------------------------------------------
        # Data quality checks
        # ----------------------------------------------

        print("\n========== DATA QUALITY CHECKS ==========")

        duplicate_ids = (
            valid_df.groupBy("address_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        invalid_city_references = (
            valid_df.select("city_id")
            .join(
                city_silver_df.select("city_id").distinct(),
                on="city_id",
                how="left_anti"
            )
            .count()
        )

        invalid_address_ids = valid_df.filter(
            F.col("address_id").isNull()
            | (F.col("address_id") <= 0)
        ).count()

        missing_addresses = valid_df.filter(
            F.col("address").isNull()
            | (F.trim(F.col("address")) == "")
        ).count()

        missing_districts = valid_df.filter(
            F.col("district").isNull()
            | (F.trim(F.col("district")) == "")
        ).count()

        print(f"Duplicate address IDs: {duplicate_ids}")
        print(f"Invalid city references: {invalid_city_references}")
        print(f"Invalid address IDs: {invalid_address_ids}")
        print(f"Missing addresses: {missing_addresses}")
        print(f"Missing districts: {missing_districts}")

        assert source_count == valid_count + rejected_count
        assert duplicate_ids == 0
        assert invalid_city_references == 0
        assert invalid_address_ids == 0
        assert missing_addresses == 0
        assert missing_districts == 0

        print("\nAll Address data quality checks passed.")

        # ----------------------------------------------
        # Write Silver and Quarantine
        # ----------------------------------------------

        writer = SilverWriter(
            silver_base_path=str(SILVER_PATH),
            quarantine_base_path=str(QUARANTINE_PATH),
        )

        result = writer.write(
            valid_df=valid_df,
            rejected_df=rejected_df,
            table_name="address",
            batch_id=BATCH_ID,
            ingestion_date=INGESTION_DATE,
        )

        print("\n========== ADDRESS WRITE SUMMARY ==========")
        print(result)

        print("\nAddress Silver processing completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
