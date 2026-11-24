# from pyspark.sql import SparkSession
# from pyspark.sql import functions as F

# from src.silver.city_processor import CityProcessor
# from src.silver.silver_writer import SilverWriter

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.city_processor import CityProcessor
from src.silver.silver_writer import SilverWriter

def main():
    spark = (
        SparkSession.builder
        .appName("DataForge-City-Silver-Test")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        # -----------------------------------------------------
        # 1. Read raw City data
        # -----------------------------------------------------
        city_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv("data/raw/city.csv")
        )

        # -----------------------------------------------------
        # 2. Read already-cleansed Country Silver data
        # -----------------------------------------------------
        country_silver_df = spark.read.parquet(
            "data/processed/silver/country"
        )

        print("\n========== CITY SILVER PROCESSING ==========")

        print(f"Raw City records: {city_df.count()}")
        print(
            f"Country Silver records: {country_silver_df.count()}"
        )

        # -----------------------------------------------------
        # 3. Execute City processor
        # -----------------------------------------------------
        processor = CityProcessor()

        valid_df, rejected_df = processor.process(
            city_df=city_df,
            country_silver_df=country_silver_df,
        )

        valid_count = valid_df.count()
        rejected_count = rejected_df.count()

        print("\n========== PROCESSING SUMMARY ==========")
        print(f"Source records: {city_df.count()}")
        print(f"Silver records: {valid_count}")
        print(f"Quarantine records: {rejected_count}")
        print(f"Total processed: {valid_count + rejected_count}")

        # -----------------------------------------------------
        # 4. Rejection reason analysis
        # -----------------------------------------------------
        print("\n========== REJECTION REASONS ==========")

        (
            rejected_df
            .groupBy("rejection_reason")
            .count()
            .orderBy(F.desc("count"))
            .show(truncate=False)
        )

        # -----------------------------------------------------
        # 5. Timestamp quality analysis
        # -----------------------------------------------------
        print("\n========== TIMESTAMP QUALITY ==========")

        (
            valid_df
            .groupBy("last_update_quality")
            .count()
            .orderBy("last_update_quality")
            .show()
        )

        # -----------------------------------------------------
        # 6. Inspect valid City records
        # -----------------------------------------------------
        print("\n========== CITY SILVER SAMPLE ==========")

        valid_df.orderBy("city_id").show(
            50,
            truncate=False,
        )

        # -----------------------------------------------------
        # 7. Inspect quarantined records
        # -----------------------------------------------------
        print("\n========== CITY QUARANTINE SAMPLE ==========")

        rejected_df.orderBy("city_id").show(
            50,
            truncate=False,
        )

        # -----------------------------------------------------
        # 8. Data quality assertions
        # -----------------------------------------------------
        print("\n========== DATA QUALITY CHECKS ==========")

        duplicate_city_ids = (
            valid_df
            .groupBy("city_id")
            .count()
            .where(F.col("count") > 1)
            .count()
        )

        invalid_country_references = (
            valid_df
            .join(
                country_silver_df.select("country_id").distinct(),
                on="country_id",
                how="left_anti",
            )
            .count()
        )

        invalid_city_ids = (
            valid_df
            .where(
                F.col("city_id").isNull()
                | (F.col("city_id") <= 0)
            )
            .count()
        )

        missing_city_names = (
            valid_df
            .where(
                F.col("city").isNull()
                | (F.trim("city") == "")
            )
            .count()
        )

        print(f"Duplicate city IDs: {duplicate_city_ids}")
        print(
            f"Invalid Country references: {invalid_country_references}"
        )
        print(f"Invalid city IDs: {invalid_city_ids}")
        print(f"Missing city names: {missing_city_names}")

        assert valid_count + rejected_count == city_df.count()
        assert duplicate_city_ids == 0
        assert invalid_country_references == 0
        assert invalid_city_ids == 0
        assert missing_city_names == 0

        # Expected results based on the agreed cleansing rules
        assert valid_count == 38, (
            f"Expected 38 Silver records, got {valid_count}"
        )
        assert rejected_count == 16, (
            f"Expected 16 Quarantine records, got {rejected_count}"
        )

        print("\nAll City data quality checks passed.")

        # -----------------------------------------------------
        # 9. Write Silver and Quarantine outputs
        # -----------------------------------------------------
        batch_id = "20261122_city_test_01"

        writer = SilverWriter(
            silver_base_path="data/processed/silver",
            quarantine_base_path="data/processed/quarantine",
        )

        result = writer.write(
            valid_df=valid_df,
            rejected_df=rejected_df,
            table_name="city",
            batch_id=batch_id,
            ingestion_date="2026-11-22",
        )

        print("\n========== CITY WRITE SUMMARY ==========")
        print(result)

        print("\nCity Silver processing completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
