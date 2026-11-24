
from pathlib import Path
from datetime import date

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.inventory_processor import InventoryProcessor
from src.silver.silver_writer import SilverWriter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_INVENTORY_PATH = PROJECT_ROOT / "data" / "raw" / "inventory.csv"
FILM_SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver" / "film"
STORE_SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver" / "store"

SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver"
QUARANTINE_PATH = PROJECT_ROOT / "data" / "processed" / "quarantine"

BATCH_ID = "20261122_inventory_test_02"
INGESTION_DATE = date(2026, 11, 22)


def main():
    required_paths = [
        RAW_INVENTORY_PATH,
        FILM_SILVER_PATH,
        STORE_SILVER_PATH,
    ]

    missing_paths = [
        str(path) for path in required_paths if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Inventory integration test cannot run because these dependencies "
            "are missing:\n- " + "\n- ".join(missing_paths)
        )

    spark = (
        SparkSession.builder
        .appName("DataForge-Inventory-Silver-Test")
        .master("local[2]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        print("\n========== INVENTORY SILVER PROCESSING ==========")

        raw_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(str(RAW_INVENTORY_PATH))
        )

        film_df = spark.read.parquet(str(FILM_SILVER_PATH))
        store_df = spark.read.parquet(str(STORE_SILVER_PATH))

        source_count = raw_df.count()

        print(f"Raw Inventory records: {source_count}")
        print(f"Film Silver records: {film_df.count()}")
        print(f"Store Silver records: {store_df.count()}")

        processor = InventoryProcessor()

        valid_df, rejected_df = processor.process(
            source_df=raw_df,
            film_df=film_df,
            store_df=store_df,
        )

        valid_count = valid_df.count()
        rejected_count = rejected_df.count()

        print("\n========== PROCESSING SUMMARY ==========")
        print(f"Source records: {source_count}")
        print(f"Silver records: {valid_count}")
        print(f"Quarantine records: {rejected_count}")
        print(f"Total processed: {valid_count + rejected_count}")

        print("\n========== REJECTION REASONS ==========")

        (
            rejected_df.groupBy("rejection_reason")
            .count()
            .orderBy(F.desc("count"))
            .show(100, truncate=False)
        )

        print("\n========== INVENTORY SILVER SAMPLE ==========")
        valid_df.orderBy("inventory_id").show(50, truncate=False)

        print("\n========== INVENTORY QUARANTINE SAMPLE ==========")
        rejected_df.orderBy("inventory_id").show(100, truncate=False)

        print("\n========== TIMESTAMP QUALITY ==========")
        valid_df.groupBy("last_update_quality").count().show(
            truncate=False
        )

        print("\n========== FILM REFERENCE QUALITY ==========")
        valid_df.groupBy("film_reference_quality").count().show(
            truncate=False
        )

        print("\n========== STORE REFERENCE QUALITY ==========")
        valid_df.groupBy("store_reference_quality").count().show(
            truncate=False
        )

        # Data quality checks
        duplicate_inventory_ids = (
            valid_df.groupBy("inventory_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        invalid_film_references = valid_df.filter(
            F.col("film_reference_quality") != "VALID"
        ).count()

        invalid_store_statuses = valid_df.filter(
            F.col("store_reference_quality") == "INVALID_ID"
        ).count()

        unresolved_store_references = valid_df.filter(
            F.col("store_reference_quality") == "UNRESOLVED"
        ).count()

        invalid_inventory_ids = valid_df.filter(
            F.col("inventory_id").isNull()
            | (F.col("inventory_id") <= 0)
        ).count()

        print("\n========== DATA QUALITY CHECKS ==========")
        print(f"Duplicate Inventory IDs: {duplicate_inventory_ids}")
        print(f"Invalid Film references: {invalid_film_references}")
        print(f"Invalid Store IDs: {invalid_store_statuses}")
        print(f"Unresolved Store references: {unresolved_store_references}")
        print(f"Invalid Inventory IDs: {invalid_inventory_ids}")

        # General assertions
        assert source_count == valid_count + rejected_count
        assert duplicate_inventory_ids == 0
        assert invalid_film_references == 0
        assert invalid_store_statuses == 0
        assert invalid_inventory_ids == 0

        # Inventory ID 488:
        # Both records reference Film ID 150, which is absent
        # from Film Silver. Therefore, both must be quarantined.
        assert valid_df.filter(
            F.col("inventory_id") == 488
        ).count() == 0

        assert rejected_df.filter(
            (F.col("inventory_id") == 488)
            & (
                F.col("rejection_reason")
                == "INVALID_FILM_REFERENCE"
            )
        ).count() == 2

        # Inventory ID 31:
        # Three identical records exist.
        # One should survive in Silver and two should be quarantined.
        assert valid_df.filter(
            F.col("inventory_id") == 31
        ).count() == 1

        assert rejected_df.filter(
            (F.col("inventory_id") == 31)
            & (
                F.col("rejection_reason")
                == "DUPLICATE_INVENTORY_RECORD"
            )
        ).count() == 2

        print("\nAll Inventory Silver data-quality checks passed.")

        # Write Silver and Quarantine datasets
        writer = SilverWriter(
            silver_base_path=str(SILVER_PATH),
            quarantine_base_path=str(QUARANTINE_PATH),
        )

        result = writer.write(
            valid_df=valid_df,
            rejected_df=rejected_df,
            table_name="inventory",
            batch_id=BATCH_ID,
            ingestion_date=INGESTION_DATE,
        )

        print("\n========== INVENTORY WRITE SUMMARY ==========")
        print(result)

        print("\nInventory Silver processing completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
