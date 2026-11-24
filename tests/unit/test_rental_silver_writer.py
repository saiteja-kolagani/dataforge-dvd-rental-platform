from pathlib import Path
from datetime import date

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.rental_processor import RentalProcessor
from src.silver.silver_writer import SilverWriter

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "rental.csv"
INVENTORY = ROOT / "data" / "processed" / "silver" / "inventory"
CUSTOMER = ROOT / "data" / "processed" / "silver" / "customer"
STAFF = ROOT / "data" / "processed" / "silver" / "staff"
SILVER = ROOT / "data" / "processed" / "silver"
QUARANTINE = ROOT / "data" / "processed" / "quarantine"

BATCH_ID = "20261123_rental_test_01"
INGESTION_DATE = date(2026, 11, 23)


def main():
    required = [RAW, INVENTORY, CUSTOMER, STAFF]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Rental test dependencies are missing:\n- " + "\n- ".join(missing)
        )

    spark = (
        SparkSession.builder.appName("DataForge-Rental-Silver-Test")
        .master("local[2]").getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        print("\n========== RENTAL SILVER PROCESSING ==========")
        raw = spark.read.option("header", True).option(
            "inferSchema", False
        ).csv(str(RAW))
        inventory = spark.read.parquet(str(INVENTORY))
        customer = spark.read.parquet(str(CUSTOMER))
        staff = spark.read.parquet(str(STAFF))

        source_count = raw.count()
        print(f"Raw Rental records: {source_count}")
        print(f"Inventory Silver records: {inventory.count()}")
        print(f"Customer Silver records: {customer.count()}")
        print(f"Staff Silver records: {staff.count()}")

        valid, rejected = RentalProcessor().process(
            source_df=raw,
            inventory_df=inventory,
            customer_df=customer,
            staff_df=staff,
        )
        valid_count, rejected_count = valid.count(), rejected.count()

        print("\n========== PROCESSING SUMMARY ==========")
        print(f"Source records: {source_count}")
        print(f"Silver records: {valid_count}")
        print(f"Quarantine records: {rejected_count}")
        print(f"Total processed: {valid_count + rejected_count}")

        print("\n========== REJECTION REASONS ==========")
        rejected.groupBy("rejection_reason").count().orderBy(
            F.desc("count")
        ).show(100, truncate=False)

        print("\n========== RENTAL SILVER SAMPLE ==========")
        valid.orderBy("rental_id").show(40, truncate=False)
        print("\n========== RENTAL QUARANTINE SAMPLE ==========")
        rejected.orderBy("rental_id").show(80, truncate=False)

        print("\n========== TIMESTAMP QUALITY ==========")
        for c in ["rental_date_quality", "return_date_quality", "last_update_quality"]:
            print(c)
            valid.groupBy(c).count().show(truncate=False)

        duplicate_ids = valid.groupBy("rental_id").count().filter(
            F.col("count") > 1
        ).count()
        bad_inventory = valid.filter(
            F.col("inventory_reference_quality") != "VALID"
        ).count()
        bad_customer = valid.filter(
            F.col("customer_reference_quality") != "VALID"
        ).count()
        bad_staff = valid.filter(
            ~F.col("staff_reference_quality").isin("VALID", "UNRESOLVED")
        ).count()
        unresolved_staff = valid.filter(
            F.col("staff_reference_quality") == "UNRESOLVED"
        ).count()
        invalid_staff_ids_quarantined = rejected.filter(
            F.col("staff_reference_quality") == "INVALID_ID"
        ).count()
        bad_rental_dates = valid.filter(F.col("rental_date").isNull()).count()

        print("\n========== DATA QUALITY CHECKS ==========")
        print(f"Duplicate Rental IDs: {duplicate_ids}")
        print(f"Invalid Inventory references: {bad_inventory}")
        print(f"Invalid Customer references: {bad_customer}")
        print(f"Invalid Staff references in Silver: {bad_staff}")
        print(f"Unresolved Staff references in Silver: {unresolved_staff}")
        print(f"Invalid Staff IDs quarantined: {invalid_staff_ids_quarantined}")
        print(f"Invalid Rental dates: {bad_rental_dates}")

        assert source_count == valid_count + rejected_count
        assert duplicate_ids == 0
        assert bad_inventory == 0
        assert bad_customer == 0
        assert bad_staff == 0
        assert invalid_staff_ids_quarantined >= 0
        assert bad_rental_dates == 0
        print("\nAll Rental Silver data-quality checks passed.")

        writer = SilverWriter(
            silver_base_path=str(SILVER),
            quarantine_base_path=str(QUARANTINE),
        )
        result = writer.write(
            valid_df=valid,
            rejected_df=rejected,
            table_name="rental",
            batch_id=BATCH_ID,
            ingestion_date=INGESTION_DATE,
        )
        print("\n========== RENTAL WRITE SUMMARY ==========")
        print(result)
        print("\nRental Silver processing completed successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
