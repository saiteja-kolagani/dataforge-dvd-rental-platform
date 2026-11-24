
from pathlib import Path
from datetime import date

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.customer_processor import CustomerProcessor
from src.silver.silver_writer import SilverWriter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_CUSTOMER_PATH = PROJECT_ROOT / "data" / "raw" / "customer.csv"
ADDRESS_SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver" / "address"
STORE_SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver" / "store"

SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver"
QUARANTINE_PATH = PROJECT_ROOT / "data" / "processed" / "quarantine"

BATCH_ID = "20261122_customer_test_02"
INGESTION_DATE = date(2026, 11, 22)


def main():

    spark = (
        SparkSession.builder
        .appName("DataForge-Customer-Silver-Test")
        .master("local[2]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        print("\n========== CUSTOMER SILVER PROCESSING ==========")

        # ----------------------------------------------
        # Read source datasets
        # ----------------------------------------------

        raw_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(str(RAW_CUSTOMER_PATH))
        )

        address_silver_df = spark.read.parquet(
            str(ADDRESS_SILVER_PATH)
        )

        store_silver_df = spark.read.parquet(
            str(STORE_SILVER_PATH)
        )

        source_count = raw_df.count()
        address_count = address_silver_df.count()
        store_count = store_silver_df.count()

        print(f"Raw Customer records: {source_count}")
        print(f"Address Silver records: {address_count}")
        print(f"Store Silver records: {store_count}")

        # ----------------------------------------------
        # Process Customer
        # ----------------------------------------------

        processor = CustomerProcessor()

        valid_df, rejected_df = processor.process(
            source_df=raw_df,
            store_df=store_silver_df,
            address_df=address_silver_df,
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

        (
            rejected_df
            .groupBy("rejection_reason")
            .count()
            .orderBy(F.desc("count"))
            .show(100, truncate=False)
        )

        # ----------------------------------------------
        # Silver records
        # ----------------------------------------------

        print("\n========== CUSTOMER SILVER SAMPLE ==========")

        valid_df.orderBy(
            "customer_id"
        ).show(50, truncate=False)

        # ----------------------------------------------
        # Quarantine records
        # ----------------------------------------------

        print("\n========== CUSTOMER QUARANTINE SAMPLE ==========")

        rejected_df.orderBy(
            "customer_id"
        ).show(100, truncate=False)

        # ----------------------------------------------
        # Email quality
        # ----------------------------------------------

        print("\n========== EMAIL QUALITY ==========")

        (
            valid_df
            .groupBy("email_quality")
            .count()
            .orderBy("email_quality")
            .show(truncate=False)
        )

        # ----------------------------------------------
        # Active status quality
        # ----------------------------------------------

        print("\n========== ACTIVEBOOL QUALITY ==========")

        (
            valid_df
            .groupBy("activebool", "activebool_quality")
            .count()
            .orderBy("activebool_quality")
            .show(truncate=False)
        )

        print("\n========== ACTIVE QUALITY ==========")

        (
            valid_df
            .groupBy("active", "active_quality")
            .count()
            .orderBy("active_quality")
            .show(truncate=False)
        )

        print("\n========== ACTIVE STATUS CONSISTENCY ==========")

        (
            valid_df
            .groupBy("active_status_quality")
            .count()
            .show(truncate=False)
        )

        # ----------------------------------------------
        # Timestamp quality
        # ----------------------------------------------

        print("\n========== CREATE DATE QUALITY ==========")

        (
            valid_df
            .groupBy("create_date_quality")
            .count()
            .show(truncate=False)
        )

        print("\n========== LAST UPDATE QUALITY ==========")

        (
            valid_df
            .groupBy("last_update_quality")
            .count()
            .show(truncate=False)
        )

        # ----------------------------------------------
        # Referential integrity quality
        # ----------------------------------------------

        print("\n========== STORE REFERENCE QUALITY ==========")

        (
            valid_df
            .groupBy("store_reference_quality")
            .count()
            .show(truncate=False)
        )

        print("\n========== ADDRESS REFERENCE QUALITY ==========")

        (
            valid_df
            .groupBy("address_reference_quality")
            .count()
            .show(truncate=False)
        )

        # ----------------------------------------------
        # Data quality checks
        # ----------------------------------------------

        print("\n========== DATA QUALITY CHECKS ==========")

        duplicate_customer_ids = (
            valid_df
            .groupBy("customer_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        invalid_address_references = (
            valid_df
            .join(
                address_silver_df.select("address_id").distinct(),
                on="address_id",
                how="left_anti",
            )
            .count()
        )

        invalid_store_statuses = valid_df.filter(
            F.col("store_reference_quality") == "INVALID"
        ).count()

        unresolved_store_references = valid_df.filter(
            F.col("store_reference_quality") == "UNRESOLVED"
        ).count()

        invalid_customer_ids = valid_df.filter(
            F.col("customer_id").isNull()
            | (F.col("customer_id") <= 0)
        ).count()

        missing_customer_names = valid_df.filter(
            F.col("first_name").isNull()
            | F.col("last_name").isNull()
        ).count()

        invalid_create_dates = valid_df.filter(
            F.col("create_date").isNull()
        ).count()

        invalid_address_ids = valid_df.filter(
            F.col("address_id").isNull()
            | (F.col("address_id") <= 0)
        ).count()

        print(f"Duplicate Customer IDs: {duplicate_customer_ids}")
        print(f"Invalid Address references: {invalid_address_references}")
        print(f"Invalid Store references: {invalid_store_statuses}")
        print(f"Unresolved Store references: {unresolved_store_references}")
        print(f"Invalid Customer IDs: {invalid_customer_ids}")
        print(f"Missing Customer names: {missing_customer_names}")
        print(f"Invalid Create dates: {invalid_create_dates}")
        print(f"Invalid Address IDs: {invalid_address_ids}")

        # ----------------------------------------------
        # Controlled survivorship regression test: Customer 6
        # ----------------------------------------------

        print("\\n========== CUSTOMER 6 SURVIVORSHIP CHECK ==========")

        customer_6_silver = valid_df.filter(
            F.col("customer_id") == 6
        )

        customer_6_quarantine = rejected_df.filter(
            F.col("customer_id") == 6
        )

        customer_6_silver_count = customer_6_silver.count()
        customer_6_quarantine_count = customer_6_quarantine.count()

        customer_6_silver_row = customer_6_silver.select(
            "active",
            "active_quality",
            "raw_active"
        ).first()

        customer_6_quarantine_row = customer_6_quarantine.select(
            "active_quality",
            "raw_active",
            "rejection_reason"
        ).first()

        print(f"Customer 6 Silver records: {customer_6_silver_count}")
        print(f"Customer 6 Quarantine records: {customer_6_quarantine_count}")
        print(f"Customer 6 Silver survivor: {customer_6_silver_row}")
        print(f"Customer 6 quarantined version: {customer_6_quarantine_row}")

        assert customer_6_silver_count == 1, (
            "Customer 6 must have exactly one Silver survivor."
        )
        assert customer_6_quarantine_count == 1, (
            "Customer 6 must have exactly one quarantined duplicate."
        )
        assert customer_6_silver_row["active"] is False, (
            "Customer 6 survivor must have active=False."
        )
        assert customer_6_silver_row["active_quality"] == "VALID"
        assert customer_6_silver_row["raw_active"] == "N"
        assert customer_6_quarantine_row["active_quality"] == "MISSING"
        assert customer_6_quarantine_row["rejection_reason"] == (
            "DUPLICATE_CUSTOMER_RECORD"
        )

        # ----------------------------------------------
        # Processing reconciliation
        # ----------------------------------------------

        print("\n========== PROCESSING RECONCILIATION ==========")

        print(f"Source count: {source_count}")
        print(f"Silver count: {valid_count}")
        print(f"Quarantine count: {rejected_count}")

        assert source_count == valid_count + rejected_count
        assert duplicate_customer_ids == 0
        assert invalid_address_references == 0
        assert invalid_store_statuses == 0
        assert invalid_customer_ids == 0
        assert missing_customer_names == 0
        assert invalid_create_dates == 0
        assert invalid_address_ids == 0

        print("\nAll Customer data quality checks passed.")

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
            table_name="customer",
            batch_id=BATCH_ID,
            ingestion_date=INGESTION_DATE,
        )

        print("\n========== CUSTOMER WRITE SUMMARY ==========")
        print(result)

        print("\nCustomer Silver processing completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
