
from datetime import date

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.staff_processor import StaffProcessor
from src.silver.silver_writer import SilverWriter


spark = (
    SparkSession.builder
    .appName("DataForge-Staff-Silver-Test")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    raw_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv("data/raw/staff.csv")
    )

    address_silver_df = spark.read.parquet(
        "data/processed/silver/address"
    )

    store_silver_df = spark.read.parquet(
        "data/processed/silver/store"
    )

    processor = StaffProcessor()

    valid_df, rejected_df = processor.process(
        raw_df=raw_df,
        address_silver_df=address_silver_df,
        store_silver_df=store_silver_df,
    )

    source_count = raw_df.count()
    valid_count = valid_df.count()
    rejected_count = rejected_df.count()

    print("\n========== STAFF SILVER PROCESSING ==========")
    print("Source records:", source_count)
    print("Silver records:", valid_count)
    print("Quarantine records:", rejected_count)
    print("Total processed:", valid_count + rejected_count)

    print("\n========== SILVER RECORDS ==========")
    valid_df.orderBy("staff_id").show(truncate=False)

    print("\n========== QUARANTINE RECORDS ==========")
    rejected_df.orderBy("staff_id").show(truncate=False)

    print("\n========== REJECTION REASONS ==========")
    (
        rejected_df
        .groupBy("rejection_reason")
        .count()
        .orderBy(F.desc("count"))
        .show(truncate=False)
    )

    print("\n========== ACTIVE STATUS QUALITY ==========")
    (
        valid_df
        .groupBy("active", "active_quality")
        .count()
        .orderBy("active_quality")
        .show(truncate=False)
    )

    print("\n========== EMAIL QUALITY ==========")
    (
        valid_df
        .groupBy("email_quality")
        .count()
        .show(truncate=False)
    )

    print("\n========== TIMESTAMP QUALITY ==========")
    (
        valid_df
        .groupBy("last_update_quality")
        .count()
        .show(truncate=False)
    )

    print("\n========== SECURITY CHECKS ==========")

    sensitive_columns = [
        column for column in valid_df.columns
        if "password" in column.lower()
    ]

    rejected_sensitive_columns = [
        column for column in rejected_df.columns
        if "password" in column.lower()
    ]

    print("Password-related Silver columns:", sensitive_columns)
    print("Password-related Quarantine columns:", rejected_sensitive_columns)

    assert not sensitive_columns
    assert not rejected_sensitive_columns

    print("\n========== DATA QUALITY CHECKS ==========")

    duplicate_staff_ids = (
        valid_df
        .groupBy("staff_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    invalid_address_refs = (
        valid_df
        .join(
            address_silver_df.select("address_id").distinct(),
            on="address_id",
            how="left_anti",
        )
        .count()
    )

    invalid_store_refs = (
        valid_df
        .join(
            store_silver_df.select("store_id").distinct(),
            on="store_id",
            how="left_anti",
        )
        .count()
    )

    invalid_mandatory_fields = valid_df.filter(
        F.col("staff_id").isNull()
        | (F.col("staff_id") <= 0)
        | F.col("first_name").isNull()
        | F.col("last_name").isNull()
        | F.col("address_id").isNull()
        | F.col("store_id").isNull()
        | F.col("username").isNull()
    ).count()

    print("Duplicate Staff IDs:", duplicate_staff_ids)
    print("Invalid Address references:", invalid_address_refs)
    print("Invalid Store references:", invalid_store_refs)
    print("Invalid mandatory fields:", invalid_mandatory_fields)

    assert source_count == valid_count + rejected_count
    assert duplicate_staff_ids == 0
    assert invalid_address_refs == 0
    assert invalid_store_refs == 0
    assert invalid_mandatory_fields == 0

    writer = SilverWriter(
        silver_base_path="data/processed/silver",
        quarantine_base_path="data/processed/quarantine",
    )

    summary = writer.write(
        valid_df=valid_df,
        rejected_df=rejected_df,
        table_name="staff",
        batch_id="20261122_staff_test_01",
        ingestion_date=date(2026, 11, 22),
    )

    print("\n========== STAFF WRITE SUMMARY ==========")
    print(summary)

    print("\nStaff Silver processing completed successfully.")

finally:
    spark.stop()
