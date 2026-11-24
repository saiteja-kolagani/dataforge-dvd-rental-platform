
from datetime import date

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.store_processor import StoreProcessor
from src.silver.silver_writer import SilverWriter


spark = (
    SparkSession.builder
    .appName("DataForge-Store-Silver-Test")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    raw_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv("data/raw/store.csv")
    )

    address_silver_df = spark.read.parquet(
        "data/processed/silver/address"
    )

    processor = StoreProcessor()

    valid_df, rejected_df = processor.process(
        raw_df,
        address_silver_df,
    )

    source_count = raw_df.count()
    valid_count = valid_df.count()
    rejected_count = rejected_df.count()

    print("\n========== STORE SILVER PROCESSING ==========")
    print("Source records:", source_count)
    print("Silver records:", valid_count)
    print("Quarantine records:", rejected_count)
    print("Total processed:", valid_count + rejected_count)

    print("\n========== SILVER RECORDS ==========")
    valid_df.orderBy("store_id").show(truncate=False)

    print("\n========== QUARANTINE RECORDS ==========")
    rejected_df.orderBy("store_id").show(truncate=False)

    print("\n========== REJECTION REASONS ==========")
    (
        rejected_df
        .groupBy("rejection_reason")
        .count()
        .orderBy(F.desc("count"))
        .show(truncate=False)
    )

    print("\n========== DATA QUALITY CHECKS ==========")

    duplicate_store_ids = (
        valid_df
        .groupBy("store_id")
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

    invalid_ids = valid_df.filter(
        (F.col("store_id").isNull())
        | (F.col("store_id") <= 0)
        | (F.col("manager_staff_id").isNull())
        | (F.col("manager_staff_id") <= 0)
        | (F.col("address_id").isNull())
        | (F.col("address_id") <= 0)
    ).count()

    print("Duplicate Store IDs:", duplicate_store_ids)
    print("Invalid Address references:", invalid_address_refs)
    print("Invalid mandatory IDs:", invalid_ids)

    assert source_count == valid_count + rejected_count
    assert duplicate_store_ids == 0
    assert invalid_address_refs == 0
    assert invalid_ids == 0

    writer = SilverWriter(
        silver_base_path="data/processed/silver",
        quarantine_base_path="data/processed/quarantine",
    )

    summary = writer.write(
        valid_df=valid_df,
        rejected_df=rejected_df,
        table_name="store",
        batch_id="20261122_store_test_01",
        ingestion_date=date(2026, 11, 22),
    )

    print("\n========== STORE WRITE SUMMARY ==========")
    print(summary)

    print("\nStore Silver processing completed successfully.")

finally:
    spark.stop()
