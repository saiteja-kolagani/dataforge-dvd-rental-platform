
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.referential_integrity import (
    ReferentialIntegrityValidator,
)


spark = (
    SparkSession.builder
    .appName("DataForge-Referential-Integrity-Test")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    store_df = spark.read.parquet(
        "data/processed/silver/store"
    )

    staff_df = spark.read.parquet(
        "data/processed/silver/staff"
    )

    validator = ReferentialIntegrityValidator()

    print("\n========== STORE -> STAFF VALIDATION ==========")

    store_staff_report = validator.validate(
        source_df=store_df,
        reference_df=staff_df,
        source_key="manager_staff_id",
        reference_key="staff_id",
        relationship_name="STORE_MANAGER_TO_STAFF",
        reference_status="UNRESOLVED",
    )

    store_staff_report.show(truncate=False)

    print("\n========== STAFF -> STORE VALIDATION ==========")

    staff_store_report = validator.validate(
        source_df=staff_df,
        reference_df=store_df,
        source_key="store_id",
        reference_key="store_id",
        relationship_name="STAFF_TO_STORE",
        reference_status="UNRESOLVED",
    )

    staff_store_report.show(truncate=False)

    print("\n========== COMBINED VALIDATION REPORT ==========")

    combined_report = (
        store_staff_report
        .unionByName(staff_store_report)
    )

    combined_report.show(truncate=False)

    print("\n========== VALIDATION SUMMARY ==========")

    validator.summarize(combined_report).show(
        truncate=False
    )

    print("\n========== DATA QUALITY SUMMARY ==========")

    combined_report.groupBy(
        "validation_status"
    ).count().orderBy(
        "validation_status"
    ).show()

    print("\nReferential integrity validation completed.")

finally:
    spark.stop()
