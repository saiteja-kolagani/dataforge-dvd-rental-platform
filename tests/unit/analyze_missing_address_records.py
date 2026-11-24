
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = "/mnt/c/dataforge-dvd-rental-platform"

CUSTOMER_PATH = f"{ROOT}/data/raw/customer.csv"
ADDRESS_RAW_PATH = f"{ROOT}/data/raw/address.csv"
ADDRESS_SILVER_PATH = f"{ROOT}/data/processed/silver/address"
ADDRESS_QUARANTINE_PATH = f"{ROOT}/data/processed/quarantine/address"

spark = (
    SparkSession.builder
    .appName("Missing-Address-Lineage-Analysis")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Read datasets
customer = spark.read.option("header", True).csv(CUSTOMER_PATH)
address_raw = spark.read.option("header", True).csv(ADDRESS_RAW_PATH)
address_silver = spark.read.parquet(ADDRESS_SILVER_PATH)
address_quarantine = spark.read.parquet(ADDRESS_QUARANTINE_PATH)

# Get positive numeric Customer address IDs
customer_ids = (
    customer
    .withColumn(
        "address_id_normalized",
        F.expr("try_cast(trim(address_id) AS BIGINT)")
    )
    .filter(
        F.col("address_id_normalized").isNotNull()
        & (F.col("address_id_normalized") > 0)
    )
)

silver_ids = address_silver.select(
    F.col("address_id").cast("long").alias("address_id")
).distinct()

raw_ids = address_raw.select(
    F.expr("try_cast(trim(address_id) AS BIGINT)").alias("address_id")
).filter(
    F.col("address_id").isNotNull()
).distinct()

quarantine_ids = address_quarantine.select(
    F.col("address_id").cast("long").alias("address_id"),
    "rejection_reason"
).distinct()

# Find positive Customer IDs absent from Address Silver
missing = (
    customer_ids
    .join(
        silver_ids,
        customer_ids.address_id_normalized == silver_ids.address_id,
        "left_anti"
    )
)

# Count by address ID
missing_summary = (
    missing
    .groupBy("address_id_normalized")
    .agg(F.count("*").alias("customer_count"))
)

# Determine whether each missing ID exists in raw Address
raw_check = (
    missing_summary.alias("m")
    .join(
        raw_ids.alias("r"),
        F.col("m.address_id_normalized") == F.col("r.address_id"),
        "left"
    )
    .withColumn(
        "source_status",
        F.when(F.col("r.address_id").isNull(), "NOT_IN_ADDRESS_SOURCE")
        .otherwise("EXISTS_IN_ADDRESS_SOURCE")
    )
)

# Attach quarantine reason when available
lineage = (
    raw_check.alias("m")
    .join(
        quarantine_ids.alias("q"),
        F.col("m.address_id_normalized") == F.col("q.address_id"),
        "left"
    )
    .select(
        F.col("m.address_id_normalized").alias("address_id"),
        "customer_count",
        "source_status",
        "rejection_reason"
    )
    .orderBy("address_id")
)

print("\n========== MISSING ADDRESS LINEAGE ==========")
lineage.show(200, truncate=False)

print("\n========== SUMMARY BY SOURCE STATUS ==========")
raw_check.groupBy("source_status").agg(
    F.sum("customer_count").alias("customer_records"),
    F.count("*").alias("distinct_address_ids")
).show(truncate=False)

print("\n========== SUMMARY BY QUARANTINE REASON ==========")
lineage.groupBy("rejection_reason").agg(
    F.sum("customer_count").alias("customer_records"),
    F.countDistinct("address_id").alias("distinct_address_ids")
).orderBy(F.desc("customer_records")).show(truncate=False)

spark.stop()
