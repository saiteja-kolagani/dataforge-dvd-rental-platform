
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = "/mnt/c/dataforge-dvd-rental-platform"

CUSTOMER = f"{ROOT}/data/raw/customer.csv"
ADDRESS_RAW = f"{ROOT}/data/raw/address.csv"
ADDRESS_QUARANTINE = f"{ROOT}/data/processed/quarantine/address"
CITY_RAW = f"{ROOT}/data/raw/city.csv"
CITY_SILVER = f"{ROOT}/data/processed/silver/city"

spark = (
    SparkSession.builder
    .appName("Address-City-Lineage-Analysis")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Customer counts by address ID
customer = (
    spark.read.option("header", True).csv(CUSTOMER)
    .withColumn("address_id", F.expr("try_cast(trim(address_id) AS BIGINT)"))
    .filter(F.col("address_id").isNotNull() & (F.col("address_id") > 0))
)

customer_counts = (
    customer.groupBy("address_id")
    .agg(F.count("*").alias("customer_count"))
)

# Address raw records
address_raw = (
    spark.read.option("header", True).csv(ADDRESS_RAW)
    .withColumn("address_id", F.expr("try_cast(trim(address_id) AS BIGINT)"))
    .withColumn("city_id", F.expr("try_cast(trim(city_id) AS BIGINT)"))
)

# Address quarantine records
address_quarantine = (
    spark.read.parquet(ADDRESS_QUARANTINE)
    .withColumn("address_id", F.col("address_id").cast("long"))
    .withColumn("city_id", F.col("city_id").cast("string"))
)

# Aggregate quarantine records by Address ID to avoid
# multiplying Customer counts for duplicate Address records.
quarantine_summary = (
    address_quarantine
    .groupBy("address_id")
    .agg(
        F.array_join(
            F.sort_array(F.collect_set("rejection_reason")),
            " | "
        ).alias("address_rejection_reason"),
        F.array_join(
            F.sort_array(F.collect_set("city_id")),
            ", "
        ).alias("quarantined_city_ids")
    )
)

# City source and City Silver IDs
city_raw = (
    spark.read.option("header", True).csv(CITY_RAW)
    .withColumn("city_id", F.expr("try_cast(trim(city_id) AS BIGINT)"))
    .filter(F.col("city_id").isNotNull())
    .select("city_id").distinct()
)

city_silver = (
    spark.read.parquet(CITY_SILVER)
    .select(F.col("city_id").cast("long").alias("city_id"))
    .distinct()
)

# Attach raw Address city ID to Customer impact
affected_addresses = (
    customer_counts
    .join(address_raw.select("address_id", "city_id"),
          "address_id", "left")
    .join(quarantine_summary, "address_id", "left")
    .join(
        city_raw.withColumn("in_city_raw", F.lit(True)),
        "city_id", "left"
    )
    .join(
        city_silver.withColumn("in_city_silver", F.lit(True)),
        "city_id", "left"
    )
    .withColumn(
        "city_lineage_status",
        F.when(F.col("city_id").isNull(), "MISSING_OR_INVALID_CITY_ID")
        .when(F.col("in_city_silver") == True, "CITY_EXISTS_IN_SILVER")
        .when(F.col("in_city_raw") == True, "CITY_EXISTS_IN_RAW_BUT_QUARANTINED")
        .otherwise("CITY_NOT_IN_SOURCE")
    )
)

print("\n========== ADDRESS TO CITY LINEAGE ==========")

affected_addresses.select(
    "address_id",
    "customer_count",
    "city_id",
    "city_lineage_status",
    "address_rejection_reason"
).orderBy("address_id").show(200, truncate=False)

print("\n========== CUSTOMER IMPACT BY CITY STATUS ==========")

affected_addresses.groupBy("city_lineage_status").agg(
    F.sum("customer_count").alias("customer_records"),
    F.countDistinct("address_id").alias("address_records")
).orderBy(F.desc("customer_records")).show(truncate=False)

print("\n========== CUSTOMER IMPACT BY CITY ID ==========")

affected_addresses.groupBy(
    "city_id", "city_lineage_status"
).agg(
    F.sum("customer_count").alias("customer_records"),
    F.countDistinct("address_id").alias("address_records")
).orderBy("city_id").show(200, truncate=False)

spark.stop()
