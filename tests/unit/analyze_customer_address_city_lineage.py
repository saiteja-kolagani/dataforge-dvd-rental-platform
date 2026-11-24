
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = "/mnt/c/dataforge-dvd-rental-platform"

spark = (
    SparkSession.builder
    .appName("Customer-Address-City-Lineage")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Read source data
customer = spark.read.option("header", True).csv(
    f"{ROOT}/data/raw/customer.csv"
)

address_raw = spark.read.option("header", True).csv(
    f"{ROOT}/data/raw/address.csv"
)

address_silver = spark.read.parquet(
    f"{ROOT}/data/processed/silver/address"
)

city_raw = spark.read.option("header", True).csv(
    f"{ROOT}/data/raw/city.csv"
)

city_silver = spark.read.parquet(
    f"{ROOT}/data/processed/silver/city"
)

city_quarantine = spark.read.parquet(
    f"{ROOT}/data/processed/quarantine/city"
)

# Normalize IDs
customer = customer.withColumn(
    "address_id",
    F.expr("try_cast(trim(address_id) AS BIGINT)")
)

address_raw = (
    address_raw
    .withColumn(
        "address_id",
        F.expr("try_cast(trim(address_id) AS BIGINT)")
    )
    .withColumn(
        "city_id",
        F.expr("try_cast(trim(city_id) AS BIGINT)")
    )
)

# Count customers per Address ID
customer_counts = (
    customer
    .filter(
        F.col("address_id").isNotNull()
        & (F.col("address_id") > 0)
    )
    .groupBy("address_id")
    .agg(F.count("*").alias("customer_count"))
)

# Identify only Address IDs missing from Address Silver
silver_address_ids = (
    address_silver
    .select(F.col("address_id").cast("long").alias("address_id"))
    .distinct()
)

missing_customer_addresses = (
    customer_counts
    .join(silver_address_ids, "address_id", "left_anti")
)

# Aggregate raw Address records by ID
# This avoids multiplying Customer counts due to duplicate Address rows.
raw_address_summary = (
    address_raw
    .groupBy("address_id")
    .agg(
        F.count("*").alias("raw_address_versions"),
        F.sort_array(
            F.collect_set("city_id")
        ).alias("city_ids")
    )
)

# City ID reference sets
raw_city_ids = (
    city_raw
    .select(
        F.expr("try_cast(trim(city_id) AS BIGINT)").alias("city_id")
    )
    .filter(F.col("city_id").isNotNull())
    .distinct()
)

silver_city_ids = (
    city_silver
    .select(F.col("city_id").cast("long").alias("city_id"))
    .distinct()
)

quarantined_city_ids = (
    city_quarantine
    .select(
        F.col("city_id").cast("long").alias("city_id"),
        "rejection_reason"
    )
    .groupBy("city_id")
    .agg(
        F.array_join(
            F.sort_array(F.collect_set("rejection_reason")),
            " | "
        ).alias("city_rejection_reason")
    )
)

# Join missing Customer Address IDs to raw Address
address_lineage = (
    missing_customer_addresses
    .join(raw_address_summary, "address_id", "left")
    .withColumn(
        "address_source_status",
        F.when(
            F.col("raw_address_versions").isNull(),
            "NOT_IN_ADDRESS_SOURCE"
        ).otherwise("EXISTS_IN_ADDRESS_SOURCE")
    )
)

print("\n========== CUSTOMER ADDRESS LINEAGE ==========")

address_lineage.select(
    "address_id",
    "customer_count",
    "address_source_status",
    "raw_address_versions",
    "city_ids"
).orderBy("address_id").show(200, truncate=False)

# Expand City IDs for reference validation
city_lineage = (
    address_lineage
    .filter(F.col("city_ids").isNotNull())
    .select(
        "address_id",
        "customer_count",
        F.explode("city_ids").alias("city_id")
    )
    .join(
        raw_city_ids.withColumn("in_city_raw", F.lit(True)),
        "city_id",
        "left"
    )
    .join(
        silver_city_ids.withColumn("in_city_silver", F.lit(True)),
        "city_id",
        "left"
    )
    .join(
        quarantined_city_ids,
        "city_id",
        "left"
    )
    .withColumn(
        "city_status",
        F.when(F.col("in_city_silver") == True, "CITY_IN_SILVER")
        .when(F.col("in_city_raw") == True, "CITY_IN_RAW_BUT_QUARANTINED")
        .otherwise("CITY_NOT_IN_SOURCE")
    )
)

print("\n========== CITY REFERENCES FOR MISSING ADDRESSES ==========")

city_lineage.select(
    "address_id",
    "customer_count",
    "city_id",
    "city_status",
    "city_rejection_reason"
).orderBy("address_id", "city_id").show(200, truncate=False)

print("\n========== CITY STATUS COUNTS ==========")

city_lineage.groupBy("city_status").agg(
    F.countDistinct("city_id").alias("distinct_city_ids"),
    F.countDistinct("address_id").alias("distinct_address_ids")
).show(truncate=False)

print("\n========== CUSTOMER IMPACT BY CITY ID ==========")

city_lineage.groupBy(
    "city_id",
    "city_status",
    "city_rejection_reason"
).agg(
    F.countDistinct("address_id").alias("distinct_address_ids")
).orderBy("city_id").show(200, truncate=False)

print("\n========== ANALYSIS COMPLETED ==========")

spark.stop()
