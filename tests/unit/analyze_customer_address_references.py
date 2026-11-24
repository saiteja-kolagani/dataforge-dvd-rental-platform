
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_ROOT = "/mnt/c/dataforge-dvd-rental-platform"

CUSTOMER_CSV = f"{PROJECT_ROOT}/data/raw/customer.csv"
ADDRESS_SILVER = f"{PROJECT_ROOT}/data/processed/silver/address"

spark = (
    SparkSession.builder
    .appName("Customer-Address-Reference-Analysis")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# --------------------------------------------------
# 1. Read source Customer records
# --------------------------------------------------

customer_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)
    .csv(CUSTOMER_CSV)
)

customer_df = customer_df.withColumn(
    "normalized_address_id",
    F.expr("try_cast(trim(address_id) AS BIGINT)")
)

# --------------------------------------------------
# 2. Read Address Silver records
# --------------------------------------------------

address_df = spark.read.parquet(ADDRESS_SILVER)

valid_address_ids = (
    address_df
    .select(F.col("address_id").cast("long").alias("address_id"))
    .dropDuplicates()
)

# --------------------------------------------------
# 3. Identify invalid Customer address IDs
# --------------------------------------------------

customer_with_id = customer_df.filter(
    F.col("normalized_address_id").isNotNull()
    & (F.col("normalized_address_id") > 0)
)

missing_address_records = (
    customer_with_id.alias("c")
    .join(
        valid_address_ids.alias("a"),
        F.col("c.normalized_address_id") == F.col("a.address_id"),
        "left_anti"
    )
)

# --------------------------------------------------
# 4. Count records by missing address ID
# --------------------------------------------------

missing_address_summary = (
    missing_address_records
    .groupBy("normalized_address_id")
    .agg(
        F.count("*").alias("customer_record_count")
    )
    .orderBy("normalized_address_id")
)

# --------------------------------------------------
# 5. Identify invalid-format / non-positive IDs
# --------------------------------------------------

invalid_address_ids = customer_df.filter(
    F.col("normalized_address_id").isNull()
    | (F.col("normalized_address_id") <= 0)
)

# --------------------------------------------------
# 6. Print analysis
# --------------------------------------------------

print("\n========== CUSTOMER ADDRESS REFERENCE ANALYSIS ==========")

print(f"Total Customer source records: {customer_df.count()}")
print(f"Total Address Silver records: {address_df.count()}")

print(
    "Distinct Address IDs in Address Silver: "
    f"{valid_address_ids.count()}"
)

print(
    "\nCustomer records with positive numeric address IDs "
    "missing from Address Silver: "
    f"{missing_address_records.count()}"
)

print(
    "Customer records with null, non-numeric or non-positive "
    f"address IDs: {invalid_address_ids.count()}"
)

print("\n========== MISSING ADDRESS ID BREAKDOWN ==========")

missing_address_summary.show(200, truncate=False)

print("\n========== INVALID ADDRESS ID SAMPLE ==========")

invalid_address_ids.select(
    "customer_id",
    "address_id",
    "normalized_address_id"
).show(50, truncate=False)

print("\n========== ANALYSIS COMPLETED ==========")

spark.stop()
