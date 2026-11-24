
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = "/mnt/c/dataforge-dvd-rental-platform"

spark = (
    SparkSession.builder
    .appName("City-Country-Lineage-Analysis")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Read datasets
city_raw = (
    spark.read.option("header", True)
    .csv(f"{ROOT}/data/raw/city.csv")
    .withColumn(
        "city_id",
        F.expr("try_cast(trim(city_id) AS BIGINT)")
    )
    .withColumn(
        "country_id",
        F.expr("try_cast(trim(country_id) AS BIGINT)")
    )
)

city_quarantine = spark.read.parquet(
    f"{ROOT}/data/processed/quarantine/city"
)

country_raw = (
    spark.read.option("header", True)
    .csv(f"{ROOT}/data/raw/country.csv")
    .withColumn(
        "country_id",
        F.expr("try_cast(trim(country_id) AS BIGINT)")
    )
)

country_silver = spark.read.parquet(
    f"{ROOT}/data/processed/silver/country"
)

country_quarantine = spark.read.parquet(
    f"{ROOT}/data/processed/quarantine/country"
)

# City IDs that were identified in the previous analysis
target_city_ids = [6, 8, 13, 16, 22, 27, 28, 31, 43, 47]

target_cities = (
    city_raw
    .filter(F.col("city_id").isin(target_city_ids))
    .select("city_id", "city", "country_id")
)

# Country reference sets
raw_country_ids = (
    country_raw
    .select("country_id")
    .filter(F.col("country_id").isNotNull())
    .distinct()
)

silver_country_ids = (
    country_silver
    .select(F.col("country_id").cast("long").alias("country_id"))
    .distinct()
)

quarantine_country_ids = (
    country_quarantine
    .select(
        F.col("country_id").cast("long").alias("country_id"),
        "rejection_reason"
    )
    .groupBy("country_id")
    .agg(
        F.array_join(
            F.sort_array(F.collect_set("rejection_reason")),
            " | "
        ).alias("country_rejection_reason")
    )
)

# Join City records to Country datasets
lineage = (
    target_cities
    .join(
        raw_country_ids.withColumn("in_country_raw", F.lit(True)),
        "country_id",
        "left"
    )
    .join(
        silver_country_ids.withColumn("in_country_silver", F.lit(True)),
        "country_id",
        "left"
    )
    .join(
        quarantine_country_ids,
        "country_id",
        "left"
    )
    .withColumn(
        "country_lineage_status",
        F.when(
            F.col("country_id").isNull()
            | (F.col("country_id") <= 0),
            "INVALID_COUNTRY_ID"
        )
        .when(
            F.col("in_country_silver") == True,
            "COUNTRY_IN_SILVER"
        )
        .when(
            F.col("in_country_raw") == True,
            "COUNTRY_IN_RAW_BUT_QUARANTINED"
        )
        .otherwise("COUNTRY_NOT_IN_SOURCE")
    )
)

print("\n========== CITY TO COUNTRY LINEAGE ==========")

lineage.select(
    "city_id",
    "city",
    "country_id",
    "country_lineage_status",
    "country_rejection_reason"
).orderBy("city_id").show(200, truncate=False)

print("\n========== COUNTRY STATUS SUMMARY ==========")

lineage.groupBy("country_lineage_status").agg(
    F.countDistinct("city_id").alias("distinct_city_ids"),
    F.countDistinct("country_id").alias("distinct_country_ids")
).show(truncate=False)

print("\n========== ANALYSIS COMPLETED ==========")

spark.stop()
