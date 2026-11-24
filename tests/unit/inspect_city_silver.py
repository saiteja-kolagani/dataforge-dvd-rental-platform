
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CITY_SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver" / "city"

spark = (
    SparkSession.builder
    .appName("DataForge-City-Silver-Inspection")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("\n========== CITY SILVER INSPECTION ==========")

    if not CITY_SILVER_PATH.exists():
        raise FileNotFoundError(
            f"City Silver path not found: {CITY_SILVER_PATH}"
        )

    city_df = spark.read.parquet(str(CITY_SILVER_PATH))

    print(f"\nPath: {CITY_SILVER_PATH}")
    print(f"Total records: {city_df.count()}")

    print("\n--- SCHEMA ---")
    city_df.printSchema()

    print("\n--- AVAILABLE CITY IDs ---")
    city_df.select("city_id").orderBy("city_id").show(
        100,
        truncate=False
    )

    print("\n--- DATA QUALITY CHECKS ---")

    duplicate_count = (
        city_df.groupBy("city_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    null_id_count = city_df.filter(
        F.col("city_id").isNull()
    ).count()

    print(f"Duplicate city IDs: {duplicate_count}")
    print(f"Null city IDs: {null_id_count}")

    print("\nCity Silver inspection completed.")

finally:
    spark.stop()
