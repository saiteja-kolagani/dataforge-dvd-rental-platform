
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("InspectStoreStaffSilver")
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

    print("\n========== STORE SILVER ==========")
    store_df.select(
        "store_id",
        "manager_staff_id",
        "manager_staff_reference_quality",
        "address_id",
        "batch_id",
    ).show(truncate=False)

    print("\n========== STAFF SILVER ==========")
    staff_df.select(
        "staff_id",
        "first_name",
        "last_name",
        "store_id",
        "active",
        "batch_id",
    ).show(truncate=False)

    print("\n========== MANAGER REFERENCE CHECK ==========")

    manager_check = store_df.alias("s").join(
        staff_df.select("staff_id").distinct().alias("st"),
        F.col("s.manager_staff_id") == F.col("st.staff_id"),
        "left",
    )

    manager_check.select(
        F.col("s.store_id"),
        F.col("s.manager_staff_id"),
        F.when(
            F.col("st.staff_id").isNotNull(),
            "VALID",
        ).otherwise("UNRESOLVED").alias("manager_reference_status"),
    ).show(truncate=False)

finally:
    spark.stop()
