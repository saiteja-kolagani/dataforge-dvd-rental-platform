from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.film_category_processor import FilmCategoryProcessor
from src.silver.silver_writer import SilverWriter


RAW_PATH = "data/raw/film_category.csv"

FILM_PATH = (
    "data/processed/silver/film/"
    "ingestion_date=2026-11-22/"
    "batch_id=20261122_film_test_01"
)

CATEGORY_PATH = (
    "data/processed/silver/category/"
    "ingestion_date=2026-11-22/"
    "batch_id=20261122_category_test_01"
)

SILVER_BASE_PATH = "data/processed/silver"
QUARANTINE_BASE_PATH = "data/processed/quarantine"

BATCH_ID = "20261123_film_category_test_01"
INGESTION_DATE = "2026-11-23"


def assert_zero(df, description):
    count = df.count()
    print(f"{description}: {count}")
    assert count == 0, f"{description} expected 0, found {count}"


def main():
    spark = (
        SparkSession.builder
        .appName("FilmCategorySilverTest")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        source_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(RAW_PATH)
        )

        film_df = spark.read.parquet(FILM_PATH)
        category_df = spark.read.parquet(CATEGORY_PATH)

        print("========== INPUT COUNTS ==========")
        source_count = source_df.count()
        print("Raw Film Category:", source_count)
        print("Film dependency:", film_df.select("film_id").distinct().count())
        print(
            "Category dependency:",
            category_df.select("category_id").distinct().count(),
        )

        processor = FilmCategoryProcessor()
        valid_df, rejected_df = processor.process(
            source_df=source_df,
            film_df=film_df,
            category_df=category_df,
        )

        valid_count = valid_df.count()
        rejected_count = rejected_df.count()

        print("\n========== PROCESSING RESULTS ==========")
        print("Silver candidates:", valid_count)
        print("Quarantine records:", rejected_count)
        print("Total processed:", valid_count + rejected_count)

        assert valid_count + rejected_count == source_count, (
            "Source reconciliation failed"
        )
        print("PASS: Source reconciliation")

        print("\n========== REJECTION REASONS ==========")
        rejected_df.groupBy("rejection_reason").count().orderBy(
            F.desc("count")
        ).show(100, truncate=False)

        print("\n========== TIMESTAMP QUALITY ==========")
        valid_df.groupBy("last_update_quality").count().show()
        rejected_df.groupBy("last_update_quality").count().show()

        print("\n========== SILVER DATA QUALITY ==========")
        assert_zero(
            valid_df.filter(
                F.col("film_id").isNull() | (F.col("film_id") <= 0)
            ),
            "Invalid Film IDs in Silver",
        )
        assert_zero(
            valid_df.filter(
                F.col("category_id").isNull() | (F.col("category_id") <= 0)
            ),
            "Invalid Category IDs in Silver",
        )
        assert_zero(
            valid_df.filter(
                (F.col("film_reference_quality") != "VALID")
                | (F.col("category_reference_quality") != "VALID")
            ),
            "Invalid/unverified references in Silver",
        )

        duplicate_keys = (
            valid_df
            .groupBy("film_id", "category_id")
            .count()
            .filter(F.col("count") > 1)
        )
        assert_zero(duplicate_keys, "Duplicate composite keys in Silver")

        assert_zero(
            valid_df.filter(
                ~F.col("last_update_quality").isin(
                    "VALID", "MISSING", "MALFORMED"
                )
            ),
            "Unexpected timestamp quality values in Silver",
        )

        missing_timestamps = valid_df.filter(
            F.col("last_update_quality") == "MISSING"
        ).count()
        malformed_timestamps = valid_df.filter(
            F.col("last_update_quality") == "MALFORMED"
        ).count()

        print("Missing timestamps retained in Silver:", missing_timestamps)
        print("Malformed timestamps retained in Silver:", malformed_timestamps)

        print("\n========== DUPLICATE RELATIONSHIP REJECTIONS ==========")
        duplicate_rejections = rejected_df.filter(
            F.col("rejection_reason")
            == "DUPLICATE_FILM_CATEGORY_RELATIONSHIP"
        )
        print("Additional duplicate rows quarantined:", duplicate_rejections.count())

        # Write only after all processing and DQ assertions pass.
        writer = SilverWriter(
            silver_base_path=SILVER_BASE_PATH,
            quarantine_base_path=QUARANTINE_BASE_PATH,
        )

        print("\n========== WRITING OUTPUT ==========")
        result = writer.write(
            valid_df=valid_df,
            rejected_df=rejected_df,
            table_name="film_category",
            batch_id=BATCH_ID,
            ingestion_date=INGESTION_DATE,
        )
        print(result)

        print("\n========== FINAL ACCEPTANCE ==========")
        print("PASS: Film Category Silver processing and DQ checks")
        print("Batch ID:", BATCH_ID)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
