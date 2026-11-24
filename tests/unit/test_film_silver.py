
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.silver.film_processor import FilmProcessor
from src.silver.silver_writer import SilverWriter


def create_spark_session():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    return (
        SparkSession.builder
        .appName("DataForge-Film-Silver-Test")
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config(
            "spark.hadoop.fs.file.impl",
            "org.apache.hadoop.fs.RawLocalFileSystem",
        )
        .getOrCreate()
    )


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        film_path = PROJECT_ROOT / "data" / "raw" / "film.csv"
        language_path = (
            PROJECT_ROOT / "data" / "processed" / "silver" / "language"
        )

        film_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(str(film_path))
        )

        language_df = spark.read.parquet(str(language_path))

        print("\n--- SOURCE COUNTS ---")
        print(f"Film source records: {film_df.count()}")
        print(f"Language Silver records: {language_df.count()}")

        valid_df, rejected_df = FilmProcessor().process(
            source_df=film_df,
            language_df=language_df,
        )

        print("\n--- PROCESSING SUMMARY ---")
        valid_count = valid_df.count()
        rejected_count = rejected_df.count()

        print(f"Valid Film records: {valid_count}")
        print(f"Rejected Film records: {rejected_count}")
        print(f"Total processed: {valid_count + rejected_count}")

        print("\n--- VALID FILM SAMPLE ---")
        valid_df.orderBy("film_id").show(10, truncate=False)

        print("\n--- REJECTION REASONS ---")
        (
            rejected_df
            .groupBy("rejection_reason")
            .count()
            .orderBy("count", ascending=False)
            .show(truncate=False)
        )

        print("\n--- DUPLICATE FILM CHECK ---")
        (
            valid_df
            .groupBy("film_id")
            .count()
            .filter("count > 1")
            .show()
        )

        # Verify the key business rules.
        assert valid_count + rejected_count == film_df.count()
        assert valid_df.filter("film_id IS NULL").count() == 0
        assert valid_df.filter("film_id <= 0").count() == 0
        assert valid_df.filter(
            "release_year < 2000 OR release_year > 2026"
        ).count() == 0
        assert valid_df.filter(
            "rating NOT IN ('G', 'PG', 'PG-13', 'R', 'NC-17')"
        ).count() == 0

        duplicate_ids = (
            valid_df
            .groupBy("film_id")
            .count()
            .filter("count > 1")
            .count()
        )
        assert duplicate_ids == 0

        # Verify all Film language IDs exist in Language Silver.
        invalid_language_references = (
            valid_df
            .select("language_id")
            .distinct()
            .join(
                language_df.select("language_id").distinct(),
                on="language_id",
                how="left_anti",
            )
            .count()
        )
        assert invalid_language_references == 0

        # Write outputs using a new batch ID.
        output_root = PROJECT_ROOT / "data" / "processed"

        writer = SilverWriter(
            silver_base_path=str(output_root / "silver"),
            quarantine_base_path=str(output_root / "quarantine"),
        )

        result = writer.write(
            valid_df=valid_df,
            rejected_df=rejected_df,
            table_name="film",
            batch_id="20261122_film_test_01",
            ingestion_date="2026-11-22",
        )

        print("\n--- WRITE SUMMARY ---")
        for key, value in result.items():
            print(f"{key}: {value}")

        print("\nSUCCESS: Film Silver processing and validation passed.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
