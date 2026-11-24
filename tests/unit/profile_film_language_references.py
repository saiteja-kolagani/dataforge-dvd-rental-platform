
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def create_spark_session():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    return (
        SparkSession.builder
        .appName("DataForge-Film-Language-Reference-Profile")
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.pyspark.python", sys.executable)
        .getOrCreate()
    )


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        film_path = PROJECT_ROOT / "data/raw/film.csv"
        language_path = (
            PROJECT_ROOT / "data/processed/silver/language"
        )

        film_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(str(film_path))
        )

        language_df = spark.read.parquet(str(language_path))

        valid_language_ids = (
            language_df
            .select("language_id")
            .distinct()
        )

        film_language_ids = (
            film_df
            .withColumn(
                "language_id_clean",
                F.expr("try_cast(trim(language_id) AS INT)")
            )
        )

        print("\n--- VALID LANGUAGE SILVER IDs ---")
        valid_language_ids.orderBy("language_id").show()

        print("\n--- FILM LANGUAGE ID DISTRIBUTION ---")
        (
            film_language_ids
            .groupBy("language_id_clean")
            .count()
            .orderBy("language_id_clean")
            .show(50)
        )

        print("\n--- INVALID LANGUAGE REFERENCES ---")
        invalid_references = (
            film_language_ids
            .filter(
                F.col("language_id_clean").isNotNull()
                & (F.col("language_id_clean") > 0)
            )
            .join(
                valid_language_ids,
                film_language_ids.language_id_clean
                == valid_language_ids.language_id,
                "left_anti",
            )
        )

        (
            invalid_references
            .groupBy("language_id_clean")
            .count()
            .orderBy("language_id_clean")
            .show(50)
        )

        print("\n--- INVALID REFERENCE SAMPLE ---")
        invalid_references.select(
            "film_id",
            "title",
            "language_id",
        ).show(30, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
