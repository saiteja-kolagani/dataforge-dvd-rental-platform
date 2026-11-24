
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.silver.language_processor import LanguageProcessor
from src.silver.silver_writer import SilverWriter



def create_spark_session():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    return (
        SparkSession.builder
        .appName("DataForge-Language-Silver-Writer-Test")
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
        source_path = PROJECT_ROOT / "data" / "raw" / "language.csv"

        source_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(str(source_path))
        )

        valid_df, rejected_df = LanguageProcessor().process(source_df)

        output_root = PROJECT_ROOT / "data" / "processed"

        writer = SilverWriter(
            silver_base_path=str(output_root / "silver"),
            quarantine_base_path=str(output_root / "quarantine"),
        )

        batch_id = "20261122_language_test"
        ingestion_date = "2026-11-22"

        result = writer.write(
            valid_df=valid_df,
            rejected_df=rejected_df,
            table_name="language",
            batch_id=batch_id,
            ingestion_date=ingestion_date,
        )

        print("\n--- WRITE SUMMARY ---")
        for key, value in result.items():
            print(f"{key}: {value}")

        print("\n--- VERIFY SILVER PARQUET ---")
        silver_df = spark.read.parquet(result["silver_path"])
        silver_df.show(truncate=False)
        print(f"Silver records: {silver_df.count()}")

        print("\n--- VERIFY QUARANTINE PARQUET ---")
        quarantine_df = spark.read.parquet(result["quarantine_path"])
        quarantine_df.show(truncate=False)
        print(f"Quarantine records: {quarantine_df.count()}")

        assert silver_df.count() == 7, "Unexpected Silver record count"
        assert quarantine_df.count() == 6, "Unexpected Quarantine record count"

        assert "batch_id" in silver_df.columns
        assert "ingestion_date" in silver_df.columns
        assert "raw_last_update" in silver_df.columns

        print("\nSUCCESS: Silver and Quarantine Parquet verification passed.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
