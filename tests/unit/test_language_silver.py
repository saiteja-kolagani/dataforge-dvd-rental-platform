
import os
import sys

from pathlib import Path

# Resolve project root and add it to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Configure PySpark Python worker.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession
from src.silver.language_processor import LanguageProcessor


LANGUAGE_FILE = PROJECT_ROOT / "data" / "raw" / "language.csv"

spark = (
    SparkSession.builder
    .appName("DataForge-Language-Silver-Test")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("\nReading language.csv...")

    source_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("mode", "PERMISSIVE")
        .csv(str(LANGUAGE_FILE))
    )

    print("\n--- SOURCE DATA ---")
    source_df.show(truncate=False)

    processor = LanguageProcessor()
    valid_df, rejected_df = processor.process(source_df)

    print("\n--- VALID SILVER RECORDS ---")
    valid_df.orderBy("language_id").show(truncate=False)

    print("\n--- REJECTED RECORDS ---")
    rejected_df.orderBy("raw_language_id").show(truncate=False)

    print("\n--- PROCESSING SUMMARY ---")
    print("Source records:", source_df.count())
    print("Valid records:", valid_df.count())
    print("Rejected records:", rejected_df.count())

    print("\n--- REJECTION REASONS ---")
    rejected_df.groupBy("rejection_reason").count().show(
        truncate=False
    )

finally:
    spark.stop()
