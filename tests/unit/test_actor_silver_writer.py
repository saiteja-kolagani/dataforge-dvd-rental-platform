
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.actor_processor import ActorProcessor
from src.silver.silver_writer import SilverWriter


def main():
    spark = (
        SparkSession.builder
        .appName("DataForge-Actor-Silver-Test")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        # -------------------------------------------------
        # 1. Read raw Actor dataset
        # -------------------------------------------------
        actor_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv("data/raw/actor.csv")
        )

        source_count = actor_df.count()

        print("\n========== ACTOR SILVER PROCESSING ==========")
        print(f"Raw Actor records: {source_count}")

        # -------------------------------------------------
        # 2. Execute Actor processor
        # -------------------------------------------------
        processor = ActorProcessor()

        valid_df, rejected_df = processor.process(actor_df)

        valid_count = valid_df.count()
        rejected_count = rejected_df.count()

        print("\n========== PROCESSING SUMMARY ==========")
        print(f"Source records: {source_count}")
        print(f"Silver records: {valid_count}")
        print(f"Quarantine records: {rejected_count}")
        print(f"Total processed: {valid_count + rejected_count}")

        # -------------------------------------------------
        # 3. Rejection reason analysis
        # -------------------------------------------------
        print("\n========== REJECTION REASONS ==========")

        (
            rejected_df
            .groupBy("rejection_reason")
            .count()
            .orderBy(F.desc("count"))
            .show(truncate=False)
        )

        # -------------------------------------------------
        # 4. Timestamp quality analysis
        # -------------------------------------------------
        print("\n========== TIMESTAMP QUALITY ==========")

        (
            valid_df
            .groupBy("last_update_quality")
            .count()
            .orderBy("last_update_quality")
            .show()
        )

        # -------------------------------------------------
        # 5. Silver sample
        # -------------------------------------------------
        print("\n========== ACTOR SILVER SAMPLE ==========")

        valid_df.orderBy("actor_id").show(
            50,
            truncate=False,
        )

        # -------------------------------------------------
        # 6. Quarantine sample
        # -------------------------------------------------
        print("\n========== ACTOR QUARANTINE SAMPLE ==========")

        rejected_df.orderBy("actor_id").show(
            50,
            truncate=False,
        )

        # -------------------------------------------------
        # 7. Data quality checks
        # -------------------------------------------------
        print("\n========== DATA QUALITY CHECKS ==========")

        duplicate_actor_ids = (
            valid_df
            .groupBy("actor_id")
            .count()
            .where(F.col("count") > 1)
            .count()
        )

        invalid_actor_ids = (
            valid_df
            .where(
                F.col("actor_id").isNull()
                | (F.col("actor_id") <= 0)
            )
            .count()
        )

        missing_first_names = (
            valid_df
            .where(
                F.col("first_name").isNull()
                | (F.trim("first_name") == "")
            )
            .count()
        )

        missing_last_names = (
            valid_df
            .where(
                F.col("last_name").isNull()
                | (F.trim("last_name") == "")
            )
            .count()
        )

        print(f"Duplicate Actor IDs: {duplicate_actor_ids}")
        print(f"Invalid Actor IDs: {invalid_actor_ids}")
        print(f"Missing first names: {missing_first_names}")
        print(f"Missing last names: {missing_last_names}")

        assert valid_count + rejected_count == source_count
        assert duplicate_actor_ids == 0
        assert invalid_actor_ids == 0
        assert missing_first_names == 0
        assert missing_last_names == 0

        # Expected results based on profiling
        assert valid_count == 141, (
            f"Expected 141 Silver records, got {valid_count}"
        )

        assert rejected_count == 19, (
            f"Expected 19 Quarantine records, got {rejected_count}"
        )

        print("\nAll Actor data quality checks passed.")

        # -------------------------------------------------
        # 8. Write Silver and Quarantine datasets
        # -------------------------------------------------
        batch_id = "20261122_actor_test_01"

        writer = SilverWriter(
            silver_base_path="data/processed/silver",
            quarantine_base_path="data/processed/quarantine",
        )

        result = writer.write(
            valid_df=valid_df,
            rejected_df=rejected_df,
            table_name="actor",
            batch_id=batch_id,
            ingestion_date="2026-11-22",
        )

        print("\n========== ACTOR WRITE SUMMARY ==========")
        print(result)

        print("\nActor Silver processing completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
