from datetime import date
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.film_actor_processor import FilmActorProcessor
from src.silver.silver_writer import SilverWriter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILM_ACTOR_PATH = PROJECT_ROOT / "data" / "raw" / "film_actor.csv"
ACTOR_SILVER_BATCH_PATH = (
    PROJECT_ROOT
    / "data" / "processed" / "silver" / "actor"
    / "ingestion_date=2026-11-22"
    / "batch_id=20261122_actor_test_01"
)
FILM_SILVER_BATCH_PATH = (
    PROJECT_ROOT
    / "data" / "processed" / "silver" / "film"
    / "ingestion_date=2026-11-22"
    / "batch_id=20261122_film_test_01"
)

SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver"
QUARANTINE_PATH = PROJECT_ROOT / "data" / "processed" / "quarantine"

BATCH_ID = "20261123_film_actor_test_01"
INGESTION_DATE = date(2026, 11, 23)


def require_paths():
    required_paths = [
        RAW_FILM_ACTOR_PATH,
        ACTOR_SILVER_BATCH_PATH,
        FILM_SILVER_BATCH_PATH,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Film Actor integration test cannot run because these paths are "
            "missing:\n- " + "\n- ".join(missing_paths)
        )


def main():
    require_paths()

    spark = (
        SparkSession.builder
        .appName("DataForge-FilmActor-Silver-Test")
        .master("local[2]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        print("\n========== FILM ACTOR SILVER PROCESSING ==========")

        raw_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(str(RAW_FILM_ACTOR_PATH))
        )
        actor_df = spark.read.parquet(str(ACTOR_SILVER_BATCH_PATH))
        film_df = spark.read.parquet(str(FILM_SILVER_BATCH_PATH))

        source_count = raw_df.count()
        actor_count = actor_df.count()
        film_count = film_df.count()

        print(f"Raw Film Actor records: {source_count}")
        print(f"Actor Silver dependency records: {actor_count}")
        print(f"Film Silver dependency records: {film_count}")

        assert source_count == 648, (
            f"Expected 648 raw Film Actor records; found {source_count}"
        )

        processor = FilmActorProcessor()
        valid_df, rejected_df = processor.process(
            source_df=raw_df,
            actor_df=actor_df,
            film_df=film_df,
        )

        valid_count = valid_df.count()
        rejected_count = rejected_df.count()

        print("\n========== PROCESSING SUMMARY ==========")
        print(f"Source records: {source_count}")
        print(f"Silver records: {valid_count}")
        print(f"Quarantine records: {rejected_count}")
        print(f"Total processed: {valid_count + rejected_count}")

        print("\n========== REJECTION REASONS ==========")
        (
            rejected_df.groupBy("rejection_reason")
            .count()
            .orderBy(F.desc("count"), F.asc("rejection_reason"))
            .show(100, truncate=False)
        )

        print("\n========== FILM ACTOR SILVER SAMPLE ==========")
        valid_df.orderBy("actor_id", "film_id").show(50, truncate=False)

        print("\n========== FILM ACTOR QUARANTINE SAMPLE ==========")
        rejected_df.orderBy("actor_id", "film_id").show(50, truncate=False)

        print("\n========== TIMESTAMP QUALITY ==========")
        valid_df.groupBy("last_update_quality").count().show(truncate=False)

        print("\n========== FOREIGN-KEY QUALITY ==========")
        print("Actor reference quality:")
        valid_df.groupBy("actor_reference_quality").count().show(truncate=False)
        print("Film reference quality:")
        valid_df.groupBy("film_reference_quality").count().show(truncate=False)

        # Silver must contain only positive IDs and verified references.
        invalid_actor_ids = valid_df.filter(
            F.col("actor_id").isNull() | (F.col("actor_id") <= 0)
        ).count()
        invalid_film_ids = valid_df.filter(
            F.col("film_id").isNull() | (F.col("film_id") <= 0)
        ).count()
        invalid_actor_references = valid_df.filter(
            F.col("actor_reference_quality") != "VALID"
        ).count()
        invalid_film_references = valid_df.filter(
            F.col("film_reference_quality") != "VALID"
        ).count()

        duplicate_relationships = (
            valid_df.groupBy("actor_id", "film_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        missing_raw_timestamp_count = valid_df.filter(
            F.col("last_update_quality") == "MISSING"
        ).count()
        malformed_timestamp_count = valid_df.filter(
            F.col("last_update_quality") == "MALFORMED"
        ).count()
        invalid_actor_id_rejections = rejected_df.filter(
            F.array_contains(
                F.split(F.col("rejection_reason"), r"\|"),
                "INVALID_ACTOR_ID",
            )
        ).count()
        invalid_film_id_rejections = rejected_df.filter(
            F.array_contains(
                F.split(F.col("rejection_reason"), r"\|"),
                "INVALID_FILM_ID",
            )
        ).count()
        duplicate_relationship_rejections = rejected_df.filter(
            F.col("rejection_reason") == "DUPLICATE_FILM_ACTOR_RELATIONSHIP"
        ).count()

        print("\n========== DATA QUALITY CHECKS ==========")
        print(f"Invalid Actor IDs in Silver: {invalid_actor_ids}")
        print(f"Invalid Film IDs in Silver: {invalid_film_ids}")
        print(f"Invalid Actor references in Silver: {invalid_actor_references}")
        print(f"Invalid Film references in Silver: {invalid_film_references}")
        print(f"Duplicate composite keys in Silver: {duplicate_relationships}")
        print(f"Missing timestamps retained in Silver: {missing_raw_timestamp_count}")
        print(f"Malformed timestamps retained in Silver: {malformed_timestamp_count}")
        print(f"Rows rejected for invalid Actor IDs: {invalid_actor_id_rejections}")
        print(f"Rows rejected for invalid Film IDs: {invalid_film_id_rejections}")
        print(
            "Additional duplicate relationship rows quarantined: "
            f"{duplicate_relationship_rejections}"
        )

        # Reconciliation and core Silver constraints.
        assert source_count == valid_count + rejected_count, (
            "Source reconciliation failed"
        )
        assert invalid_actor_ids == 0, "Invalid Actor IDs found in Silver"
        assert invalid_film_ids == 0, "Invalid Film IDs found in Silver"
        assert invalid_actor_references == 0, "Unverified Actor references in Silver"
        assert invalid_film_references == 0, "Unverified Film references in Silver"
        assert duplicate_relationships == 0, "Duplicate composite keys found in Silver"

        # Every Silver row must carry the expected quality vocabulary.
        assert valid_df.filter(
            ~F.col("last_update_quality").isin("VALID", "MISSING", "MALFORMED")
        ).count() == 0
        assert valid_df.filter(
            F.col("actor_reference_quality") != "VALID"
        ).count() == 0
        assert valid_df.filter(
            F.col("film_reference_quality") != "VALID"
        ).count() == 0

        # Verify that expected audit columns are present in both outputs.
        expected_audit_columns = {
            "raw_actor_id",
            "raw_film_id",
            "raw_last_update",
            "last_update_quality",
            "actor_reference_quality",
            "film_reference_quality",
            "rejection_reason",
        }
        missing_valid_columns = expected_audit_columns - set(valid_df.columns)
        missing_rejected_columns = expected_audit_columns - set(rejected_df.columns)
        assert not missing_valid_columns, (
            f"Silver output missing audit columns: {sorted(missing_valid_columns)}"
        )
        assert not missing_rejected_columns, (
            "Quarantine output missing audit columns: "
            f"{sorted(missing_rejected_columns)}"
        )

        # Confirm the known source profile's invalid identifier patterns are
        # represented in quarantine. Counts are intentionally not hard-coded
        # because rows may carry more than one rejection reason.
        assert invalid_actor_id_rejections > 0
        assert invalid_film_id_rejections > 0

        print("\n========== FINAL ACCEPTANCE CHECKS ==========")
        print(f"Raw source records: {source_count}")
        print(f"Silver records: {valid_count}")
        print(f"Quarantine records: {rejected_count}")
        print("Source reconciliation: PASS")
        print("Positive identifier checks: PASS")
        print("Actor and Film foreign-key checks: PASS")
        print("Composite-key uniqueness: PASS")
        print("Timestamp quality checks: PASS")
        print("Audit-column checks: PASS")
        print("Invalid-ID quarantine checks: PASS")

        print("\nAll Film Actor Silver acceptance checks passed.")

        # Write only after every processing and acceptance assertion passes.
        writer = SilverWriter(
            silver_base_path=str(SILVER_PATH),
            quarantine_base_path=str(QUARANTINE_PATH),
        )
        result = writer.write(
            valid_df=valid_df,
            rejected_df=rejected_df,
            table_name="film_actor",
            batch_id=BATCH_ID,
            ingestion_date=INGESTION_DATE.isoformat(),
        )

        print("\n========== FILM ACTOR WRITE SUMMARY ==========")
        print(result)
        print("\nFilm Actor Silver processing completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
