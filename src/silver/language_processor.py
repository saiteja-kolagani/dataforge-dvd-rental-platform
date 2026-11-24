
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class LanguageProcessor:
    """
    Cleans and validates the DVD Rental language reference data.

    Hard validation:
      - language_id must be a positive integer.
      - name must be present and not a configured null-like value.
      - A language_id with conflicting normalized names is rejected.

    Soft validation:
      - last_update is parsed when possible.
      - Missing or malformed timestamps are flagged, but do not
        invalidate an otherwise valid language record.
    """

    NULL_VALUES = ["", "NULL", "NA", "N/A", "UNKNOWN"]

    TIMESTAMP_FORMATS = [
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
        "dd-MM-yyyy HH:mm",
    ]

    def process(self, source_df: DataFrame):
        required_columns = {"language_id", "name", "last_update"}
        missing_columns = required_columns - set(source_df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required source columns: {sorted(missing_columns)}"
            )

        # Preserve original source values for traceability.
        df = (
            source_df
            .withColumn("raw_language_id", F.col("language_id").cast("string"))
            .withColumn("raw_name", F.col("name").cast("string"))
            .withColumn("raw_last_update", F.col("last_update").cast("string"))
        )

        # Normalize null-like values and whitespace.
        df = (
            df
            .withColumn(
                "language_id_clean",
                F.when(
                    F.upper(F.trim(F.col("raw_language_id"))).isin(
                        self.NULL_VALUES
                    ),
                    None,
                ).otherwise(F.trim(F.col("raw_language_id"))),
            )
            .withColumn(
                "name_clean",
                F.when(
                    F.upper(F.trim(F.col("raw_name"))).isin(self.NULL_VALUES),
                    None,
                ).otherwise(F.trim(F.col("raw_name"))),
            )
            .withColumn(
                "last_update_clean",
                F.when(
                    F.upper(F.trim(F.col("raw_last_update"))).isin(
                        self.NULL_VALUES
                    ),
                    None,
                ).otherwise(F.trim(F.col("raw_last_update"))),
            )
        )

        # Enforce the language_id data type.
        df = df.withColumn(
            "language_id",
            F.when(
                F.col("language_id_clean").rlike(r"^[0-9]+$"),
                F.col("language_id_clean").cast("int"),
            ),
        )

        # Standardize language names.
        df = df.withColumn(
            "name",
            F.initcap(F.col("name_clean")),
        )

        # Parse timestamps using supported source formats.
        parsed_timestamps = [
            F.try_to_timestamp(
                F.col("last_update_clean"),
                F.lit(timestamp_format),
            )
            for timestamp_format in self.TIMESTAMP_FORMATS
        ]

        df = df.withColumn(
            "last_update",
            F.coalesce(*parsed_timestamps),
        )

        # Timestamp quality is a warning/quality attribute, not a rejection.
        df = df.withColumn(
            "last_update_quality",
            F.when(
                F.col("last_update_clean").isNull(),
                F.lit("MISSING"),
            )
            .when(
                F.col("last_update").isNull(),
                F.lit("MALFORMED"),
            )
            .otherwise(F.lit("VALID")),
        )

        # Hard validation: ID and name.
        df = df.withColumn(
            "base_rejection_reason",
            F.when(
                F.col("language_id").isNull() | (F.col("language_id") <= 0),
                F.lit("INVALID_LANGUAGE_ID"),
            )
            .when(
                F.col("name").isNull(),
                F.lit("MISSING_OR_INVALID_LANGUAGE_NAME"),
            ),
        )

        valid_candidates = df.filter(
            F.col("base_rejection_reason").isNull()
        )

        base_rejected = df.filter(
            F.col("base_rejection_reason").isNotNull()
        ).withColumn(
            "rejection_reason",
            F.col("base_rejection_reason"),
        )

        # Find IDs associated with more than one distinct normalized name.
        conflicting_ids = (
            valid_candidates
            .groupBy("language_id")
            .agg(F.countDistinct("name").alias("distinct_name_count"))
            .filter(F.col("distinct_name_count") > 1)
            .select("language_id")
            .withColumn("has_conflict", F.lit(True))
        )

        candidates_with_conflicts = valid_candidates.join(
            conflicting_ids,
            on="language_id",
            how="left",
        )

        conflicting_rejected = (
            candidates_with_conflicts
            .filter(F.col("has_conflict") == True)
            .withColumn(
                "rejection_reason",
                F.lit("CONFLICTING_DUPLICATE_LANGUAGE_ID"),
            )
        )

        non_conflicting = candidates_with_conflicts.filter(
            F.col("has_conflict").isNull()
        )

        # For identical normalized records, retain one deterministic row.
        duplicate_window = (
            Window
            .partitionBy("language_id", "name")
            .orderBy(
                F.col("raw_name").asc_nulls_last(),
                F.col("raw_last_update").asc_nulls_last(),
            )
        )

        ranked = non_conflicting.withColumn(
            "duplicate_rank",
            F.row_number().over(duplicate_window),
        )

        valid_df = (
            ranked
            .filter(F.col("duplicate_rank") == 1)
            .select(
                "language_id",
                "name",
                "last_update",
                "last_update_quality",
                "raw_language_id",
                "raw_name",
                "raw_last_update",
            )
        )

        duplicate_rejected = (
            ranked
            .filter(F.col("duplicate_rank") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("DUPLICATE_RECORD"),
            )
        )

        # Combine all rejected records with a consistent output schema.
        rejected_df = (
            base_rejected
            .unionByName(
                conflicting_rejected.withColumn(
                    "rejection_reason",
                    F.lit("CONFLICTING_DUPLICATE_LANGUAGE_ID"),
                ),
                allowMissingColumns=True,
            )
            .unionByName(
                duplicate_rejected.withColumn(
                    "rejection_reason",
                    F.lit("DUPLICATE_RECORD"),
                ),
                allowMissingColumns=True,
            )
            .select(
                "language_id",
                "name",
                "last_update",
                "last_update_quality",
                "raw_language_id",
                "raw_name",
                "raw_last_update",
                "rejection_reason",
            )
        )

        return valid_df, rejected_df
