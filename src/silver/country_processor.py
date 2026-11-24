
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class CountryProcessor:
    """
    Cleans and validates DVD Rental country reference data.

    Hard validation:
      - country_id must be a positive integer.
      - country must be present and not a configured null-like value.
      - A country_id with conflicting normalized country names is rejected.

    Soft validation:
      - last_update is parsed when possible.
      - Missing or malformed timestamps are flagged, not rejected.

    Duplicate handling:
      - Identical country names: retain one deterministic record.
      - Conflicting country names for the same ID: reject the entire group.
    """

    NULL_VALUES = ["", "NULL", "NA", "N/A", "UNKNOWN"]

    TIMESTAMP_FORMATS = [
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
        "MM/dd/yyyy",
        "yyyy/MM/dd",
    ]

    def process(self, source_df: DataFrame):
        required_columns = {
            "country_id",
            "country",
            "last_update",
        }

        missing_columns = required_columns - set(source_df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required source columns: {sorted(missing_columns)}"
            )

        # Preserve original source values.
        df = (
            source_df
            .withColumn(
                "raw_country_id",
                F.col("country_id").cast("string"),
            )
            .withColumn(
                "raw_country",
                F.col("country").cast("string"),
            )
            .withColumn(
                "raw_last_update",
                F.col("last_update").cast("string"),
            )
        )

        # Normalize null-like values and whitespace.
        df = (
            df
            .withColumn(
                "country_id_clean",
                F.when(
                    F.upper(F.trim(F.col("raw_country_id"))).isin(
                        self.NULL_VALUES
                    ),
                    F.lit(None),
                ).otherwise(F.trim(F.col("raw_country_id"))),
            )
            .withColumn(
                "country_clean",
                F.when(
                    F.upper(F.trim(F.col("raw_country"))).isin(
                        self.NULL_VALUES
                    ),
                    F.lit(None),
                ).otherwise(F.trim(F.col("raw_country"))),
            )
            .withColumn(
                "last_update_clean",
                F.when(
                    F.upper(F.trim(F.col("raw_last_update"))).isin(
                        self.NULL_VALUES
                    ),
                    F.lit(None),
                ).otherwise(F.trim(F.col("raw_last_update"))),
            )
        )

        # Enforce country_id as a positive integer.
        df = df.withColumn(
            "country_id",
            F.when(
                F.col("country_id_clean").rlike(r"^[0-9]+$"),
                F.col("country_id_clean").cast("int"),
            ),
        )

        # Standardize country names.
        df = df.withColumn(
            "country",
            F.initcap(F.col("country_clean")),
        )

        # Parse timestamps using supported formats.
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

        # Timestamp quality is a soft validation attribute.
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

        # Hard validation: country ID and country name.
        df = df.withColumn(
            "base_rejection_reason",
            F.when(
                F.col("country_id").isNull()
                | (F.col("country_id") <= 0),
                F.lit("INVALID_COUNTRY_ID"),
            )
            .when(
                F.col("country").isNull()
                | (F.length(F.trim(F.col("country"))) == 0),
                F.lit("MISSING_OR_INVALID_COUNTRY_NAME"),
            ),
        )

        valid_candidates = df.filter(
            F.col("base_rejection_reason").isNull()
        )

        base_rejected = (
            df
            .filter(F.col("base_rejection_reason").isNotNull())
            .withColumn(
                "rejection_reason",
                F.col("base_rejection_reason"),
            )
        )

        # Detect country IDs associated with conflicting normalized names.
        conflicting_ids = (
            valid_candidates
            .groupBy("country_id")
            .agg(
                F.countDistinct("country").alias("distinct_country_count")
            )
            .filter(F.col("distinct_country_count") > 1)
            .select("country_id")
            .withColumn("_has_conflict", F.lit(True))
        )

        candidates_with_conflicts = valid_candidates.join(
            conflicting_ids,
            on="country_id",
            how="left",
        )

        conflicting_rejected = (
            candidates_with_conflicts
            .filter(F.col("_has_conflict") == True)
            .withColumn(
                "rejection_reason",
                F.lit("CONFLICTING_DUPLICATE_COUNTRY_ID"),
            )
        )

        non_conflicting = candidates_with_conflicts.filter(
            F.col("_has_conflict").isNull()
        )

        # For duplicate IDs with the same normalized country name,
        # retain one deterministic record.
        duplicate_window = (
            Window
            .partitionBy("country_id", "country")
            .orderBy(
                F.when(
                    F.col("last_update_quality") == "VALID",
                    0,
                ).otherwise(1),
                F.col("raw_country").asc_nulls_last(),
                F.col("raw_last_update").asc_nulls_last(),
            )
        )

        ranked = non_conflicting.withColumn(
            "_duplicate_rank",
            F.row_number().over(duplicate_window),
        )

        valid_df = (
            ranked
            .filter(F.col("_duplicate_rank") == 1)
            .select(
                "country_id",
                "country",
                "last_update",
                "last_update_quality",
                "raw_country_id",
                "raw_country",
                "raw_last_update",
            )
        )

        duplicate_rejected = (
            ranked
            .filter(F.col("_duplicate_rank") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("DUPLICATE_COUNTRY_RECORD"),
            )
        )

        # Combine all rejected records.
        rejected_df = (
            base_rejected
            .select(
                "country_id",
                "country",
                "last_update",
                "last_update_quality",
                "raw_country_id",
                "raw_country",
                "raw_last_update",
                "rejection_reason",
            )
            .unionByName(
                conflicting_rejected.select(
                    "country_id",
                    "country",
                    "last_update",
                    "last_update_quality",
                    "raw_country_id",
                    "raw_country",
                    "raw_last_update",
                    "rejection_reason",
                )
            )
            .unionByName(
                duplicate_rejected.select(
                    "country_id",
                    "country",
                    "last_update",
                    "last_update_quality",
                    "raw_country_id",
                    "raw_country",
                    "raw_last_update",
                    "rejection_reason",
                )
            )
        )

        return valid_df, rejected_df
