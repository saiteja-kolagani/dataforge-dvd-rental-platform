
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class CategoryProcessor:
    """
    Cleans and validates DVD Rental category reference data.

    Hard validation:
      - category_id must be a positive integer.
      - name must be present and not a configured null-like value.
      - A category_id with conflicting normalized names is rejected.

    Soft validation:
      - last_update is parsed when possible.
      - Missing or malformed timestamps are flagged, not rejected.

    Duplicate handling:
      - Identical category names: retain one deterministic record.
      - Prefer a record with a valid timestamp.
      - Conflicting normalized names for the same ID: reject the group.
    """

    NULL_VALUES = ["", "NULL", "NA", "N/A", "UNKNOWN"]

    TIMESTAMP_FORMATS = [
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
        "dd-MM-yyyy HH:mm",
        "MM/dd/yyyy",
    ]

    def process(self, source_df: DataFrame):
        required_columns = {
            "category_id",
            "name",
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
                "raw_category_id",
                F.col("category_id").cast("string"),
            )
            .withColumn(
                "raw_name",
                F.col("name").cast("string"),
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
                "category_id_clean",
                F.when(
                    F.upper(F.trim(F.col("raw_category_id"))).isin(
                        self.NULL_VALUES
                    ),
                    F.lit(None),
                ).otherwise(F.trim(F.col("raw_category_id"))),
            )
            .withColumn(
                "name_clean",
                F.when(
                    F.upper(F.trim(F.col("raw_name"))).isin(
                        self.NULL_VALUES
                    ),
                    F.lit(None),
                ).otherwise(F.trim(F.col("raw_name"))),
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

        # Enforce category_id as a positive integer.
        df = df.withColumn(
            "category_id",
            F.when(
                F.col("category_id_clean").rlike(r"^[0-9]+$"),
                F.col("category_id_clean").cast("int"),
            ),
        )

        # Standardize category names.
        df = df.withColumn(
            "name",
            F.initcap(F.col("name_clean")),
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

        # Hard validation: category ID and category name.
        df = df.withColumn(
            "base_rejection_reason",
            F.when(
                F.col("category_id").isNull()
                | (F.col("category_id") <= 0),
                F.lit("INVALID_CATEGORY_ID"),
            )
            .when(
                F.col("name").isNull()
                | (F.length(F.trim(F.col("name"))) == 0),
                F.lit("MISSING_OR_INVALID_CATEGORY_NAME"),
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

        # Find category IDs associated with conflicting names.
        conflicting_ids = (
            valid_candidates
            .groupBy("category_id")
            .agg(
                F.countDistinct("name").alias("distinct_name_count")
            )
            .filter(F.col("distinct_name_count") > 1)
            .select("category_id")
            .withColumn("_has_conflict", F.lit(True))
        )

        candidates_with_conflicts = valid_candidates.join(
            conflicting_ids,
            on="category_id",
            how="left",
        )

        conflicting_rejected = (
            candidates_with_conflicts
            .filter(F.col("_has_conflict") == True)
            .withColumn(
                "rejection_reason",
                F.lit("CONFLICTING_DUPLICATE_CATEGORY_ID"),
            )
        )

        non_conflicting = candidates_with_conflicts.filter(
            F.col("_has_conflict").isNull()
        )

        # For duplicate IDs with the same normalized name,
        # prefer valid timestamps and then use source values
        # for deterministic ordering.
        duplicate_window = (
            Window
            .partitionBy("category_id", "name")
            .orderBy(
                F.when(
                    F.col("last_update_quality") == "VALID",
                    0,
                ).otherwise(1),
                F.col("raw_name").asc_nulls_last(),
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
                "category_id",
                "name",
                "last_update",
                "last_update_quality",
                "raw_category_id",
                "raw_name",
                "raw_last_update",
            )
        )

        duplicate_rejected = (
            ranked
            .filter(F.col("_duplicate_rank") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("DUPLICATE_CATEGORY_RECORD"),
            )
        )

        # Combine all rejected records.
        rejected_df = (
            base_rejected
            .select(
                "category_id",
                "name",
                "last_update",
                "last_update_quality",
                "raw_category_id",
                "raw_name",
                "raw_last_update",
                "rejection_reason",
            )
            .unionByName(
                conflicting_rejected.select(
                    "category_id",
                    "name",
                    "last_update",
                    "last_update_quality",
                    "raw_category_id",
                    "raw_name",
                    "raw_last_update",
                    "rejection_reason",
                )
            )
            .unionByName(
                duplicate_rejected.select(
                    "category_id",
                    "name",
                    "last_update",
                    "last_update_quality",
                    "raw_category_id",
                    "raw_name",
                    "raw_last_update",
                    "rejection_reason",
                )
            )
        )

        return valid_df, rejected_df
