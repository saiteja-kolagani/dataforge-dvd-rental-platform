
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class ActorProcessor:
    """
    Cleanses raw Actor records.

    Rules:
    - actor_id must be a positive integer.
    - first_name and last_name are mandatory.
    - Names are trimmed and standardized using InitCap.
    - Exact duplicate actor IDs: keep one, quarantine extras.
    - Conflicting names for the same actor_id: quarantine all.
    - Duplicate names across different actor IDs are allowed.
    - last_update is a soft-quality field.
    """

    TIMESTAMP_FORMATS = [
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
        "dd-MMM-yyyy",
        "MM/dd/yyyy",
        "dd/MM/yyyy",
        "yyyy/MM/dd",
        "dd-MM-yyyy HH:mm",
    ]

    NULL_LIKE_VALUES = [
        "",
        "NULL",
        "NA",
        "N/A",
        "UNKNOWN",
    ]

    def _normalize_text(self, column_name: str):
        value = F.trim(F.col(column_name))

        return (
            F.when(
                F.upper(value).isin(self.NULL_LIKE_VALUES),
                F.lit(None).cast("string"),
            )
            .otherwise(value)
        )

    def _parse_timestamp(self, column_name: str):
        raw_timestamp = F.col(column_name)

        parsed_values = [
            F.try_to_timestamp(raw_timestamp, F.lit(fmt))
            for fmt in self.TIMESTAMP_FORMATS
        ]

        return F.coalesce(*parsed_values)

    def process(
        self,
        actor_df: DataFrame,
    ) -> tuple[DataFrame, DataFrame]:

        # -----------------------------------------------------
        # 1. Validate required source columns
        # -----------------------------------------------------
        required_columns = {
            "actor_id",
            "first_name",
            "last_name",
            "last_update",
        }

        missing_columns = required_columns - set(actor_df.columns)

        if missing_columns:
            raise ValueError(
                "Actor dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # -----------------------------------------------------
        # 2. Preserve original source values
        # -----------------------------------------------------
        df = actor_df.select(
            *[
                F.col(column).cast("string").alias(f"raw_{column}")
                for column in actor_df.columns
            ]
        )

        # -----------------------------------------------------
        # 3. Normalize and cast fields
        # -----------------------------------------------------
        df = (
            df
            .withColumn(
                "actor_id_text",
                self._normalize_text("raw_actor_id"),
            )
            .withColumn(
                "first_name_clean",
                self._normalize_text("raw_first_name"),
            )
            .withColumn(
                "last_name_clean",
                self._normalize_text("raw_last_name"),
            )
            .withColumn(
                "last_update_text",
                self._normalize_text("raw_last_update"),
            )
            .withColumn(
                "actor_id",
                F.expr("try_cast(actor_id_text AS INT)"),
            )
            .withColumn(
                "first_name",
                F.initcap(F.trim("first_name_clean")),
            )
            .withColumn(
                "last_name",
                F.initcap(F.trim("last_name_clean")),
            )
            .withColumn(
                "last_update",
                self._parse_timestamp("last_update_text"),
            )
            .withColumn(
                "last_update_quality",
                F.when(
                    F.col("last_update_text").isNull(),
                    F.lit("MISSING"),
                )
                .when(
                    F.col("last_update").isNull(),
                    F.lit("MALFORMED"),
                )
                .otherwise(F.lit("VALID")),
            )
        )

        # -----------------------------------------------------
        # 4. Identify conflicting duplicate Actor IDs
        # -----------------------------------------------------
        actor_id_conflicts = (
            df
            .where(F.col("actor_id").isNotNull())
            .groupBy("actor_id")
            .agg(
                F.countDistinct(
                    F.struct(
                        F.lower(F.trim("first_name_clean")),
                        F.lower(F.trim("last_name_clean")),
                    )
                ).alias("distinct_actor_versions")
            )
            .where(F.col("distinct_actor_versions") > 1)
            .select("actor_id")
            .withColumn("has_actor_id_conflict", F.lit(True))
        )

        df = df.join(
            actor_id_conflicts,
            on="actor_id",
            how="left",
        )

        df = df.withColumn(
            "has_actor_id_conflict",
            F.coalesce(
                F.col("has_actor_id_conflict"),
                F.lit(False),
            ),
        )

        # -----------------------------------------------------
        # 5. Base validation
        # -----------------------------------------------------
        df = df.withColumn(
            "base_rejection_reason",
            F.concat_ws(
                "|",
                F.when(
                    F.col("actor_id").isNull()
                    | (F.col("actor_id") <= 0),
                    F.lit("INVALID_ACTOR_ID"),
                ),
                F.when(
                    F.col("first_name_clean").isNull()
                    | F.col("last_name_clean").isNull(),
                    F.lit("MISSING_OR_INVALID_ACTOR_NAME"),
                ),
            ),
        )

        # -----------------------------------------------------
        # 6. Add duplicate conflict rejection reason
        # -----------------------------------------------------
        df = df.withColumn(
            "rejection_reason",
            F.concat_ws(
                "|",
                F.col("base_rejection_reason"),
                F.when(
                    F.col("has_actor_id_conflict"),
                    F.lit("CONFLICTING_DUPLICATE_ACTOR_ID"),
                ),
            ),
        )

        conflict_rejected = df.where(
            F.col("has_actor_id_conflict")
        )

        base_rejected = df.where(
            (~F.col("has_actor_id_conflict"))
            & (F.col("base_rejection_reason") != "")
        )

        candidates = df.where(
            (~F.col("has_actor_id_conflict"))
            & (F.col("base_rejection_reason") == "")
        )

        # -----------------------------------------------------
        # 7. Resolve exact duplicate Actor records
        # -----------------------------------------------------
        duplicate_window = (
            Window
            .partitionBy("actor_id")
            .orderBy(
                F.when(
                    F.col("last_update_quality") == "VALID",
                    1,
                ).otherwise(0).desc(),
                F.col("last_update").desc_nulls_last(),
                F.col("raw_last_update").desc_nulls_last(),
                F.col("raw_first_name").asc_nulls_last(),
                F.col("raw_last_name").asc_nulls_last(),
            )
        )

        candidates = candidates.withColumn(
            "duplicate_rank",
            F.row_number().over(duplicate_window),
        )

        duplicate_rejected = (
            candidates
            .where(F.col("duplicate_rank") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("DUPLICATE_ACTOR_RECORD"),
            )
            .drop("duplicate_rank")
        )

        valid_df = candidates.where(
            F.col("duplicate_rank") == 1
        )

        # -----------------------------------------------------
        # 8. Combine rejected records
        # -----------------------------------------------------
        rejected_df = (
            base_rejected
            .unionByName(conflict_rejected)
            .unionByName(duplicate_rejected)
        )

        # -----------------------------------------------------
        # 9. Select final output columns
        # -----------------------------------------------------
        final_columns = [
            "actor_id",
            "first_name",
            "last_name",
            "last_update",
            "last_update_quality",
            "raw_actor_id",
            "raw_first_name",
            "raw_last_name",
            "raw_last_update",
        ]

        valid_df = valid_df.select(*final_columns)

        rejected_df = rejected_df.select(
            *final_columns,
            "rejection_reason",
        )

        return valid_df, rejected_df
