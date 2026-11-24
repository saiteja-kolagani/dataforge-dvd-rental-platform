from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class FilmActorProcessor:
    """Film Actor Silver transformation processor.

    Business rules:
    - actor_id and film_id must be positive BIGINT values.
    - Actor and Film references must exist in their accepted Silver datasets.
    - The composite business key is (actor_id, film_id).
    - Repeated valid relationships keep one deterministic survivor; extra
      copies are quarantined as DUPLICATE_FILM_ACTOR_RELATIONSHIP.
    - last_update is a soft-quality field. Missing/malformed values are kept
      as NULL with a quality status.
    - For duplicate relationships, prefer a valid timestamp, then the latest
      timestamp, then raw source text as a deterministic tie-breaker.
    - Original source values are retained in raw_* columns.
    """

    NULL_VALUES = ["", "NULL", "NA", "N/A", "NONE", "NAN"]

    TIMESTAMP_FORMATS = [
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
        "yyyy/MM/dd",
        "dd-MM-yyyy HH:mm",
        "dd/MM/yyyy",
        "dd-MMM-yyyy",
        "MM/dd/yyyy",
    ]

    REQUIRED_COLUMNS = {"actor_id", "film_id", "last_update"}

    def _normalize_nulls(self, df: DataFrame) -> DataFrame:
        for column in df.columns:
            if column.startswith("raw_"):
                continue
            df = df.withColumn(
                column,
                F.when(
                    F.upper(F.trim(F.col(column).cast("string"))).isin(
                        self.NULL_VALUES
                    ),
                    F.lit(None).cast("string"),
                ).otherwise(F.trim(F.col(column).cast("string"))),
            )
        return df

    def _is_missing_raw(self, column_name: str):
        return (
            F.col(column_name).isNull()
            | F.upper(F.trim(F.col(column_name).cast("string"))).isin(
                self.NULL_VALUES
            )
        )

    def _parse_timestamp(self, column_name: str):
        expressions = [
            F.try_to_timestamp(F.col(column_name), F.lit(timestamp_format))
            for timestamp_format in self.TIMESTAMP_FORMATS
        ]
        return F.coalesce(*expressions)

    def _reference_ids(
        self,
        reference_df: DataFrame,
        key: str,
        alias: str,
    ) -> DataFrame:
        if key not in reference_df.columns:
            raise ValueError(
                f"Reference dataset is missing required key column: {key}"
            )

        return (
            reference_df
            .select(F.expr(f"try_cast(`{key}` AS BIGINT)").alias(alias))
            .filter(F.col(alias).isNotNull() & (F.col(alias) > 0))
            .distinct()
        )

    def process(
        self,
        source_df: DataFrame,
        actor_df: DataFrame,
        film_df: DataFrame,
    ) -> tuple[DataFrame, DataFrame]:
        missing_columns = self.REQUIRED_COLUMNS - set(source_df.columns)
        if missing_columns:
            raise ValueError(
                "Film Actor source is missing columns: "
                f"{sorted(missing_columns)}"
            )

        # 1. Preserve source values for auditability.
        df = source_df
        for column in sorted(self.REQUIRED_COLUMNS):
            df = df.withColumn(
                f"raw_{column}", F.col(column).cast("string")
            )

        df = self._normalize_nulls(df)

        # 2. Safely normalize relationship identifiers.
        for column in ["actor_id", "film_id"]:
            df = df.withColumn(
                column, F.expr(f"try_cast(`{column}` AS BIGINT)")
            )

        # 3. Parse optional last_update and retain its quality status.
        df = df.withColumn(
            "last_update", self._parse_timestamp("last_update")
        )
        df = df.withColumn(
            "last_update_quality",
            F.when(self._is_missing_raw("raw_last_update"), F.lit("MISSING"))
            .when(F.col("last_update").isNull(), F.lit("MALFORMED"))
            .otherwise(F.lit("VALID")),
        )

        # 4. Verify Actor reference.
        actor_ids = self._reference_ids(actor_df, "actor_id", "_valid_actor_id")
        df = df.join(
            F.broadcast(actor_ids),
            F.col("actor_id") == F.col("_valid_actor_id"),
            "left",
        )
        df = df.withColumn(
            "actor_reference_quality",
            F.when(
                F.col("actor_id").isNull() | (F.col("actor_id") <= 0),
                F.lit("INVALID_ID"),
            )
            .when(F.col("_valid_actor_id").isNotNull(), F.lit("VALID"))
            .otherwise(F.lit("INVALID_REFERENCE")),
        ).drop("_valid_actor_id")

        # 5. Verify Film reference.
        film_ids = self._reference_ids(film_df, "film_id", "_valid_film_id")
        df = df.join(
            F.broadcast(film_ids),
            F.col("film_id") == F.col("_valid_film_id"),
            "left",
        )
        df = df.withColumn(
            "film_reference_quality",
            F.when(
                F.col("film_id").isNull() | (F.col("film_id") <= 0),
                F.lit("INVALID_ID"),
            )
            .when(F.col("_valid_film_id").isNotNull(), F.lit("VALID"))
            .otherwise(F.lit("INVALID_REFERENCE")),
        ).drop("_valid_film_id")

        # 6. Build row-level rejection reasons.
        df = df.withColumn(
            "_rejection_reasons",
            F.array(
                F.when(
                    F.col("actor_reference_quality") == "INVALID_ID",
                    F.lit("INVALID_ACTOR_ID"),
                ),
                F.when(
                    F.col("actor_reference_quality") == "INVALID_REFERENCE",
                    F.lit("INVALID_ACTOR_REFERENCE"),
                ),
                F.when(
                    F.col("film_reference_quality") == "INVALID_ID",
                    F.lit("INVALID_FILM_ID"),
                ),
                F.when(
                    F.col("film_reference_quality") == "INVALID_REFERENCE",
                    F.lit("INVALID_FILM_REFERENCE"),
                ),
            ),
        )
        df = df.withColumn(
            "_rejection_reasons",
            F.expr("filter(_rejection_reasons, x -> x IS NOT NULL)"),
        )
        df = df.withColumn(
            "rejection_reason", F.concat_ws("|", F.col("_rejection_reasons"))
        )

        valid_candidates = df.filter(F.size("_rejection_reasons") == 0)
        rejected_base = df.filter(F.size("_rejection_reasons") > 0)

        # 7. Deduplicate valid composite relationships.
        # Timestamp is metadata, not part of the relationship business key.
        relationship_window = (
            Window.partitionBy("actor_id", "film_id")
            .orderBy(
                F.when(F.col("last_update_quality") == "VALID", 0)
                .otherwise(1)
                .asc(),
                F.col("last_update").desc_nulls_last(),
                F.col("raw_last_update").asc_nulls_last(),
            )
        )

        ranked = valid_candidates.withColumn(
            "_row_number", F.row_number().over(relationship_window)
        )

        valid_df = (
            ranked.filter(F.col("_row_number") == 1)
            .drop("_row_number")
        )

        duplicate_rejections = (
            ranked.filter(F.col("_row_number") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("DUPLICATE_FILM_ACTOR_RELATIONSHIP"),
            )
            .drop("_row_number")
        )

        rejected_df = rejected_base.unionByName(
            duplicate_rejections, allowMissingColumns=True
        )

        # 8. Remove internal processing columns; retain DQ/audit columns.
        valid_internal = [c for c in valid_df.columns if c.startswith("_")]
        rejected_internal = [c for c in rejected_df.columns if c.startswith("_")]

        valid_df = valid_df.drop(*valid_internal)
        rejected_df = rejected_df.drop(*rejected_internal)

        return valid_df, rejected_df
