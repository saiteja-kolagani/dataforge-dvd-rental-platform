from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class FilmCategoryProcessor:
    """
    Film Category Silver transformation processor.

    Business rules:
    - film_id and category_id must be positive BIGINT values.
    - Film and Category references are verified against their Silver datasets.
    - The natural business key is (film_id, category_id).
    - Repeated valid relationships retain one deterministic survivor;
      additional rows are quarantined.
    - last_update is optional. Missing/malformed values are retained as NULL
      with a timestamp quality status.
    - Prefer a valid timestamp, then the latest timestamp, when selecting a
      survivor for a repeated relationship.
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
    ]

    REQUIRED_COLUMNS = {"film_id", "category_id", "last_update"}

    def _normalize_nulls(self, df: DataFrame) -> DataFrame:
        for column in df.columns:
            if column.startswith("raw_"):
                continue
            df = df.withColumn(
                column,
                F.when(
                    F.col(column).isNull()
                    | F.upper(F.trim(F.col(column).cast("string"))).isin(
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
            F.try_to_timestamp(
                F.col(column_name),
                F.lit(timestamp_format),
            )
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
        film_df: DataFrame,
        category_df: DataFrame,
    ) -> tuple[DataFrame, DataFrame]:

        missing_columns = self.REQUIRED_COLUMNS - set(source_df.columns)
        if missing_columns:
            raise ValueError(
                "Film Category source is missing columns: "
                f"{sorted(missing_columns)}"
            )

        # 1. Preserve original source values for auditability.
        df = source_df
        for column in sorted(self.REQUIRED_COLUMNS):
            df = df.withColumn(
                f"raw_{column}",
                F.col(column).cast("string"),
            )

        df = self._normalize_nulls(df)

        # 2. Normalize identifiers safely.
        for column in ["film_id", "category_id"]:
            df = df.withColumn(
                column,
                F.expr(f"try_cast(`{column}` AS BIGINT)"),
            )

        # 3. Parse optional last_update. Timestamp quality is non-blocking.
        df = df.withColumn(
            "last_update",
            self._parse_timestamp("last_update"),
        )
        df = df.withColumn(
            "last_update_quality",
            F.when(
                self._is_missing_raw("raw_last_update"),
                F.lit("MISSING"),
            )
            .when(
                F.col("last_update").isNull(),
                F.lit("MALFORMED"),
            )
            .otherwise(F.lit("VALID")),
        )

        # 4. Validate Film reference (verified dependency).
        film_ids = self._reference_ids(
            film_df, "film_id", "_valid_film_id"
        )
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

        # 5. Validate Category reference (verified dependency).
        category_ids = self._reference_ids(
            category_df, "category_id", "_valid_category_id"
        )
        df = df.join(
            F.broadcast(category_ids),
            F.col("category_id") == F.col("_valid_category_id"),
            "left",
        )
        df = df.withColumn(
            "category_reference_quality",
            F.when(
                F.col("category_id").isNull() | (F.col("category_id") <= 0),
                F.lit("INVALID_ID"),
            )
            .when(
                F.col("_valid_category_id").isNotNull(),
                F.lit("VALID"),
            )
            .otherwise(F.lit("INVALID_REFERENCE")),
        ).drop("_valid_category_id")

        # 6. Build row-level rejection reasons.
        df = df.withColumn(
            "_rejection_reasons",
            F.array(
                F.when(
                    F.col("film_reference_quality") == "INVALID_ID",
                    F.lit("INVALID_FILM_ID"),
                ),
                F.when(
                    F.col("film_reference_quality") == "INVALID_REFERENCE",
                    F.lit("INVALID_FILM_REFERENCE"),
                ),
                F.when(
                    F.col("category_reference_quality") == "INVALID_ID",
                    F.lit("INVALID_CATEGORY_ID"),
                ),
                F.when(
                    F.col("category_reference_quality") == "INVALID_REFERENCE",
                    F.lit("INVALID_CATEGORY_REFERENCE"),
                ),
            ),
        )
        df = df.withColumn(
            "_rejection_reasons",
            F.expr("filter(_rejection_reasons, x -> x IS NOT NULL)"),
        )
        df = df.withColumn(
            "rejection_reason",
            F.concat_ws("|", F.col("_rejection_reasons")),
        )

        valid_candidates = df.filter(F.size("_rejection_reasons") == 0)
        rejected_base = df.filter(F.size("_rejection_reasons") > 0)

        # 7. Deduplicate only after individual ID and FK validation.
        # The relationship key itself is the complete business identity.
        # last_update is metadata, not a business attribute; differing
        # timestamps therefore do not constitute conflicting relationships.
        duplicate_groups = (
            valid_candidates
            .groupBy("film_id", "category_id")
            .agg(F.count("*").alias("_record_count"))
        )

        valid_candidates = valid_candidates.join(
            duplicate_groups,
            on=["film_id", "category_id"],
            how="left",
        )

        unique_records = valid_candidates.filter(
            F.col("_record_count") == 1
        )

        repeated_records = valid_candidates.filter(
            F.col("_record_count") > 1
        )

        survivor_window = (
            Window.partitionBy("film_id", "category_id")
            .orderBy(
                F.when(
                    F.col("last_update_quality") == "VALID", 0
                ).otherwise(1).asc(),
                F.col("last_update").desc_nulls_last(),
                F.col("raw_last_update").asc_nulls_last(),
            )
        )

        ranked_repeated = repeated_records.withColumn(
            "_row_number",
            F.row_number().over(survivor_window),
        )

        duplicate_survivors = (
            ranked_repeated
            .filter(F.col("_row_number") == 1)
            .drop("_row_number")
        )

        duplicate_rejections = (
            ranked_repeated
            .filter(F.col("_row_number") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("DUPLICATE_FILM_CATEGORY_RELATIONSHIP"),
            )
            .drop("_row_number")
        )

        valid_df = unique_records.unionByName(
            duplicate_survivors,
            allowMissingColumns=True,
        )

        rejected_df = rejected_base.unionByName(
            duplicate_rejections,
            allowMissingColumns=True,
        )

        # 8. Remove internal processing columns; retain DQ/audit columns.
        valid_internal = [c for c in valid_df.columns if c.startswith("_")]
        rejected_internal = [c for c in rejected_df.columns if c.startswith("_")]

        valid_df = valid_df.drop(*valid_internal)
        rejected_df = rejected_df.drop(*rejected_internal)

        return valid_df, rejected_df
