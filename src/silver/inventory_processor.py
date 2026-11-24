from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class InventoryProcessor:
    """
    Inventory Silver transformation processor.

    Business rules:
    - inventory_id, film_id and store_id must be positive BIGINT values.
    - Film references are verified against Film Silver.
    - Store Silver is provisional: missing positive references are UNRESOLVED.
    - last_update is optional; missing/malformed values are retained as NULL
      with a quality status.
    - Exact duplicates keep one deterministic record.
    - If duplicate versions differ only by last_update quality and one has a
      valid timestamp, retain the valid-timestamp version.
    - Conflicting valid business versions are all quarantined.
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

    REQUIRED_COLUMNS = {
        "inventory_id",
        "film_id",
        "store_id",
        "last_update",
    }

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
                ).otherwise(F.trim(F.col(column).cast("string")))
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
                F.lit(timestamp_format)
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
        store_df: DataFrame,
    ) -> tuple[DataFrame, DataFrame]:

        missing_columns = self.REQUIRED_COLUMNS - set(source_df.columns)
        if missing_columns:
            raise ValueError(
                "Inventory source is missing columns: "
                f"{sorted(missing_columns)}"
            )

        # 1. Preserve raw source values.
        df = source_df
        for column in sorted(self.REQUIRED_COLUMNS):
            df = df.withColumn(
                f"raw_{column}",
                F.col(column).cast("string")
            )

        df = self._normalize_nulls(df)

        # 2. Normalize identifiers safely.
        for column in ["inventory_id", "film_id", "store_id"]:
            df = df.withColumn(
                column,
                F.expr(f"try_cast(`{column}` AS BIGINT)")
            )

        # 3. Parse optional last_update.
        df = df.withColumn(
            "last_update",
            self._parse_timestamp("last_update")
        )
        df = df.withColumn(
            "last_update_quality",
            F.when(
                self._is_missing_raw("raw_last_update"),
                F.lit("MISSING")
            )
            .when(
                F.col("last_update").isNull(),
                F.lit("MALFORMED")
            )
            .otherwise(F.lit("VALID"))
        )

        # 4. Validate Film reference (verified dependency).
        film_ids = self._reference_ids(
            film_df, "film_id", "_valid_film_id"
        )
        df = df.join(
            F.broadcast(film_ids),
            F.col("film_id") == F.col("_valid_film_id"),
            "left"
        )
        df = df.withColumn(
            "film_reference_quality",
            F.when(
                F.col("film_id").isNull() | (F.col("film_id") <= 0),
                F.lit("INVALID_ID")
            )
            .when(F.col("_valid_film_id").isNotNull(), F.lit("VALID"))
            .otherwise(F.lit("INVALID_REFERENCE"))
        ).drop("_valid_film_id")

        # 5. Validate Store reference (provisional dependency).
        store_ids = self._reference_ids(
            store_df, "store_id", "_valid_store_id"
        )
        df = df.join(
            F.broadcast(store_ids),
            F.col("store_id") == F.col("_valid_store_id"),
            "left"
        )
        df = df.withColumn(
            "store_reference_quality",
            F.when(
                F.col("store_id").isNull() | (F.col("store_id") <= 0),
                F.lit("INVALID_ID")
            )
            .when(F.col("_valid_store_id").isNotNull(), F.lit("VALID"))
            .otherwise(F.lit("UNRESOLVED"))
        ).drop("_valid_store_id")

        # 6. Build row-level rejection reasons.
        df = df.withColumn(
            "_rejection_reasons",
            F.array(
                F.when(
                    F.col("inventory_id").isNull()
                    | (F.col("inventory_id") <= 0),
                    F.lit("INVALID_INVENTORY_ID")
                ),
                F.when(
                    F.col("film_reference_quality") == "INVALID_ID",
                    F.lit("INVALID_FILM_ID")
                ),
                F.when(
                    F.col("film_reference_quality") == "INVALID_REFERENCE",
                    F.lit("INVALID_FILM_REFERENCE")
                ),
                F.when(
                    F.col("store_reference_quality") == "INVALID_ID",
                    F.lit("INVALID_STORE_ID")
                ),
            )
        )
        df = df.withColumn(
            "_rejection_reasons",
            F.expr("filter(_rejection_reasons, x -> x IS NOT NULL)")
        )
        df = df.withColumn(
            "rejection_reason",
            F.concat_ws("|", F.col("_rejection_reasons"))
        )

        valid_candidates = df.filter(F.size("_rejection_reasons") == 0)
        rejected_base = df.filter(F.size("_rejection_reasons") > 0)

        # 7. Duplicate handling is applied after individual row validation.
        # last_update is excluded from the business signature so that a
        # missing/malformed timestamp does not create a business conflict.
        business_columns = ["film_id", "store_id"]
        valid_candidates = valid_candidates.withColumn(
            "_business_signature",
            F.to_json(F.struct(*[F.col(c) for c in business_columns]))
        )

        duplicate_groups = (
            valid_candidates
            .groupBy("inventory_id")
            .agg(
                F.count("*").alias("_record_count"),
                F.countDistinct("_business_signature").alias(
                    "_business_versions"
                ),
                F.sum(
                    F.when(F.col("last_update_quality") == "VALID", 1)
                    .otherwise(0)
                ).alias("_valid_timestamp_count"),
            )
        )

        valid_candidates = valid_candidates.join(
            duplicate_groups,
            on="inventory_id",
            how="left"
        )

        conflicting = (
            valid_candidates
            .filter(F.col("_business_versions") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("CONFLICTING_DUPLICATE_INVENTORY_ID")
            )
        )

        same_business_duplicates = valid_candidates.filter(
            (F.col("_record_count") > 1)
            & (F.col("_business_versions") == 1)
        )

        unique_records = valid_candidates.filter(
            F.col("_record_count") == 1
        )

        # Prefer a valid last_update; then use raw source values as a
        # deterministic tie-breaker. One record survives per ID.
        survivor_window = (
            Window.partitionBy("inventory_id")
            .orderBy(
                F.when(
                    F.col("last_update_quality") == "VALID", 0
                ).otherwise(1).asc(),
                F.col("raw_film_id").asc_nulls_last(),
                F.col("raw_store_id").asc_nulls_last(),
                F.col("raw_last_update").asc_nulls_last(),
            )
        )

        ranked_duplicates = same_business_duplicates.withColumn(
            "_row_number",
            F.row_number().over(survivor_window)
        )

        duplicate_survivors = (
            ranked_duplicates
            .filter(F.col("_row_number") == 1)
            .drop("_row_number")
        )

        duplicate_rejections = (
            ranked_duplicates
            .filter(F.col("_row_number") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("DUPLICATE_INVENTORY_RECORD")
            )
            .drop("_row_number")
        )

        valid_df = unique_records.unionByName(
            duplicate_survivors, allowMissingColumns=True
        )

        rejected_df = (
            rejected_base
            .unionByName(conflicting, allowMissingColumns=True)
            .unionByName(duplicate_rejections, allowMissingColumns=True)
        )

        # 8. Remove internal processing columns, retain audit/DQ columns.
        valid_internal = [c for c in valid_df.columns if c.startswith("_")]
        rejected_internal = [c for c in rejected_df.columns if c.startswith("_")]

        valid_df = valid_df.drop(*valid_internal)
        rejected_df = rejected_df.drop(*rejected_internal)

        return valid_df, rejected_df
