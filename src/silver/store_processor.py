
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class StoreProcessor:
    """
    Processes raw Store records into Silver and Quarantine datasets.

    Staff reference existence is intentionally deferred because Store
    and Staff have a circular dependency.

    Address references are validated against Address Silver.
    """

    REQUIRED_COLUMNS = [
        "store_id",
        "manager_staff_id",
        "address_id",
        "last_update",
    ]

    TIMESTAMP_FORMATS = [
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
        "dd-MM-yyyy HH:mm",
        "MM/dd/yyyy",
        "dd/MM/yyyy",
        "yyyy/MM/dd",
        "dd-MMM-yyyy",
    ]

    def _parse_timestamp(self, column_name: str):
        source = F.col(column_name)
        parsed = None

        for fmt in self.TIMESTAMP_FORMATS:
            candidate = F.try_to_timestamp(source, F.lit(fmt))
            parsed = (
                candidate
                if parsed is None
                else F.coalesce(parsed, candidate)
            )

        return parsed

    def process(
        self,
        raw_df: DataFrame,
        address_silver_df: DataFrame,
    ) -> tuple[DataFrame, DataFrame]:

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in raw_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Store source is missing required columns: {missing_columns}"
            )

        # Preserve original source values.
        df = raw_df

        for column in self.REQUIRED_COLUMNS:
            df = df.withColumn(
                f"raw_{column}",
                F.col(column).cast("string"),
            )

        # Normalize source strings.
        for column in self.REQUIRED_COLUMNS:
            df = df.withColumn(
                f"{column}_clean",
                F.when(
                    F.col(column).isNull()
                    | (F.trim(F.col(column).cast("string")) == "")
                    | (
                        F.upper(
                            F.trim(F.col(column).cast("string"))
                        ).isin("NULL", "NA", "N/A", "NONE")
                    ),
                    F.lit(None),
                ).otherwise(
                    F.trim(F.col(column).cast("string"))
                ),
            )

        # Convert identifiers safely.
        df = (
            df.withColumn(
                "store_id",
                F.expr("try_cast(store_id_clean AS INT)"),
            )
            .withColumn(
                "manager_staff_id",
                F.expr("try_cast(manager_staff_id_clean AS INT)"),
            )
            .withColumn(
                "address_id",
                F.expr("try_cast(address_id_clean AS INT)"),
            )
        )

        # Manager Staff reference quality.
        # Existence validation is deferred until Staff Silver is available.
        df = df.withColumn(
            "manager_staff_reference_quality",
            F.when(
                F.col("manager_staff_id").isNull()
                | (F.col("manager_staff_id") <= 0),
                F.lit("INVALID_ID"),
            ).otherwise(F.lit("UNRESOLVED")),
        )

        # Parse timestamp. Timestamp quality is a soft validation.
        df = df.withColumn(
            "last_update",
            self._parse_timestamp("last_update_clean"),
        )

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

        # Validate Address references.
        valid_address_ids = (
            address_silver_df
            .select(
                F.col("address_id")
                .cast("int")
                .alias("_valid_address_id")
            )
            .filter(F.col("_valid_address_id").isNotNull())
            .distinct()
        )

        df = df.join(
            valid_address_ids,
            df.address_id == valid_address_ids._valid_address_id,
            "left",
        )

        # Individual record validation.
        df = (
            df.withColumn(
                "_invalid_store_id",
                F.col("store_id").isNull()
                | (F.col("store_id") <= 0),
            )
            .withColumn(
                "_invalid_manager_staff_id",
                F.col("manager_staff_id").isNull()
                | (F.col("manager_staff_id") <= 0),
            )
            .withColumn(
                "_invalid_address_id",
                F.col("address_id").isNull()
                | (F.col("address_id") <= 0),
            )
            .withColumn(
                "_invalid_address_reference",
                F.col("address_id").isNotNull()
                & (F.col("address_id") > 0)
                & F.col("_valid_address_id").isNull(),
            )
        )

        df = df.withColumn(
            "_base_rejection_reason",
            F.concat_ws(
                "|",
                F.when(
                    F.col("_invalid_store_id"),
                    F.lit("INVALID_STORE_ID"),
                ),
                F.when(
                    F.col("_invalid_manager_staff_id"),
                    F.lit("INVALID_MANAGER_STAFF_ID"),
                ),
                F.when(
                    F.col("_invalid_address_id"),
                    F.lit("INVALID_ADDRESS_ID"),
                ),
                F.when(
                    F.col("_invalid_address_reference"),
                    F.lit("INVALID_ADDRESS_REFERENCE"),
                ),
            ),
        )

        # Only individually valid records participate in duplicate analysis.
        candidates = df.filter(
            F.col("_base_rejection_reason") == ""
        )

        duplicate_stats = (
            candidates
            .groupBy("store_id")
            .agg(
                F.count("*").alias("_store_record_count"),
                F.countDistinct(
                    F.struct(
                        "manager_staff_id",
                        "address_id",
                    )
                ).alias("_business_version_count"),
            )
        )

        candidates = candidates.join(
            duplicate_stats,
            on="store_id",
            how="left",
        )

        # Conflicting valid versions: reject all versions.
        survivor_window = (
            Window
            .partitionBy("store_id")
            .orderBy(
                F.col("last_update").desc_nulls_last(),
                F.col("raw_last_update").desc_nulls_last(),
                F.col("raw_manager_staff_id").asc_nulls_last(),
                F.col("raw_address_id").asc_nulls_last(),
            )
        )

        candidates = candidates.withColumn(
            "_duplicate_rank",
            F.row_number().over(survivor_window),
        )

        candidates = candidates.withColumn(
            "_duplicate_rejection_reason",
            F.when(
                F.col("_business_version_count") > 1,
                F.lit("CONFLICTING_DUPLICATE_STORE_ID"),
            )
            .when(
                (F.col("_store_record_count") > 1)
                & (F.col("_duplicate_rank") > 1),
                F.lit("DUPLICATE_STORE_RECORD"),
            )
            .otherwise(F.lit("")),
        )

        # Combine individually invalid records and duplicate results.
        invalid_records = (
            df.filter(F.col("_base_rejection_reason") != "")
            .withColumn(
                "rejection_reason",
                F.col("_base_rejection_reason"),
            )
        )

        candidates = candidates.withColumn(
            "rejection_reason",
            F.col("_duplicate_rejection_reason"),
        )

        processed_df = invalid_records.unionByName(
            candidates,
            allowMissingColumns=True,
        )

        # Final output columns.
        output_columns = [
            "store_id",
            "manager_staff_id",
            "manager_staff_reference_quality",
            "address_id",
            "last_update",
            "last_update_quality",
            "raw_store_id",
            "raw_manager_staff_id",
            "raw_address_id",
            "raw_last_update",
            "rejection_reason",
        ]

        processed_df = processed_df.select(*output_columns)

        valid_df = (
            processed_df
            .filter(F.col("rejection_reason") == "")
            .drop("rejection_reason")
        )

        rejected_df = processed_df.filter(
            F.col("rejection_reason") != ""
        )

        return valid_df, rejected_df
