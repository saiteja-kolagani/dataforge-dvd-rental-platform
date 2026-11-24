
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class CustomerProcessor:
    """
    Customer Silver transformation processor.

    Responsibilities:
    - Normalize customer attributes.
    - Validate mandatory fields and identifiers.
    - Validate Address references.
    - Track provisional Store references.
    - Normalize email and status fields.
    - Validate timestamps.
    - Detect duplicate and conflicting customer IDs.
    - Separate valid and rejected records.
    """

    TIMESTAMP_FORMATS = [
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
        "dd-MM-yyyy HH:mm",
        "dd/MM/yyyy",
        "MM/dd/yyyy",
        "dd-MMM-yyyy",
        "yyyy/MM/dd",
    ]

    NULL_VALUES = ["", "NULL", "NA", "N/A", "NONE", "NAN"]

    TRUE_VALUES = ["Y", "YES", "TRUE", "1", "ACTIVE"]
    FALSE_VALUES = ["N", "NO", "FALSE", "0", "INACTIVE"]

    # --------------------------------------------------
    # Helper methods
    # --------------------------------------------------

    def _normalize_nulls(self, df: DataFrame) -> DataFrame:
        """
        Normalize source columns without modifying raw audit columns.
        """

        for column in df.columns:

            if column.startswith("raw_"):
                continue

            df = df.withColumn(
                column,
                F.when(
                    F.upper(
                        F.trim(F.col(column).cast("string"))
                    ).isin(self.NULL_VALUES),
                    F.lit(None).cast("string"),
                ).otherwise(
                    F.trim(F.col(column).cast("string"))
                )
            )

        return df

    def _is_missing_raw(self, column_name: str):
        """Check whether a raw value represents a missing value."""

        return (
            F.col(column_name).isNull()
            | F.upper(
                F.trim(F.col(column_name).cast("string"))
            ).isin(self.NULL_VALUES)
        )

    def _parse_timestamp(self, column_name: str):
        """Parse timestamps using supported explicit formats."""

        expressions = [
            F.try_to_timestamp(
                F.col(column_name),
                F.lit(timestamp_format)
            )
            for timestamp_format in self.TIMESTAMP_FORMATS
        ]

        return F.coalesce(*expressions)

    def _normalize_boolean(self, column_name: str):
        """Normalize common Boolean representations."""

        value = F.upper(F.trim(F.col(column_name)))

        return (
            F.when(
                value.isin(self.TRUE_VALUES),
                F.lit(True)
            )
            .when(
                value.isin(self.FALSE_VALUES),
                F.lit(False)
            )
            .otherwise(F.lit(None).cast("boolean"))
        )

    # --------------------------------------------------
    # Main processing method
    # --------------------------------------------------

    def process(
        self,
        source_df: DataFrame,
        store_df: DataFrame,
        address_df: DataFrame,
    ) -> tuple[DataFrame, DataFrame]:

        required_columns = {
            "customer_id",
            "store_id",
            "first_name",
            "last_name",
            "email",
            "address_id",
            "activebool",
            "create_date",
            "last_update",
            "active",
        }

        missing_columns = required_columns - set(source_df.columns)

        if missing_columns:
            raise ValueError(
                f"Customer source is missing columns: {sorted(missing_columns)}"
            )

        # --------------------------------------------------
        # 1. Preserve original source values
        # --------------------------------------------------

        df = source_df

        for column in sorted(required_columns):
            df = df.withColumn(
                f"raw_{column}",
                F.col(column).cast("string")
            )

        # Normalize only transformed columns.
        df = self._normalize_nulls(df)

        # --------------------------------------------------
        # 2. Normalize identifiers
        # --------------------------------------------------

        for column in ["customer_id", "store_id", "address_id"]:

            df = df.withColumn(
                f"_{column}_numeric",
                F.expr(f"try_cast({column} AS BIGINT)")
            )

        df = (
            df
            .withColumn("customer_id", F.col("_customer_id_numeric"))
            .withColumn("store_id", F.col("_store_id_numeric"))
            .withColumn("address_id", F.col("_address_id_numeric"))
        )

        # --------------------------------------------------
        # 3. Normalize customer names
        # --------------------------------------------------

        df = (
            df
            .withColumn(
                "first_name",
                F.initcap(F.trim(F.col("first_name")))
            )
            .withColumn(
                "last_name",
                F.initcap(F.trim(F.col("last_name")))
            )
        )

        # --------------------------------------------------
        # 4. Email normalization and quality
        # --------------------------------------------------

        df = df.withColumn(
            "email",
            F.lower(F.trim(F.col("email")))
        )

        email_pattern = (
            r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
            r"@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$"
        )

        df = df.withColumn(
            "email_quality",
            F.when(
                F.col("email").isNull(),
                F.lit("MISSING")
            )
            .when(
                F.col("email").rlike(email_pattern),
                F.lit("VALID")
            )
            .otherwise(F.lit("INVALID"))
        )

        # Invalid optional email is retained as NULL.
        df = df.withColumn(
            "email",
            F.when(
                F.col("email_quality") == "VALID",
                F.col("email")
            ).otherwise(F.lit(None).cast("string"))
        )

        # --------------------------------------------------
        # 5. Normalize active status fields independently
        # --------------------------------------------------

        df = (
            df
            .withColumn(
                "activebool",
                self._normalize_boolean("activebool")
            )
            .withColumn(
                "active",
                self._normalize_boolean("active")
            )
        )

        for column in ["activebool", "active"]:

            raw_column = f"raw_{column}"

            df = df.withColumn(
                f"{column}_quality",
                F.when(
                    self._is_missing_raw(raw_column),
                    F.lit("MISSING")
                )
                .when(
                    F.col(column).isNotNull(),
                    F.lit("VALID")
                )
                .otherwise(F.lit("INVALID"))
            )

        # Compare both status fields without choosing
        # one as the authoritative source.
        df = df.withColumn(
            "active_status_quality",
            F.when(
                F.col("activebool").isNull()
                | F.col("active").isNull(),
                F.lit("NOT_COMPARABLE")
            )
            .when(
                F.col("activebool") != F.col("active"),
                F.lit("ACTIVE_STATUS_MISMATCH")
            )
            .otherwise(F.lit("CONSISTENT"))
        )

        # --------------------------------------------------
        # 6. Parse timestamps
        # --------------------------------------------------

        df = (
            df
            .withColumn(
                "create_date",
                self._parse_timestamp("create_date")
            )
            .withColumn(
                "last_update",
                self._parse_timestamp("last_update")
            )
        )

        df = df.withColumn(
            "create_date_quality",
            F.when(
                F.col("create_date").isNull(),
                F.lit("MISSING_OR_MALFORMED")
            ).otherwise(F.lit("VALID"))
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

        # --------------------------------------------------
        # 7. Validate Store references
        # Store Silver is provisional.
        # Missing positive IDs are UNRESOLVED, not rejected.
        # --------------------------------------------------

        valid_store_ids = (
            store_df
            .select(
                F.expr(
                    "try_cast(store_id AS BIGINT)"
                ).alias("_valid_store_id")
            )
            .filter(F.col("_valid_store_id").isNotNull())
            .dropDuplicates()
        )

        df = df.join(
            F.broadcast(valid_store_ids),
            df.store_id == valid_store_ids._valid_store_id,
            "left"
        )

        df = df.withColumn(
            "store_reference_quality",
            F.when(
                F.col("store_id").isNull()
                | (F.col("store_id") <= 0),
                F.lit("INVALID")
            )
            .when(
                F.col("_valid_store_id").isNotNull(),
                F.lit("VALID")
            )
            .otherwise(F.lit("UNRESOLVED"))
        )

        df = df.drop("_valid_store_id")

        # --------------------------------------------------
        # 8. Validate Address references
        # Address Silver is treated as verified.
        # --------------------------------------------------

        valid_address_ids = (
            address_df
            .select(
                F.expr(
                    "try_cast(address_id AS BIGINT)"
                ).alias("_valid_address_id")
            )
            .filter(F.col("_valid_address_id").isNotNull())
            .dropDuplicates()
        )

        df = df.join(
            F.broadcast(valid_address_ids),
            df.address_id == valid_address_ids._valid_address_id,
            "left"
        )

        df = df.withColumn(
            "address_reference_quality",
            F.when(
                F.col("address_id").isNull()
                | (F.col("address_id") <= 0),
                F.lit("INVALID")
            )
            .when(
                F.col("_valid_address_id").isNotNull(),
                F.lit("VALID")
            )
            .otherwise(F.lit("INVALID"))
        )

        df = df.drop("_valid_address_id")

        # --------------------------------------------------
        # 9. Build mandatory-field rejection reasons
        # --------------------------------------------------

        df = df.withColumn(
            "_rejection_reasons",
            F.array(
                F.when(
                    F.col("customer_id").isNull()
                    | (F.col("customer_id") <= 0),
                    F.lit("INVALID_CUSTOMER_ID")
                ),

                F.when(
                    F.col("store_reference_quality") == "INVALID",
                    F.lit("INVALID_STORE_ID")
                ),

                F.when(
                    F.col("first_name").isNull(),
                    F.lit("MISSING_FIRST_NAME")
                ),

                F.when(
                    F.col("last_name").isNull(),
                    F.lit("MISSING_LAST_NAME")
                ),

                F.when(
                    F.col("address_id").isNull()
                    | (F.col("address_id") <= 0),
                    F.lit("INVALID_ADDRESS_ID")
                ),

                F.when(
                    (F.col("address_reference_quality") == "INVALID")
                    & F.col("address_id").isNotNull()
                    & (F.col("address_id") > 0),
                    F.lit("INVALID_ADDRESS_REFERENCE")
                ),

                F.when(
                    F.col("create_date").isNull(),
                    F.lit("INVALID_CREATE_DATE")
                ),
            )
        )

        df = df.withColumn(
            "_rejection_reasons",
            F.expr(
                "filter(_rejection_reasons, x -> x IS NOT NULL)"
            )
        )

        df = df.withColumn(
            "rejection_reason",
            F.concat_ws("|", F.col("_rejection_reasons"))
        )

        # --------------------------------------------------
        # 10. Separate individually valid candidates
        # --------------------------------------------------

        valid_candidates = df.filter(
            F.size(F.col("_rejection_reasons")) == 0
        )

        rejected_base = df.filter(
            F.size(F.col("_rejection_reasons")) > 0
        )

        # --------------------------------------------------
        # 11. Detect duplicate customer IDs
        # --------------------------------------------------

        business_columns = [
            "store_id",
            "first_name",
            "last_name",
            "email",
            "address_id",
            "activebool",
            "active",
            "create_date",
        ]

        # The base signature excludes only the optional `active` field.
        # It is used exclusively to identify a safe missing-value
        # completion case; all other duplicate comparisons retain
        # the complete business signature.
        base_business_columns = [
            column for column in business_columns
            if column != "active"
        ]

        valid_candidates = (
            valid_candidates
            .withColumn(
                "_business_signature",
                F.to_json(
                    F.struct(*[
                        F.col(column)
                        for column in business_columns
                    ])
                )
            )
            .withColumn(
                "_base_business_signature",
                F.to_json(
                    F.struct(*[
                        F.col(column)
                        for column in base_business_columns
                    ])
                )
            )
        )

        duplicate_groups = (
            valid_candidates
            .groupBy("customer_id")
            .agg(
                F.count("*").alias("_record_count"),
                F.countDistinct("_business_signature").alias(
                    "_business_versions"
                ),
                F.countDistinct("_base_business_signature").alias(
                    "_base_business_versions"
                ),
                F.sum(
                    F.when(
                        F.col("active_quality") == "MISSING",
                        F.lit(1)
                    ).otherwise(F.lit(0))
                ).alias("_missing_active_count"),
                F.sum(
                    F.when(
                        (F.col("active_quality") == "VALID")
                        & F.col("active").isNotNull(),
                        F.lit(1)
                    ).otherwise(F.lit(0))
                ).alias("_valid_active_count"),
                F.countDistinct(
                    F.when(
                        F.col("active_quality") == "VALID",
                        F.col("active")
                    )
                ).alias("_distinct_valid_active_values")
            )
            .withColumn(
                "_status_completion_eligible",
                (F.col("_record_count") > 1)
                & (F.col("_base_business_versions") == 1)
                & (F.col("_missing_active_count") > 0)
                & (F.col("_valid_active_count") > 0)
                & (F.col("_distinct_valid_active_values") == 1)
            )
        )

        valid_candidates = valid_candidates.join(
            duplicate_groups,
            on="customer_id",
            how="left"
        )

        # Controlled survivorship:
        # If all business attributes except `active` are identical,
        # and missing `active` values can be completed unambiguously
        # from one consistent valid value, retain a populated version.
        status_completion_candidates = valid_candidates.filter(
            F.col("_status_completion_eligible")
        )

        regular_candidates = valid_candidates.filter(
            ~F.coalesce(
                F.col("_status_completion_eligible"),
                F.lit(False)
            )
        )

        survivor_window = (
            Window
            .partitionBy("customer_id")
            .orderBy(
                F.when(
                    F.col("active_quality") == "VALID",
                    F.lit(0)
                ).otherwise(F.lit(1)).asc(),
                F.col("last_update").desc_nulls_last(),
                F.col("raw_last_update").asc_nulls_last(),
                F.col("raw_email").asc_nulls_last()
            )
        )

        status_completion_ranked = (
            status_completion_candidates
            .withColumn(
                "_survivor_rank",
                F.row_number().over(survivor_window)
            )
        )

        status_completion_survivors = (
            status_completion_ranked
            .filter(F.col("_survivor_rank") == 1)
        )

        status_completion_extras = (
            status_completion_ranked
            .filter(F.col("_survivor_rank") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("DUPLICATE_CUSTOMER_RECORD")
            )
        )

        # Conflicting valid versions outside the controlled
        # status-completion case: reject every version.
        conflicting = (
            regular_candidates
            .filter(F.col("_business_versions") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("CONFLICTING_DUPLICATE_CUSTOMER_ID")
            )
        )

        # Exact normalized duplicates outside the controlled case:
        # retain one deterministic survivor.
        exact_duplicates = regular_candidates.filter(
            (F.col("_record_count") > 1)
            & (F.col("_business_versions") == 1)
        )

        unique_records = regular_candidates.filter(
            F.col("_record_count") == 1
        )

        exact_ranked = exact_duplicates.withColumn(
            "_survivor_rank",
            F.row_number().over(survivor_window)
        )

        exact_survivors = exact_ranked.filter(
            F.col("_survivor_rank") == 1
        )

        duplicate_extras = (
            exact_ranked
            .filter(F.col("_survivor_rank") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("DUPLICATE_CUSTOMER_RECORD")
            )
        )

        # --------------------------------------------------
        # 12. Build final Silver and Quarantine datasets
        # --------------------------------------------------

        valid_df = (
            unique_records
            .unionByName(exact_survivors, allowMissingColumns=True)
            .unionByName(
                status_completion_survivors,
                allowMissingColumns=True
            )
        )

        rejected_df = (
            rejected_base
            .unionByName(conflicting, allowMissingColumns=True)
            .unionByName(duplicate_extras, allowMissingColumns=True)
            .unionByName(
                status_completion_extras,
                allowMissingColumns=True
            )
        )

        # --------------------------------------------------
        # 13. Remove internal processing columns
        # --------------------------------------------------

        internal_columns = [
            "_customer_id_numeric",
            "_store_id_numeric",
            "_address_id_numeric",
            "_rejection_reasons",
            "_business_signature",
            "_base_business_signature",
            "_record_count",
            "_business_versions",
            "_base_business_versions",
            "_missing_active_count",
            "_valid_active_count",
            "_distinct_valid_active_values",
            "_status_completion_eligible",
            "_survivor_rank",
        ]

        for column in internal_columns:

            if column in valid_df.columns:
                valid_df = valid_df.drop(column)

            if column in rejected_df.columns:
                rejected_df = rejected_df.drop(column)

        # --------------------------------------------------
        # 14. Final Silver projection
        # --------------------------------------------------

        silver_columns = [
            "customer_id",
            "store_id",
            "first_name",
            "last_name",
            "email",
            "email_quality",
            "address_id",
            "activebool",
            "activebool_quality",
            "active",
            "active_quality",
            "active_status_quality",
            "create_date",
            "create_date_quality",
            "last_update",
            "last_update_quality",
            "store_reference_quality",
            "address_reference_quality",
        ] + [
            f"raw_{column}"
            for column in sorted(required_columns)
        ]

        valid_df = valid_df.select(*silver_columns)

        # Quarantine retains transformed and raw fields
        # together with rejection_reason for diagnosis.

        return valid_df, rejected_df
