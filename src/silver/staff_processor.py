
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class StaffProcessor:
    """
    Processes raw Staff records into Silver and Quarantine.

    Store references are checked against the provisional Store Silver
    dataset. Final circular referential-integrity checks are deferred.

    Passwords are checked for presence but never included in output.
    """

    REQUIRED_COLUMNS = [
        "staff_id",
        "first_name",
        "last_name",
        "address_id",
        "email",
        "store_id",
        "active",
        "username",
        "password",
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

    ACTIVE_TRUE_VALUES = ["Y", "YES", "TRUE", "1", "ACTIVE"]
    ACTIVE_FALSE_VALUES = ["N", "NO", "FALSE", "0", "INACTIVE"]

    def _parse_timestamp(self, column_name: str):
        source = F.col(column_name)
        parsed = None

        for fmt in self.TIMESTAMP_FORMATS:
            candidate = F.try_to_timestamp(source, F.lit(fmt))
            parsed = candidate if parsed is None else F.coalesce(
                parsed, candidate
            )

        return parsed

    def process(
        self,
        raw_df: DataFrame,
        address_silver_df: DataFrame,
        store_silver_df: DataFrame,
    ) -> tuple[DataFrame, DataFrame]:

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in raw_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Staff source is missing required columns: {missing_columns}"
            )

        df = raw_df

        # Preserve source fields, except password.
        # Never create raw_password or expose password in output.
        source_columns_to_preserve = [
            column for column in raw_df.columns
            if column.lower() != "password"
        ]

        for column in source_columns_to_preserve:
            df = df.withColumn(
                f"raw_{column}",
                F.col(column).cast("string"),
            )

        # Capture password presence without retaining its value.
        df = df.withColumn(
            "_password_present",
            F.when(
                F.col("password").isNull()
                | (
                    F.trim(F.col("password").cast("string")) == ""
                )
                | (
                    F.upper(F.trim(F.col("password").cast("string"))).isin(
                        "NULL", "NA", "N/A", "NONE"
                    )
                ),
                F.lit(False),
            ).otherwise(F.lit(True)),
        )

        # Normalize source values.
        fields_to_clean = [
            "staff_id",
            "first_name",
            "last_name",
            "address_id",
            "email",
            "store_id",
            "active",
            "username",
            "last_update",
        ]

        for column in fields_to_clean:
            df = df.withColumn(
                f"{column}_clean",
                F.when(
                    F.col(column).isNull()
                    | (
                        F.trim(F.col(column).cast("string")) == ""
                    )
                    | (
                        F.upper(F.trim(F.col(column).cast("string"))).isin(
                            "NULL", "NA", "N/A", "NONE"
                        )
                    ),
                    F.lit(None),
                ).otherwise(
                    F.trim(F.col(column).cast("string"))
                ),
            )

        # Normalize names.
        df = (
            df.withColumn(
                "first_name",
                F.initcap(F.col("first_name_clean")),
            )
            .withColumn(
                "last_name",
                F.initcap(F.col("last_name_clean")),
            )
        )

        # Normalize identifiers.
        df = (
            df.withColumn(
                "staff_id",
                F.expr("try_cast(staff_id_clean AS INT)"),
            )
            .withColumn(
                "address_id",
                F.expr("try_cast(address_id_clean AS INT)"),
            )
            .withColumn(
                "store_id",
                F.expr("try_cast(store_id_clean AS INT)"),
            )
        )

        # Normalize username.
        df = df.withColumn(
            "username",
            F.lower(F.col("username_clean")),
        )

        # Email quality: invalid email is a soft issue.
        email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

        df = df.withColumn(
            "email_quality",
            F.when(
                F.col("email_clean").isNull(),
                F.lit("MISSING"),
            )
            .when(
                ~F.col("email_clean").rlike(email_pattern),
                F.lit("INVALID"),
            )
            .otherwise(F.lit("VALID")),
        )

        df = df.withColumn(
            "email",
            F.when(
                F.col("email_quality") == "VALID",
                F.lower(F.col("email_clean")),
            ).otherwise(F.lit(None).cast("string")),
        )

        # Normalize active status.
        active_value = F.upper(F.col("active_clean"))

        df = (
            df.withColumn(
                "active",
                F.when(
                    active_value.isin(self.ACTIVE_TRUE_VALUES),
                    F.lit(True),
                )
                .when(
                    active_value.isin(self.ACTIVE_FALSE_VALUES),
                    F.lit(False),
                )
                .otherwise(F.lit(None).cast("boolean")),
            )
            .withColumn(
                "active_quality",
                F.when(
                    F.col("active_clean").isNull(),
                    F.lit("MISSING"),
                )
                .when(
                    active_value.isin(
                        self.ACTIVE_TRUE_VALUES
                        + self.ACTIVE_FALSE_VALUES
                    ),
                    F.lit("VALID"),
                )
                .otherwise(F.lit("INVALID")),
            )
        )

        # Timestamp quality is a soft validation.
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

        # Picture is optional. Preserve it as a string when supplied.
        if "picture" in df.columns:
            df = df.withColumn(
                "picture",
                F.when(
                    F.col("picture").isNull()
                    | (
                        F.trim(F.col("picture").cast("string")) == ""
                    ),
                    F.lit(None).cast("string"),
                ).otherwise(
                    F.col("picture").cast("string")
                ),
            )
        else:
            df = df.withColumn(
                "picture",
                F.lit(None).cast("string"),
            )

        # Reference data.
        valid_address_ids = (
            address_silver_df
            .select(
                F.col("address_id").cast("int").alias(
                    "_valid_address_id"
                )
            )
            .filter(F.col("_valid_address_id").isNotNull())
            .distinct()
        )

        valid_store_ids = (
            store_silver_df
            .select(
                F.col("store_id").cast("int").alias(
                    "_valid_store_id"
                )
            )
            .filter(F.col("_valid_store_id").isNotNull())
            .distinct()
        )

        df = (
            df.join(
                valid_address_ids,
                df.address_id == valid_address_ids._valid_address_id,
                "left",
            )
            .join(
                valid_store_ids,
                df.store_id == valid_store_ids._valid_store_id,
                "left",
            )
        )

        # Individual record validation.
        df = (
            df.withColumn(
                "_invalid_staff_id",
                F.col("staff_id").isNull()
                | (F.col("staff_id") <= 0),
            )
            .withColumn(
                "_missing_first_name",
                F.col("first_name").isNull(),
            )
            .withColumn(
                "_missing_last_name",
                F.col("last_name").isNull(),
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
            .withColumn(
                "_invalid_store_id",
                F.col("store_id").isNull()
                | (F.col("store_id") <= 0),
            )
            .withColumn(
                "_invalid_store_reference",
                F.col("store_id").isNotNull()
                & (F.col("store_id") > 0)
                & F.col("_valid_store_id").isNull(),
            )
            .withColumn(
                "_missing_username",
                F.col("username").isNull(),
            )
            .withColumn(
                "_missing_password",
                ~F.col("_password_present"),
            )
        )

        df = df.withColumn(
            "_base_rejection_reason",
            F.concat_ws(
                "|",
                F.when(
                    F.col("_invalid_staff_id"),
                    F.lit("INVALID_STAFF_ID"),
                ),
                F.when(
                    F.col("_missing_first_name"),
                    F.lit("MISSING_FIRST_NAME"),
                ),
                F.when(
                    F.col("_missing_last_name"),
                    F.lit("MISSING_LAST_NAME"),
                ),
                F.when(
                    F.col("_invalid_address_id"),
                    F.lit("INVALID_ADDRESS_ID"),
                ),
                F.when(
                    F.col("_invalid_address_reference"),
                    F.lit("INVALID_ADDRESS_REFERENCE"),
                ),
                F.when(
                    F.col("_invalid_store_id"),
                    F.lit("INVALID_STORE_ID"),
                ),
                F.when(
                    F.col("_invalid_store_reference"),
                    F.lit("INVALID_STORE_REFERENCE"),
                ),
                F.when(
                    F.col("_missing_username"),
                    F.lit("MISSING_USERNAME"),
                ),
                F.when(
                    F.col("_missing_password"),
                    F.lit("MISSING_PASSWORD"),
                ),
            ),
        )

        # Duplicate analysis only for individually valid candidates.
        candidates = df.filter(
            F.col("_base_rejection_reason") == ""
        )

        business_attributes = [
            "first_name",
            "last_name",
            "address_id",
            "email",
            "store_id",
            "active",
            "username",
        ]

        duplicate_stats = (
            candidates
            .groupBy("staff_id")
            .agg(
                F.count("*").alias("_staff_record_count"),
                F.countDistinct(
                    F.struct(*business_attributes)
                ).alias("_business_version_count"),
            )
        )

        candidates = candidates.join(
            duplicate_stats,
            on="staff_id",
            how="left",
        )

        survivor_window = (
            Window
            .partitionBy("staff_id")
            .orderBy(
                F.col("last_update").desc_nulls_last(),
                F.col("raw_last_update").desc_nulls_last(),
                F.col("raw_username").asc_nulls_last(),
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
                F.lit("CONFLICTING_DUPLICATE_STAFF_ID"),
            )
            .when(
                (F.col("_staff_record_count") > 1)
                & (F.col("_duplicate_rank") > 1),
                F.lit("DUPLICATE_STAFF_RECORD"),
            )
            .otherwise(F.lit("")),
        )

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

        # Password and all password-derived fields are excluded.
        output_columns = [
            "staff_id",
            "first_name",
            "last_name",
            "address_id",
            "email",
            "email_quality",
            "store_id",
            "active",
            "active_quality",
            "username",
            "last_update",
            "last_update_quality",
            "picture",
            "raw_staff_id",
            "raw_first_name",
            "raw_last_name",
            "raw_address_id",
            "raw_email",
            "raw_store_id",
            "raw_active",
            "raw_username",
            "raw_last_update",
            "raw_picture",
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
