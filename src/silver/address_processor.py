
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class AddressProcessor:

    REQUIRED_COLUMNS = [
        "address_id",
        "address",
        "address2",
        "district",
        "city_id",
        "postal_code",
        "phone",
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

    def _clean_nulls(self, column):
        value = F.trim(F.col(column))

        return F.when(
            F.upper(value).isin(
                "", "NULL", "NA", "N/A", "NONE", "UNKNOWN"
            ),
            F.lit(None)
        ).otherwise(value)

    def _parse_timestamp(self, column):
        value = F.col(column)

        parsed_values = [
            F.try_to_timestamp(
                value,
                F.lit(fmt)
            )
            for fmt in self.TIMESTAMP_FORMATS
        ]

        return F.coalesce(*parsed_values)

    def process(
        self,
        raw_df: DataFrame,
        city_silver_df: DataFrame,
    ) -> tuple[DataFrame, DataFrame]:

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in raw_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Address dataset missing columns: {missing_columns}"
            )

        if "city_id" not in city_silver_df.columns:
            raise ValueError(
                "City Silver dataset must contain city_id"
            )

        # --------------------------------------------------
        # 1. Preserve original source values
        # --------------------------------------------------

        df = raw_df

        for column in self.REQUIRED_COLUMNS:
            df = df.withColumn(
                f"raw_{column}",
                F.col(column).cast("string")
            )

        # --------------------------------------------------
        # 2. Normalize source values
        # --------------------------------------------------

        df = (
            df
            .withColumn(
                "address_id",
                F.col("address_id").cast("string")
            )
            .withColumn(
                "address_id_clean",
                self._clean_nulls("address_id")
            )
            .withColumn(
                "address_id",
                F.expr("try_cast(address_id_clean AS INT)")
            )
            .withColumn(
                "address",
                F.trim(self._clean_nulls("address"))
            )
            .withColumn(
                "address2",
                F.initcap(
                    F.trim(self._clean_nulls("address2"))
                )
            )
            .withColumn(
                "district",
                F.initcap(
                    F.trim(self._clean_nulls("district"))
                )
            )
            .withColumn(
                "city_id_clean",
                self._clean_nulls("city_id")
            )
            .withColumn(
                "city_id",
                F.expr("try_cast(city_id_clean AS INT)")
            )
            .withColumn(
                "postal_code",
                F.upper(
                    F.regexp_replace(
                        F.trim(self._clean_nulls("postal_code")),
                        r"\s+",
                        " "
                    )
                )
            )
            .withColumn(
                "phone",
                F.trim(self._clean_nulls("phone"))
            )
            .withColumn(
                "last_update_clean",
                self._clean_nulls("last_update")
            )
        )

        # --------------------------------------------------
        # 3. Timestamp parsing and quality
        # --------------------------------------------------

        df = df.withColumn(
            "last_update",
            self._parse_timestamp("last_update_clean")
        )

        df = df.withColumn(
            "last_update_quality",
            F.when(
                F.col("last_update_clean").isNull(),
                "MISSING"
            )
            .when(
                F.col("last_update").isNull(),
                "MALFORMED"
            )
            .otherwise("VALID")
        )

        # --------------------------------------------------
        # 4. Optional field quality checks
        # --------------------------------------------------

        df = df.withColumn(
            "postal_code_quality",
            F.when(
                F.col("postal_code").isNull(),
                "MISSING"
            ).otherwise("PRESENT")
        )

        phone_digits = F.regexp_replace(
            F.col("phone"),
            r"[^0-9]",
            ""
        )

        phone_valid = (
            F.col("phone").rlike(r"^[+]?[0-9() .-]+$")
            & (F.length(phone_digits) >= 7)
            & (F.length(phone_digits) <= 15)
        )

        df = df.withColumn(
            "phone_quality",
            F.when(
                F.col("phone").isNull(),
                "MISSING"
            )
            .when(
                phone_valid,
                "VALID"
            )
            .otherwise("INVALID")
        )

        # Invalid optional phone values are retained as raw data,
        # but the normalized Silver phone is set to NULL.
        df = df.withColumn(
            "phone",
            F.when(
                F.col("phone_quality") == "INVALID",
                F.lit(None)
            ).otherwise(F.col("phone"))
        )

        # --------------------------------------------------
        # 5. Base validation
        # --------------------------------------------------

        df = df.withColumn(
            "rejection_reason",
            F.concat_ws(
                "|",

                F.when(
                    F.col("address_id").isNull()
                    | (F.col("address_id") <= 0),
                    "INVALID_ADDRESS_ID"
                ),

                F.when(
                    F.col("address").isNull()
                    | (F.length(F.col("address")) == 0),
                    "MISSING_ADDRESS"
                ),

                F.when(
                    F.col("district").isNull()
                    | (F.length(F.col("district")) == 0)
                    | (F.lower(F.col("district")) == "unknowndistrict"),
                    "MISSING_OR_INVALID_DISTRICT"
                ),

                F.when(
                    F.col("city_id").isNull()
                    | (F.col("city_id") <= 0),
                    "INVALID_CITY_ID"
                ),
            )
        )

        # --------------------------------------------------
        # 6. City referential integrity
        # --------------------------------------------------

        valid_city_ids = (
            city_silver_df
            .select(
                F.col("city_id").cast("int").alias("reference_city_id")
            )
            .filter(F.col("reference_city_id").isNotNull())
            .distinct()
        )

        df = df.join(
            F.broadcast(valid_city_ids),
            F.col("city_id") == F.col("reference_city_id"),
            "left"
        )

        df = df.withColumn(
            "rejection_reason",
            F.concat_ws(
                "|",
                F.col("rejection_reason"),
                F.when(
                    F.col("city_id").isNotNull()
                    & (F.col("city_id") > 0)
                    & F.col("reference_city_id").isNull(),
                    "INVALID_CITY_REFERENCE"
                )
            )
        ).drop("reference_city_id")

        # --------------------------------------------------
        # 7. Duplicate address ID analysis
        # --------------------------------------------------

        business_columns = [
            "address",
            "address2",
            "district",
            "city_id",
            "postal_code",
            "phone",
        ]

        duplicate_window = (
            Window
            .partitionBy("address_id")
        )

        df = df.withColumn(
            "_duplicate_group_count",
            F.count("*").over(duplicate_window)
        )

        normalized_business_record = F.struct(
            *[
                F.col(column).alias(column)
                for column in business_columns
            ]
        )

        df = df.withColumn(
            "_distinct_business_records",
            F.size(
                F.collect_set(
                    normalized_business_record
                ).over(duplicate_window)
            )
        )

        df = df.withColumn(
            "rejection_reason",
            F.concat_ws(
                "|",
                F.col("rejection_reason"),
                F.when(
                    (F.col("_duplicate_group_count") > 1)
                    & (
                        F.col("_distinct_business_records") > 1
                    ),
                    "CONFLICTING_DUPLICATE_ADDRESS_ID"
                )
            )
        )

        # Exact duplicate groups: choose one deterministic survivor.
        # Prefer a valid timestamp, then use raw values for stable ordering.
        survivor_window = (
            Window
            .partitionBy("address_id")
            .orderBy(
                F.when(
                    F.col("last_update_quality") == "VALID",
                    1
                ).otherwise(0).desc(),
                F.col("raw_last_update").asc_nulls_last(),
                F.col("raw_address").asc_nulls_last(),
                F.col("raw_postal_code").asc_nulls_last(),
                F.col("raw_phone").asc_nulls_last(),
            )
        )

        df = df.withColumn(
            "_survivor_rank",
            F.row_number().over(survivor_window)
        )

        df = df.withColumn(
            "rejection_reason",
            F.concat_ws(
                "|",
                F.col("rejection_reason"),
                F.when(
                    (F.col("_duplicate_group_count") > 1)
                    & (
                        F.col("_distinct_business_records") == 1
                    )
                    & (F.col("_survivor_rank") > 1),
                    "DUPLICATE_ADDRESS_RECORD"
                )
            )
        )

        # --------------------------------------------------
        # 8. Split Silver and Quarantine
        # --------------------------------------------------

        df = df.drop(
            "address_id_clean",
            "city_id_clean",
            "last_update_clean",
            "_duplicate_group_count",
            "_distinct_business_records",
            "_survivor_rank",
        )

        valid_df = df.filter(
            F.col("rejection_reason") == ""
        )

        rejected_df = df.filter(
            F.col("rejection_reason") != ""
        )

        # --------------------------------------------------
        # 9. Select clean Silver schema
        # --------------------------------------------------

        valid_df = valid_df.select(
            "address_id",
            "address",
            "address2",
            "district",
            "city_id",
            "postal_code",
            "postal_code_quality",
            "phone",
            "phone_quality",
            "last_update",
            "last_update_quality",
            "raw_address_id",
            "raw_address",
            "raw_address2",
            "raw_district",
            "raw_city_id",
            "raw_postal_code",
            "raw_phone",
            "raw_last_update",
        )

        return valid_df, rejected_df
