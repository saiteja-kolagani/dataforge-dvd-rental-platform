from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class CityProcessor:
    """
    Cleanses raw City records and validates country references
    against the already-cleansed Country Silver dataset.
    """

    TIMESTAMP_FORMATS = [
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
        "dd-MM-yyyy HH:mm",
        "MM/dd/yyyy",
    ]

    NULL_LIKE_VALUES = ["", "NULL", "NA", "N/A", "UNKNOWN"]

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
        city_df: DataFrame,
        country_silver_df: DataFrame,
    ) -> tuple[DataFrame, DataFrame]:

        required_columns = {
            "city_id",
            "city",
            "country_id",
            "last_update",
        }

        missing_columns = required_columns - set(city_df.columns)

        if missing_columns:
            raise ValueError(
                f"City dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if "country_id" not in country_silver_df.columns:
            raise ValueError(
                "Country Silver DataFrame must contain country_id"
            )

        # ---------------------------------------------------------
        # 1. Preserve source values and normalize input
        # ---------------------------------------------------------
        df = city_df.select(
            *[
                F.col(column).cast("string").alias(f"raw_{column}")
                for column in city_df.columns
            ]
        )

        df = (
            df
            .withColumn("city_id_text", self._normalize_text("raw_city_id"))
            .withColumn("city_name_clean", self._normalize_text("raw_city"))
            .withColumn(
                "country_id_text",
                self._normalize_text("raw_country_id"),
            )
            .withColumn(
                "last_update_text",
                self._normalize_text("raw_last_update"),
            )
            .withColumn(
                "city_id",
                F.expr(
                    "try_cast(city_id_text AS INT)"
                ),
            )
            .withColumn(
                "country_id",
                F.expr(
                    "try_cast(country_id_text AS INT)"
                ),
            )
            .withColumn(
                "city",
                F.initcap(F.trim("city_name_clean")),
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

        # ---------------------------------------------------------
        # 2. Build the valid Country Silver reference list
        # ---------------------------------------------------------
        valid_country_ids = (
            country_silver_df
            .select(
                F.col("country_id").cast("int").alias("country_id")
            )
            .where(F.col("country_id").isNotNull())
            .distinct()
            .withColumn("country_exists", F.lit(True))
        )

        df = df.join(
            valid_country_ids,
            on="country_id",
            how="left",
        )

        # ---------------------------------------------------------
        # 3. Identify conflicting duplicate city IDs
        # ---------------------------------------------------------
        city_id_conflicts = (
            df
            .where(F.col("city_id").isNotNull())
            .groupBy("city_id")
            .agg(
                F.countDistinct(
                    F.struct(
                        F.lower(F.trim("city_name_clean")),
                        F.col("country_id_text"),
                    )
                ).alias("distinct_city_versions")
            )
            .where(F.col("distinct_city_versions") > 1)
            .select("city_id")
            .withColumn("has_city_id_conflict", F.lit(True))
        )

        df = df.join(
            city_id_conflicts,
            on="city_id",
            how="left",
        )

        df = df.withColumn(
            "has_city_id_conflict",
            F.coalesce(
                F.col("has_city_id_conflict"),
                F.lit(False),
            ),
        )

        # ---------------------------------------------------------
        # 4. Assign base validation rejection reasons
        # ---------------------------------------------------------
        df = df.withColumn(
            "base_rejection_reason",
            F.concat_ws(
                "|",
                F.when(
                    F.col("city_id").isNull() | (F.col("city_id") <= 0),
                    F.lit("INVALID_CITY_ID"),
                ),
                F.when(
                    F.col("city_name_clean").isNull(),
                    F.lit("MISSING_OR_INVALID_CITY_NAME"),
                ),
                F.when(
                    F.col("country_id").isNull()
                    | (F.col("country_id") <= 0),
                    F.lit("INVALID_COUNTRY_ID"),
                ),
                F.when(
                    F.col("country_id").isNotNull()
                    & (F.col("country_id") > 0)
                    & F.col("country_exists").isNull(),
                    F.lit("INVALID_COUNTRY_REFERENCE"),
                ),
            ),
        )

        # ---------------------------------------------------------
        # 5. Separate conflicting city ID groups
        # ---------------------------------------------------------
        df = df.withColumn(
            "rejection_reason",
            F.concat_ws(
                "|",
                F.col("base_rejection_reason"),
                F.when(
                    F.col("has_city_id_conflict"),
                    F.lit("CONFLICTING_DUPLICATE_CITY_ID"),
                ),
            ),
        )

        conflict_rejected = df.where(
            F.col("has_city_id_conflict")
        )

        candidates = df.where(
            (~F.col("has_city_id_conflict"))
            & (F.col("base_rejection_reason") == "")
        )

        base_rejected = df.where(
            (~F.col("has_city_id_conflict"))
            & (F.col("base_rejection_reason") != "")
        )

        # ---------------------------------------------------------
        # 6. Resolve exact duplicate city records
        # ---------------------------------------------------------
        duplicate_window = (
            Window
            .partitionBy("city_id")
            .orderBy(
                F.when(
                    F.col("last_update_quality") == "VALID",
                    1,
                ).otherwise(0).desc(),
                F.col("last_update").desc_nulls_last(),
                F.col("raw_last_update").desc_nulls_last(),
                F.col("raw_city").asc_nulls_last(),
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
                F.lit("DUPLICATE_CITY_RECORD"),
            )
            .drop("duplicate_rank")
        )

        valid_df = candidates.where(
            F.col("duplicate_rank") == 1
        )

        # ---------------------------------------------------------
        # 7. Combine all rejected records
        # ---------------------------------------------------------
        rejected_df = (
            base_rejected
            .unionByName(conflict_rejected)
            .unionByName(duplicate_rejected)
        )

        # ---------------------------------------------------------
        # 8. Select final Silver columns
        # ---------------------------------------------------------
        final_columns = [
            "city_id",
            "city",
            "country_id",
            "last_update",
            "last_update_quality",
            "raw_city_id",
            "raw_city",
            "raw_country_id",
            "raw_last_update",
        ]

        valid_df = valid_df.select(*final_columns)

        rejected_df = rejected_df.select(
            *final_columns,
            "rejection_reason",
        )

        return valid_df, rejected_df