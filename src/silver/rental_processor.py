from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class RentalProcessor:
    """Build Rental Silver records and quarantine invalid/conflicting records."""

    NULL_VALUES = ["", "NULL", "NA", "N/A", "NONE", "NAN"]
    TIMESTAMP_FORMATS = [
        "yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd", "yyyy/MM/dd",
        "dd-MM-yyyy HH:mm", "dd/MM/yyyy", "MM/dd/yyyy", "dd-MMM-yyyy",
    ]
    REQUIRED_COLUMNS = {
        "rental_id", "rental_date", "inventory_id", "customer_id",
        "return_date", "staff_id", "last_update",
    }

    def _normalize_nulls(self, df):
        for c in df.columns:
            if not c.startswith("raw_"):
                df = df.withColumn(
                    c,
                    F.when(
                        F.upper(F.trim(F.col(c).cast("string"))).isin(self.NULL_VALUES),
                        F.lit(None).cast("string"),
                    ).otherwise(F.trim(F.col(c).cast("string")))
                )
        return df

    def _missing(self, c):
        return F.col(c).isNull() | F.upper(F.trim(F.col(c).cast("string"))).isin(
            self.NULL_VALUES
        )

    def _timestamp(self, c):
        return F.coalesce(*[
            F.try_to_timestamp(F.col(c), F.lit(fmt))
            for fmt in self.TIMESTAMP_FORMATS
        ])

    def _reference_ids(self, ref_df, key, alias):
        if key not in ref_df.columns:
            raise ValueError(f"Reference dataset is missing key column: {key}")
        return (
            ref_df.select(F.expr(f"try_cast(`{key}` AS BIGINT)").alias(alias))
            .filter(F.col(alias).isNotNull() & (F.col(alias) > 0))
            .distinct()
        )

    def process(
        self, source_df: DataFrame, inventory_df: DataFrame,
        customer_df: DataFrame, staff_df: DataFrame,
    ) -> tuple[DataFrame, DataFrame]:
        missing = self.REQUIRED_COLUMNS - set(source_df.columns)
        if missing:
            raise ValueError(f"Rental source is missing columns: {sorted(missing)}")

        df = source_df
        for c in sorted(self.REQUIRED_COLUMNS):
            df = df.withColumn(f"raw_{c}", F.col(c).cast("string"))
        df = self._normalize_nulls(df)

        for c in ["rental_id", "inventory_id", "customer_id", "staff_id"]:
            df = df.withColumn(c, F.expr(f"try_cast(`{c}` AS BIGINT)"))

        for c in ["rental_date", "return_date", "last_update"]:
            df = df.withColumn(c, self._timestamp(c))
            quality = f"{c}_quality"
            df = df.withColumn(
                quality,
                F.when(self._missing(f"raw_{c}"), "MISSING")
                 .when(F.col(c).isNull(), "MALFORMED")
                 .otherwise("VALID")
            )

        refs = [
            (inventory_df, "inventory_id", "_inventory_ref",
             "inventory_reference_quality"),
            (customer_df, "customer_id", "_customer_ref",
             "customer_reference_quality"),
            (staff_df, "staff_id", "_staff_ref", "staff_reference_quality"),
        ]
        for ref_df, key, alias, quality in refs:
            ids = self._reference_ids(ref_df, key, alias)
            df = df.join(F.broadcast(ids), F.col(key) == F.col(alias), "left")
            df = df.withColumn(
                quality,
                F.when(F.col(key).isNull() | (F.col(key) <= 0), "INVALID_ID")
                 .when(F.col(alias).isNotNull(), "VALID")
                 .when(F.lit(key == "staff_id"), "UNRESOLVED")
                 .otherwise("INVALID_REFERENCE")
            ).drop(alias)

        invalid = (
            F.col("rental_id").isNull() | (F.col("rental_id") <= 0)
            | (F.col("rental_date_quality") != "VALID")
            | (F.col("inventory_reference_quality") != "VALID")
            | (F.col("customer_reference_quality") != "VALID")
            | (F.col("staff_reference_quality").isin("INVALID_ID", "INVALID_REFERENCE"))
        )

        invalid_rows = df.filter(invalid).withColumn(
            "rejection_reason",
            F.concat_ws(
                "|",
                F.when(F.col("rental_id").isNull() | (F.col("rental_id") <= 0),
                       "INVALID_RENTAL_ID"),
                F.when(F.col("rental_date_quality") != "VALID",
                       "INVALID_RENTAL_DATE"),
                F.when(F.col("inventory_reference_quality") == "INVALID_ID",
                       "INVALID_INVENTORY_ID"),
                F.when(F.col("inventory_reference_quality") == "INVALID_REFERENCE",
                       "INVALID_INVENTORY_REFERENCE"),
                F.when(F.col("customer_reference_quality") == "INVALID_ID",
                       "INVALID_CUSTOMER_ID"),
                F.when(F.col("customer_reference_quality") == "INVALID_REFERENCE",
                       "INVALID_CUSTOMER_REFERENCE"),
                F.when(F.col("staff_reference_quality") == "INVALID_ID",
                       "INVALID_STAFF_ID"),
                F.when(F.col("staff_reference_quality") == "INVALID_REFERENCE",
                       "INVALID_STAFF_REFERENCE"),
            )
        )

        candidates = df.filter(~invalid)
        signature_cols = [
            "rental_date", "inventory_id", "customer_id", "return_date", "staff_id"
        ]
        candidates = candidates.withColumn(
            "_business_signature",
            F.to_json(F.struct(*[F.col(c).alias(c) for c in signature_cols]))
        )
        groups = candidates.groupBy("rental_id").agg(
            F.count("*").alias("_record_count"),
            F.countDistinct("_business_signature").alias("_business_versions"),
        )
        candidates = candidates.join(groups, "rental_id", "left")

        conflicts = candidates.filter(F.col("_business_versions") > 1).withColumn(
            "rejection_reason", F.lit("CONFLICTING_DUPLICATE_RENTAL_ID")
        )
        singles = candidates.filter(F.col("_record_count") == 1)
        dupes = candidates.filter(
            (F.col("_record_count") > 1) & (F.col("_business_versions") == 1)
        )

        survivor_order = Window.partitionBy("rental_id").orderBy(
            F.when(F.col("last_update_quality") == "VALID", 0).otherwise(1),
            F.col("raw_last_update").asc_nulls_last(),
            F.col("raw_return_date").asc_nulls_last(),
            F.col("raw_rental_date").asc_nulls_last(),
        )
        ranked = dupes.withColumn("_row_number", F.row_number().over(survivor_order))
        survivors = ranked.filter(F.col("_row_number") == 1).drop("_row_number")
        duplicate_rejects = ranked.filter(F.col("_row_number") > 1).withColumn(
            "rejection_reason", F.lit("DUPLICATE_RENTAL_RECORD")
        )

        drop_cols = ["_business_signature", "_record_count", "_business_versions"]
        valid_df = singles.unionByName(survivors).drop(*drop_cols)
        rejected_df = (
            invalid_rows
            .unionByName(conflicts.drop(*drop_cols), allowMissingColumns=True)
            .unionByName(
                duplicate_rejects.drop(*drop_cols, "_row_number"),
                allowMissingColumns=True,
            )
        )
        return valid_df, rejected_df
