from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


class PaymentProcessor:
    """
    Payment Silver transformation processor.

    Business rules:
    - All identifiers must be positive BIGINT values.
    - Customer, Staff and Rental references must exist in their Silver datasets.
    - Amount must be positive and have at most two decimal places.
    - Correctly comma-grouped amounts are normalized (for example, 1,000.50).
    - Amounts above 100 are retained with a HIGH_AMOUNT quality flag.
    - Payment date is mandatory and must match a supported explicit format.
    - Slash dates are interpreted as DD/MM/YYYY.
    - Exact duplicate source rows are detected across the entire source before
      foreign-key validation. One copy is retained for further processing;
      additional copies are quarantined as DUPLICATE_PAYMENT_RECORD.
    - Conflicting versions of a positive payment_id are identified before
      foreign-key validation. Every distinct version is quarantined.
    - Row-level validation reasons are preserved alongside conflict/duplicate
      reasons.
    - Original source values are retained in raw_* audit columns.
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

    IDENTIFIER_COLUMNS = [
        "payment_id",
        "customer_id",
        "staff_id",
        "rental_id",
    ]

    REQUIRED_COLUMNS = {
        "payment_id",
        "customer_id",
        "staff_id",
        "rental_id",
        "amount",
        "payment_date",
    }

    def _normalize_nulls(self, df: DataFrame) -> DataFrame:
        for column in df.columns:
            if column.startswith("raw_") or column.startswith("_"):
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

    def _parse_payment_timestamp(self, column_name: str):
        expressions = [
            F.try_to_timestamp(
                F.col(column_name),
                F.lit(timestamp_format),
            )
            for timestamp_format in self.TIMESTAMP_FORMATS
        ]
        return F.coalesce(*expressions)

    def _reference_ids(self, reference_df: DataFrame, key: str, alias: str):
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
        customer_df: DataFrame,
        staff_df: DataFrame,
        rental_df: DataFrame,
    ) -> tuple[DataFrame, DataFrame]:

        missing_columns = self.REQUIRED_COLUMNS - set(source_df.columns)
        if missing_columns:
            raise ValueError(
                f"Payment source is missing columns: {sorted(missing_columns)}"
            )

        # 1. Preserve source values and assign a per-row technical identifier.
        source_columns = sorted(self.REQUIRED_COLUMNS)
        df = source_df

        for column in source_columns:
            df = df.withColumn(
                f"raw_{column}",
                F.col(column).cast("string"),
            )

        raw_columns = [f"raw_{column}" for column in source_columns]

        df = (
            df
            .withColumn("_source_row_id", F.monotonically_increasing_id())
        )

        # 2. Detect exact source-row duplicates BEFORE any validation.
        # Nulls are grouped together by Spark's window partition semantics.
        duplicate_window = (
            Window
            .partitionBy(*[F.col(c) for c in raw_columns])
            .orderBy(F.col("_source_row_id"))
        )

        df = df.withColumn(
            "_exact_duplicate_rank",
            F.row_number().over(duplicate_window),
        )

        # 3. Normalize null representations and parse identifiers.
        df = self._normalize_nulls(df)

        for column in self.IDENTIFIER_COLUMNS:
            df = df.withColumn(
                column,
                F.expr(f"try_cast(`{column}` AS BIGINT)"),
            )

        # 4. Normalize and validate amount.
        # Negative numeric values are parsed and then classified as
        # NON_POSITIVE rather than being mislabeled as malformed text.
        amount_text = F.trim(F.col("amount").cast("string"))
        amount_format_valid = amount_text.rlike(
            r"^-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$"
        )
        amount_without_commas = F.regexp_replace(amount_text, ",", "")
        amount_decimal_places = F.length(
            F.regexp_extract(amount_without_commas, r"\.(\d+)$", 1)
        )
        parsed_amount = F.expr(
            "try_cast(regexp_replace(trim(cast(amount as string)), ',', '') "
            "AS DECIMAL(18, 6))"
        )

        df = (
            df
            .withColumn("_amount_format_valid", amount_format_valid)
            .withColumn("_amount_decimal_places", amount_decimal_places)
            .withColumn("_amount_parsed", parsed_amount)
            .withColumn(
                "amount_quality",
                F.when(self._is_missing_raw("raw_amount"), F.lit("MISSING"))
                .when(~F.col("_amount_format_valid"), F.lit("INVALID_FORMAT"))
                .when(F.col("_amount_parsed").isNull(), F.lit("INVALID"))
                .when(
                    F.col("_amount_decimal_places") > 2,
                    F.lit("INVALID_PRECISION"),
                )
                .when(F.col("_amount_parsed") <= 0, F.lit("NON_POSITIVE"))
                .when(F.col("_amount_parsed") > 100, F.lit("HIGH_AMOUNT"))
                .otherwise(F.lit("VALID")),
            )
            .withColumn(
                "amount",
                F.when(
                    F.col("amount_quality").isin("VALID", "HIGH_AMOUNT"),
                    F.col("_amount_parsed").cast("decimal(18,2)"),
                ).otherwise(F.lit(None).cast("decimal(18,2)")),
            )
        )

        # 5. Parse mandatory payment timestamp. DD/MM/YYYY is intentional.
        df = df.withColumn(
            "payment_date",
            self._parse_payment_timestamp("payment_date"),
        )

        df = df.withColumn(
            "payment_date_quality",
            F.when(
                self._is_missing_raw("raw_payment_date"),
                F.lit("MISSING"),
            )
            .when(F.col("payment_date").isNull(), F.lit("MALFORMED"))
            .otherwise(F.lit("VALID")),
        )

        # 6. Build a normalized business signature and identify conflicting
        # payment_id versions BEFORE foreign-key validation.
        #
        # Valid values use normalized representations (so 1,000.50 and
        # 1000.50 are equivalent). Invalid/unparseable values fall back to
        # their raw representation so distinct bad values are not collapsed.
        # For identifier fields, a genuinely missing source value is treated
        # as a completable gap (NULL is ignored by countDistinct). Malformed
        # populated text such as "ABC" remains a distinct invalid value.
        # Amount/date values do not use this completion rule.
        identifier_signatures = {
            "customer_id": F.when(
                self._is_missing_raw("raw_customer_id"),
                F.lit(None).cast("string"),
            ).otherwise(
                F.coalesce(
                    F.col("customer_id").cast("string"),
                    F.concat(F.lit("RAW:"), F.col("raw_customer_id")),
                )
            ),
            "staff_id": F.when(
                self._is_missing_raw("raw_staff_id"),
                F.lit(None).cast("string"),
            ).otherwise(
                F.coalesce(
                    F.col("staff_id").cast("string"),
                    F.concat(F.lit("RAW:"), F.col("raw_staff_id")),
                )
            ),
            "rental_id": F.when(
                self._is_missing_raw("raw_rental_id"),
                F.lit(None).cast("string"),
            ).otherwise(
                F.coalesce(
                    F.col("rental_id").cast("string"),
                    F.concat(F.lit("RAW:"), F.col("raw_rental_id")),
                )
            ),
        }

        distinct_source_versions = df.filter(
            F.col("_exact_duplicate_rank") == 1
        )

        conflict_aggs = [
            F.countDistinct(identifier_signatures["customer_id"]).alias("_customer_versions"),
            F.countDistinct(identifier_signatures["staff_id"]).alias("_staff_versions"),
            F.countDistinct(identifier_signatures["rental_id"]).alias("_rental_versions"),
            F.countDistinct(
                F.coalesce(F.col("_amount_parsed").cast("string"),
                           F.concat(F.lit("RAW:"), F.coalesce(F.col("raw_amount"), F.lit("<NULL>"))))
            ).alias("_amount_versions"),
            F.countDistinct(
                F.coalesce(F.col("payment_date").cast("string"),
                           F.concat(F.lit("RAW:"), F.coalesce(F.col("raw_payment_date"), F.lit("<NULL>"))))
            ).alias("_date_versions"),
        ]

        conflicting_payment_ids = (
            distinct_source_versions
            .filter(F.col("payment_id").isNotNull() & (F.col("payment_id") > 0))
            .groupBy("payment_id")
            .agg(*conflict_aggs)
            .filter(
                (F.col("_customer_versions") > 1)
                | (F.col("_staff_versions") > 1)
                | (F.col("_rental_versions") > 1)
                | (F.col("_amount_versions") > 1)
                | (F.col("_date_versions") > 1)
            )
            .select("payment_id")
            .withColumn("_is_conflicting_payment", F.lit(True))
        )

        df = (
            df
            .join(F.broadcast(conflicting_payment_ids), on="payment_id", how="left")
            .withColumn(
                "_is_conflicting_payment",
                F.coalesce(F.col("_is_conflicting_payment"), F.lit(False)),
            )
        )

        # 7. Validate all three foreign keys against their Silver datasets.
        reference_specs = [
            (
                "customer_id",
                customer_df,
                "customer_id",
                "_customer_ref_id",
                "customer_reference_quality",
            ),
            (
                "staff_id",
                staff_df,
                "staff_id",
                "_staff_ref_id",
                "staff_reference_quality",
            ),
            (
                "rental_id",
                rental_df,
                "rental_id",
                "_rental_ref_id",
                "rental_reference_quality",
            ),
        ]

        for source_key, reference_df, reference_key, ref_alias, quality_col in reference_specs:
            reference_ids = self._reference_ids(
                reference_df,
                reference_key,
                ref_alias,
            )

            df = df.join(
                F.broadcast(reference_ids),
                F.col(source_key) == F.col(ref_alias),
                "left",
            )

            df = df.withColumn(
                quality_col,
                F.when(
                    F.col(source_key).isNull() | (F.col(source_key) <= 0),
                    F.lit("INVALID_ID"),
                )
                .when(F.col(ref_alias).isNotNull(), F.lit("VALID"))
                .otherwise(F.lit("INVALID_REFERENCE")),
            ).drop(ref_alias)

        # 8. Apply the narrow missing-value completion rule for identifiers.
        # A pair can be completed only when:
        # - exactly two distinct source versions exist for the payment_id;
        # - exactly one Customer/Staff/Rental identifier is truly missing;
        # - the other version has one populated value that passes its FK check;
        # - all remaining business attributes match exactly.
        # Malformed values (for example, "ABC") are not treated as missing.
        distinct_after_fk = df.filter(
            F.col("_exact_duplicate_rank") == 1
        )

        completion_specs = [
            ("customer_id", "raw_customer_id", "customer_reference_quality"),
            ("staff_id", "raw_staff_id", "staff_reference_quality"),
            ("rental_id", "raw_rental_id", "rental_reference_quality"),
        ]

        completion_aggs = [F.count("*").alias("_distinct_version_rows")]
        completion_conditions = {}

        comparison_exprs = {
            "customer_id": identifier_signatures["customer_id"],
            "staff_id": identifier_signatures["staff_id"],
            "rental_id": identifier_signatures["rental_id"],
            "amount": F.coalesce(
                F.col("_amount_parsed").cast("string"),
                F.concat(F.lit("RAW:"), F.coalesce(F.col("raw_amount"), F.lit("<NULL>"))),
            ),
            "payment_date": F.coalesce(
                F.col("payment_date").cast("string"),
                F.concat(F.lit("RAW:"), F.coalesce(F.col("raw_payment_date"), F.lit("<NULL>"))),
            ),
        }

        for field, raw_field, quality_field in completion_specs:
            missing_count_alias = f"_{field}_missing_count"
            populated_valid_count_alias = f"_{field}_populated_valid_count"
            populated_value_count_alias = f"_{field}_populated_value_count"
            other_signature_count_alias = f"_{field}_other_signature_count"

            completion_aggs.extend([
                F.sum(
                    F.when(self._is_missing_raw(raw_field), F.lit(1)).otherwise(F.lit(0))
                ).alias(missing_count_alias),
                F.sum(
                    F.when(
                        (~self._is_missing_raw(raw_field))
                        & (F.col(quality_field) == "VALID"),
                        F.lit(1),
                    ).otherwise(F.lit(0))
                ).alias(populated_valid_count_alias),
                F.countDistinct(
                    F.when(
                        (~self._is_missing_raw(raw_field))
                        & (F.col(quality_field) == "VALID"),
                        F.col(field).cast("string"),
                    )
                ).alias(populated_value_count_alias),
            ])

            other_fields = [name for name in comparison_exprs if name != field]
            other_signature = F.to_json(
                F.struct(*[
                    comparison_exprs[name].alias(name)
                    for name in other_fields
                ])
            )
            completion_aggs.append(
                F.countDistinct(other_signature).alias(other_signature_count_alias)
            )

        completion_summary = distinct_after_fk.groupBy("payment_id").agg(
            *completion_aggs
        )

        candidate_exprs = []
        for field, _, _ in completion_specs:
            candidate = (
                (F.col("_distinct_version_rows") == 2)
                & (F.col(f"_{field}_missing_count") == 1)
                & (F.col(f"_{field}_populated_valid_count") == 1)
                & (F.col(f"_{field}_populated_value_count") == 1)
                & (F.col(f"_{field}_other_signature_count") == 1)
            )
            completion_conditions[field] = candidate
            candidate_exprs.append(F.when(candidate, F.lit(1)).otherwise(F.lit(0)))

        completion_summary = completion_summary.withColumn(
            "_completion_candidate_count",
            sum(candidate_exprs[1:], candidate_exprs[0]),
        )

        for field, _, _ in completion_specs:
            completion_summary = completion_summary.withColumn(
                f"_complete_{field}",
                (F.col("_completion_candidate_count") == 1)
                & completion_conditions[field],
            )

        df = df.join(
            completion_summary.select(
                "payment_id",
                "_complete_customer_id",
                "_complete_staff_id",
                "_complete_rental_id",
            ),
            on="payment_id",
            how="left",
        )

        df = df.fillna({
            "_complete_customer_id": False,
            "_complete_staff_id": False,
            "_complete_rental_id": False,
        })

        df = df.withColumn(
            "_is_superseded_missing_version",
            (
                F.col("_complete_customer_id")
                & self._is_missing_raw("raw_customer_id")
            )
            | (
                F.col("_complete_staff_id")
                & self._is_missing_raw("raw_staff_id")
            )
            | (
                F.col("_complete_rental_id")
                & self._is_missing_raw("raw_rental_id")
            ),
        )

        # 9. Preserve every row-level validation reason.
        df = df.withColumn(
            "_row_rejection_reasons",
            F.array(
                F.when(
                    F.col("payment_id").isNull() | (F.col("payment_id") <= 0),
                    F.lit("INVALID_PAYMENT_ID"),
                ),
                F.when(
                    F.col("customer_reference_quality") == "INVALID_ID",
                    F.lit("INVALID_CUSTOMER_ID"),
                ),
                F.when(
                    F.col("customer_reference_quality") == "INVALID_REFERENCE",
                    F.lit("INVALID_CUSTOMER_REFERENCE"),
                ),
                F.when(
                    F.col("staff_reference_quality") == "INVALID_ID",
                    F.lit("INVALID_STAFF_ID"),
                ),
                F.when(
                    F.col("staff_reference_quality") == "INVALID_REFERENCE",
                    F.lit("INVALID_STAFF_REFERENCE"),
                ),
                F.when(
                    F.col("rental_reference_quality") == "INVALID_ID",
                    F.lit("INVALID_RENTAL_ID"),
                ),
                F.when(
                    F.col("rental_reference_quality") == "INVALID_REFERENCE",
                    F.lit("INVALID_RENTAL_REFERENCE"),
                ),
                F.when(
                    F.col("amount_quality") == "MISSING",
                    F.lit("MISSING_AMOUNT"),
                ),
                F.when(
                    F.col("amount_quality") == "INVALID_FORMAT",
                    F.lit("INVALID_AMOUNT_FORMAT"),
                ),
                F.when(
                    F.col("amount_quality") == "INVALID",
                    F.lit("INVALID_AMOUNT"),
                ),
                F.when(
                    F.col("amount_quality") == "INVALID_PRECISION",
                    F.lit("INVALID_AMOUNT_PRECISION"),
                ),
                F.when(
                    F.col("amount_quality") == "NON_POSITIVE",
                    F.lit("NON_POSITIVE_AMOUNT"),
                ),
                F.when(
                    F.col("payment_date_quality") == "MISSING",
                    F.lit("MISSING_PAYMENT_DATE"),
                ),
                F.when(
                    F.col("payment_date_quality") == "MALFORMED",
                    F.lit("INVALID_PAYMENT_DATE"),
                ),
                F.when(
                    F.col("_is_superseded_missing_version"),
                    F.lit("SUPERSEDED_BY_COMPLETED_VERSION"),
                ),
            ),
        )

        df = df.withColumn(
            "_row_rejection_reasons",
            F.expr("filter(_row_rejection_reasons, x -> x IS NOT NULL)"),
        )

        # 9. Compose final rejection reasons.
        # Conflicts are flagged before FK validation, but row-level reasons
        # remain visible. Exact duplicate copies are always quarantined too.
        df = df.withColumn(
            "_all_rejection_reasons",
            F.concat(
                F.col("_row_rejection_reasons"),
                F.when(
                    F.col("_is_conflicting_payment"),
                    F.array(F.lit("CONFLICTING_DUPLICATE_PAYMENT_ID")),
                ).otherwise(F.expr("cast(array() as array<string>)")),
                F.when(
                    F.col("_exact_duplicate_rank") > 1,
                    F.array(F.lit("DUPLICATE_PAYMENT_RECORD")),
                ).otherwise(F.expr("cast(array() as array<string>)")),
            ),
        )

        df = df.withColumn(
            "_all_rejection_reasons",
            F.array_distinct(
                F.col("_all_rejection_reasons")
            ),
        )

        df = df.withColumn(
            "rejection_reason",
            F.concat_ws("|", F.col("_all_rejection_reasons")),
        )

        # 10. A record is Silver-eligible only when it is:
        # - the first copy of an exact source row,
        # - not part of a conflicting payment_id group, and
        # - free of row-level validation errors.
        valid_df = df.filter(
            (F.col("_exact_duplicate_rank") == 1)
            & (~F.col("_is_conflicting_payment"))
            & (~F.col("_is_superseded_missing_version"))
            & (F.size(F.col("_row_rejection_reasons")) == 0)
        )

        rejected_df = df.filter(
            (F.col("_exact_duplicate_rank") > 1)
            | F.col("_is_conflicting_payment")
            | F.col("_is_superseded_missing_version")
            | (F.size(F.col("_row_rejection_reasons")) > 0)
        )

        # 11. Remove internal processing columns; retain raw audit and DQ data.
        internal_columns = [
            c for c in valid_df.columns if c.startswith("_")
        ]
        valid_df = valid_df.drop(*internal_columns)

        internal_columns = [
            c for c in rejected_df.columns if c.startswith("_")
        ]
        rejected_df = rejected_df.drop(*internal_columns)

        return valid_df, rejected_df
