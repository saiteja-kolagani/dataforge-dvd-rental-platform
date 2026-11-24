
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class FilmProcessor:
    """
    Cleans and validates DVD Rental film data.

    Hard validation:
      - film_id must be a positive integer.
      - title must be present.
      - release_year must be between 2000 and 2026.
      - language_id must reference a valid Language Silver record.
      - rental_duration must be a positive integer.
      - rental_rate and replacement_cost must be non-negative decimals.
      - length must be a positive integer when supplied.
      - rating must be one of G, PG, PG-13, R, NC-17.
      - special_features must contain only recognized features.

    Soft validation:
      - last_update is parsed when possible.
      - Missing or malformed timestamps are flagged, not rejected.

    Duplicate handling:
      - Exact duplicates: retain one deterministic record.
      - Optional-field variants: retain one deterministic record.
      - Conflicting core attributes or titles: reject the entire group.
    """

    NULL_VALUES = ["", "NULL", "NA", "N/A", "UNKNOWN"]

    VALID_RATINGS = ["G", "PG", "PG-13", "R", "NC-17"]

    # Canonical feature names are used in the output.
    FEATURE_MAP = {
        "TRAILERS": "Trailers",
        "COMMENTARIES": "Commentaries",
        "DELETED SCENES": "Deleted Scenes",
        "BEHIND THE SCENES": "Behind the Scenes",
    }

    TIMESTAMP_FORMATS = [
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd",
        "dd-MM-yyyy HH:mm",
    ]

    MIN_RELEASE_YEAR = 2000
    MAX_RELEASE_YEAR = 2026

    def process(
        self,
        source_df: DataFrame,
        language_df: DataFrame,
    ):
        required_columns = {
            "film_id",
            "title",
            "description",
            "release_year",
            "language_id",
            "rental_duration",
            "rental_rate",
            "length",
            "replacement_cost",
            "rating",
            "special_features",
            "fulltext",
            "last_update",
        }

        missing_columns = required_columns - set(source_df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required source columns: {sorted(missing_columns)}"
            )

        if "language_id" not in language_df.columns:
            raise ValueError(
                "Language reference DataFrame must contain language_id"
            )

        # Preserve source values before transformation.
        df = source_df

        for column in sorted(required_columns):
            df = df.withColumn(
                f"raw_{column}",
                F.col(column).cast("string"),
            )

        # Normalize configured null-like values and whitespace.
        for column in [
            "film_id",
            "title",
            "description",
            "release_year",
            "language_id",
            "rental_duration",
            "rental_rate",
            "length",
            "replacement_cost",
            "rating",
            "special_features",
            "fulltext",
            "last_update",
        ]:
            raw_column = f"raw_{column}"

            df = df.withColumn(
                f"{column}_clean",
                F.when(
                    F.upper(F.trim(F.col(raw_column))).isin(
                        self.NULL_VALUES
                    ),
                    F.lit(None),
                ).otherwise(F.trim(F.col(raw_column))),
            )

        # Safely cast numeric fields. Invalid strings become NULL.
        df = (
            df
            .withColumn(
                "film_id",
                F.expr("try_cast(film_id_clean AS INT)"),
            )
            .withColumn(
                "release_year",
                F.expr("try_cast(release_year_clean AS INT)"),
            )
            .withColumn(
                "language_id",
                F.expr("try_cast(language_id_clean AS INT)"),
            )
            .withColumn(
                "rental_duration",
                F.expr("try_cast(rental_duration_clean AS INT)"),
            )
            .withColumn(
                "rental_rate",
                F.expr("try_cast(rental_rate_clean AS DECIMAL(10,2))"),
            )
            .withColumn(
                "length",
                F.expr("try_cast(length_clean AS INT)"),
            )
            .withColumn(
                "replacement_cost",
                F.expr(
                    "try_cast(replacement_cost_clean AS DECIMAL(10,2))"
                ),
            )
        )

        # Standardize title and optional descriptive fields.
        df = (
            df
            .withColumn("title", F.trim(F.col("title_clean")))
            .withColumn("description", F.col("description_clean"))
            .withColumn("fulltext", F.col("fulltext_clean"))
        )

        # Standardize rating values.
        df = df.withColumn(
            "rating",
            F.upper(F.trim(F.col("rating_clean"))),
        )

        # Normalize special_features.
        # The source field is comma-separated.
        df = df.withColumn(
            "_feature_tokens",
            F.when(
                F.col("special_features_clean").isNull(),
                F.lit(None).cast("array<string>"),
            ).otherwise(
                F.split(F.col("special_features_clean"), ",")
            ),
        )

        feature_map = F.create_map(
            *[
                item
                for key, value in self.FEATURE_MAP.items()
                for item in (F.lit(key), F.lit(value))
            ]
        )

        df = df.withColumn(
            "_normalized_features",
            F.transform(
                F.col("_feature_tokens"),
                lambda token: F.element_at(
                    feature_map,
                    F.upper(F.trim(token)),
                ),
            ),
        )

        df = df.withColumn(
            "_has_invalid_feature",
            F.when(
                F.col("_feature_tokens").isNull(),
                F.lit(False),
            ).otherwise(
                F.exists(
                    F.col("_normalized_features"),
                    lambda token: token.isNull(),
                )
            ),
        )

        df = df.withColumn(
            "special_features",
            F.when(
                F.col("_feature_tokens").isNull(),
                F.lit(None).cast("string"),
            ).otherwise(
                F.array_join(
                    F.sort_array(
                        F.array_distinct(
                            F.filter(
                                F.col("_normalized_features"),
                                lambda token: token.isNotNull(),
                            )
                        )
                    ),
                    ", ",
                )
            ),
        )

        # Parse last_update using the supported formats.
        parsed_timestamps = [
            F.try_to_timestamp(
                F.col("last_update_clean"),
                F.lit(timestamp_format),
            )
            for timestamp_format in self.TIMESTAMP_FORMATS
        ]

        df = df.withColumn(
            "last_update",
            F.coalesce(*parsed_timestamps),
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

        # Build the valid Language Silver reference set.
        valid_languages = (
            language_df
            .select(
                F.col("language_id")
                .cast("int")
                .alias("_reference_language_id")
            )
            .filter(F.col("_reference_language_id").isNotNull())
            .distinct()
            .withColumn("_language_exists", F.lit(True))
        )

        df = (
            df
            .join(
                valid_languages,
                F.col("language_id")
                == F.col("_reference_language_id"),
                "left",
            )
            .drop("_reference_language_id")
        )

        # Collect all hard validation errors for each source row.
        rejection_reasons = F.filter(
            F.array(
                F.when(
                    F.col("film_id").isNull() | (F.col("film_id") <= 0),
                    F.lit("INVALID_FILM_ID"),
                ),
                F.when(
                    F.col("title").isNull()
                    | (F.length(F.trim(F.col("title"))) == 0),
                    F.lit("MISSING_OR_INVALID_TITLE"),
                ),
                F.when(
                    F.col("release_year").isNull()
                    | (
                        (F.col("release_year") < self.MIN_RELEASE_YEAR)
                        | (F.col("release_year") > self.MAX_RELEASE_YEAR)
                    ),
                    F.lit("INVALID_RELEASE_YEAR"),
                ),
                F.when(
                    F.col("language_id").isNull()
                    | (F.col("language_id") <= 0),
                    F.lit("INVALID_LANGUAGE_ID"),
                ),
                F.when(
                    F.col("language_id").isNotNull()
                    & (F.col("language_id") > 0)
                    & F.col("_language_exists").isNull(),
                    F.lit("INVALID_LANGUAGE_REFERENCE"),
                ),
                F.when(
                    F.col("rental_duration").isNull()
                    | (F.col("rental_duration") <= 0),
                    F.lit("INVALID_RENTAL_DURATION"),
                ),
                F.when(
                    F.col("rental_rate").isNull()
                    | (F.col("rental_rate") < 0),
                    F.lit("INVALID_RENTAL_RATE"),
                ),
                F.when(
                    F.col("length_clean").isNotNull()
                    & (
                        F.col("length").isNull()
                        | (F.col("length") <= 0)
                    ),
                    F.lit("INVALID_LENGTH"),
                ),
                F.when(
                    F.col("replacement_cost").isNull()
                    | (F.col("replacement_cost") < 0),
                    F.lit("INVALID_REPLACEMENT_COST"),
                ),
                F.when(
                    F.col("rating").isNull()
                    | (~F.col("rating").isin(self.VALID_RATINGS)),
                    F.lit("INVALID_RATING"),
                ),
                F.when(
                    F.col("_has_invalid_feature"),
                    F.lit("INVALID_SPECIAL_FEATURE"),
                ),
            ),
            lambda reason: reason.isNotNull(),
        )

        df = df.withColumn(
            "rejection_reason",
            F.concat_ws("|", rejection_reasons),
        )

        base_valid = df.filter(F.col("rejection_reason") == "")
        base_rejected = df.filter(F.col("rejection_reason") != "")

        # Detect duplicate film IDs with conflicting titles.
        title_conflicts = (
            base_valid
            .groupBy("film_id")
            .agg(F.countDistinct("title").alias("_title_count"))
            .filter(F.col("_title_count") > 1)
            .select("film_id")
            .withColumn("_title_conflict", F.lit(True))
        )

        candidates = base_valid.join(
            title_conflicts,
            on="film_id",
            how="left",
        )

        title_conflict_rejected = (
            candidates
            .filter(F.col("_title_conflict") == True)
            .withColumn(
                "rejection_reason",
                F.lit("UNRESOLVED_DUPLICATE_TITLE_CONFLICT"),
            )
        )

        non_title_conflicts = candidates.filter(
            F.col("_title_conflict").isNull()
        )

        # Detect duplicate IDs with conflicting core business attributes.
        # Optional descriptive fields are intentionally excluded here.
        core_columns = [
            "title",
            "release_year",
            "language_id",
            "rental_duration",
            "rental_rate",
            "length",
            "replacement_cost",
            "rating",
        ]

        core_signature = F.to_json(
            F.struct(*[F.col(column) for column in core_columns])
        )

        core_conflicts = (
            non_title_conflicts
            .withColumn("_core_signature", core_signature)
            .groupBy("film_id")
            .agg(
                F.countDistinct("_core_signature").alias(
                    "_core_variant_count"
                )
            )
            .filter(F.col("_core_variant_count") > 1)
            .select("film_id")
            .withColumn("_core_conflict", F.lit(True))
        )

        candidates = non_title_conflicts.join(
            core_conflicts,
            on="film_id",
            how="left",
        )

        core_conflict_rejected = (
            candidates
            .filter(F.col("_core_conflict") == True)
            .withColumn(
                "rejection_reason",
                F.lit("UNRESOLVED_DUPLICATE_CORE_ATTRIBUTE_CONFLICT"),
            )
        )

        non_conflicting = candidates.filter(
            F.col("_core_conflict").isNull()
        )

        # Keep one deterministic row for each remaining film_id.
        # Optional-field variants are quarantined, not silently discarded.
        duplicate_window = (
            Window
            .partitionBy("film_id")
            .orderBy(
                F.col("raw_description").asc_nulls_first(),
                F.col("raw_special_features").asc_nulls_first(),
                F.col("raw_fulltext").asc_nulls_first(),
                F.col("raw_last_update").asc_nulls_first(),
            )
        )

        ranked = non_conflicting.withColumn(
            "_duplicate_rank",
            F.row_number().over(duplicate_window),
        )

        duplicate_rejected = (
            ranked
            .filter(F.col("_duplicate_rank") > 1)
            .withColumn(
                "rejection_reason",
                F.lit("DUPLICATE_RECORD_OR_OPTIONAL_FIELD_VARIANT"),
            )
        )

        valid_df = (
            ranked
            .filter(F.col("_duplicate_rank") == 1)
            .select(
                "film_id",
                "title",
                "description",
                "release_year",
                "language_id",
                "rental_duration",
                "rental_rate",
                "length",
                "replacement_cost",
                "rating",
                "special_features",
                "fulltext",
                "last_update",
                "last_update_quality",
                *[f"raw_{column}" for column in sorted(required_columns)],
            )
        )

        rejected_df = (
            base_rejected
            .select(
                "film_id",
                "title",
                "description",
                "release_year",
                "language_id",
                "rental_duration",
                "rental_rate",
                "length",
                "replacement_cost",
                "rating",
                "special_features",
                "fulltext",
                "last_update",
                "last_update_quality",
                *[f"raw_{column}" for column in sorted(required_columns)],
                "rejection_reason",
            )
            .unionByName(
                title_conflict_rejected.select(
                    "film_id",
                    "title",
                    "description",
                    "release_year",
                    "language_id",
                    "rental_duration",
                    "rental_rate",
                    "length",
                    "replacement_cost",
                    "rating",
                    "special_features",
                    "fulltext",
                    "last_update",
                    "last_update_quality",
                    *[f"raw_{column}" for column in sorted(required_columns)],
                    "rejection_reason",
                ),
                allowMissingColumns=True,
            )
            .unionByName(
                core_conflict_rejected.select(
                    "film_id",
                    "title",
                    "description",
                    "release_year",
                    "language_id",
                    "rental_duration",
                    "rental_rate",
                    "length",
                    "replacement_cost",
                    "rating",
                    "special_features",
                    "fulltext",
                    "last_update",
                    "last_update_quality",
                    *[f"raw_{column}" for column in sorted(required_columns)],
                    "rejection_reason",
                ),
                allowMissingColumns=True,
            )
            .unionByName(
                duplicate_rejected.select(
                    "film_id",
                    "title",
                    "description",
                    "release_year",
                    "language_id",
                    "rental_duration",
                    "rental_rate",
                    "length",
                    "replacement_cost",
                    "rating",
                    "special_features",
                    "fulltext",
                    "last_update",
                    "last_update_quality",
                    *[f"raw_{column}" for column in sorted(required_columns)],
                    "rejection_reason",
                ),
                allowMissingColumns=True,
            )
        )

        return valid_df, rejected_df
