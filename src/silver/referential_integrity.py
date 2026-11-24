
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class ReferentialIntegrityValidator:
    """
    Validates foreign-key relationships between Silver datasets.

    reference_status:
        VERIFIED   - Reference dataset is complete; missing keys are INVALID.
        UNRESOLVED - Reference dataset is provisional; missing keys remain UNRESOLVED.
    """

    VALID_REFERENCE_STATUSES = {"VERIFIED", "UNRESOLVED"}

    def validate(
        self,
        source_df: DataFrame,
        reference_df: DataFrame,
        source_key: str,
        reference_key: str,
        relationship_name: str,
        reference_status: str = "VERIFIED",
    ) -> DataFrame:

        reference_status = reference_status.upper()

        if reference_status not in self.VALID_REFERENCE_STATUSES:
            raise ValueError(
                "reference_status must be VERIFIED or UNRESOLVED"
            )

        source = source_df.select(
            F.col(source_key).cast("string").alias("_source_reference_id")
        )

        reference = (
            reference_df
            .select(
                F.col(reference_key).cast("string").alias("_reference_id")
            )
            .dropDuplicates(["_reference_id"])
        )

        source = source.withColumn(
            "_numeric_reference_id",
            F.expr("try_cast(_source_reference_id AS BIGINT)")
        )

        joined = source.join(
            reference,
            source["_source_reference_id"] == reference["_reference_id"],
            "left"
        )

        if reference_status == "UNRESOLVED":
            missing_reference_status = "UNRESOLVED"
        else:
            missing_reference_status = "INVALID"

        report = joined.withColumn(
            "validation_status",
            F.when(
                F.col("_source_reference_id").isNull()
                | (F.trim(F.col("_source_reference_id")) == "")
                | F.col("_numeric_reference_id").isNull()
                | (F.col("_numeric_reference_id") <= 0),
                F.lit("INVALID")
            )
            .when(
                F.col("_reference_id").isNotNull(),
                F.lit("VALID")
            )
            .otherwise(F.lit(missing_reference_status))
        )

        return report.select(
            F.col("_source_reference_id").alias("reference_id"),
            F.lit(relationship_name).alias("relationship_name"),
            F.col("validation_status")
        )

    def summarize(self, report_df: DataFrame) -> DataFrame:
        return (
            report_df
            .groupBy("relationship_name", "validation_status")
            .count()
            .orderBy("relationship_name", "validation_status")
        )
