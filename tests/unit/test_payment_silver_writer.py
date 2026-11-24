
from pathlib import Path
from datetime import date

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.silver.payment_processor import PaymentProcessor
from src.silver.silver_writer import SilverWriter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PAYMENT_PATH = PROJECT_ROOT / "data" / "raw" / "payment.csv"

# Use one specific dependency batch per table.
CUSTOMER_SILVER_PATH = (
    PROJECT_ROOT / "data" / "processed" / "silver" / "customer"
    / "ingestion_date=2026-11-22"
    / "batch_id=20261122_customer_test_02"
)

STAFF_SILVER_PATH = (
    PROJECT_ROOT / "data" / "processed" / "silver" / "staff"
    / "ingestion_date=2026-11-22"
    / "batch_id=20261122_staff_test_01"
)

RENTAL_SILVER_PATH = (
    PROJECT_ROOT / "data" / "processed" / "silver" / "rental"
    / "ingestion_date=2026-11-23"
    / "batch_id=20261123_rental_test_01"
)

SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver"
QUARANTINE_PATH = PROJECT_ROOT / "data" / "processed" / "quarantine"

# New batch ID: do not reuse an existing output batch.
BATCH_ID = "20261123_payment_test_03"
INGESTION_DATE = date(2026, 11, 23)


def assert_quarantined_twice(rejected_df, payment_id, expected_reason):
    """
    Assert that both source versions of a conflicting Payment ID
    are quarantined and contain the expected rejection reason.
    """
    records = rejected_df.filter(
        F.col("payment_id") == str(payment_id)
    )

    count = records.count()

    assert count == 2, (
        f"Payment ID {payment_id}: expected 2 quarantined versions, "
        f"found {count}."
    )

    invalid_reason_count = records.filter(
        ~F.col("rejection_reason").contains(expected_reason)
    ).count()

    assert invalid_reason_count == 0, (
        f"Payment ID {payment_id}: expected rejection reason "
        f"{expected_reason} on both versions."
    )


def main():
    required_paths = [
        RAW_PAYMENT_PATH,
        CUSTOMER_SILVER_PATH,
        STAFF_SILVER_PATH,
        RENTAL_SILVER_PATH,
    ]

    missing_paths = [
        str(path) for path in required_paths if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Payment integration test cannot run because these "
            "dependencies are missing:\n- "
            + "\n- ".join(missing_paths)
        )

    spark = (
        SparkSession.builder
        .appName("DataForge-Payment-Silver-Test")
        .master("local[2]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        print("\n========== PAYMENT SILVER PROCESSING ==========")

        raw_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(str(RAW_PAYMENT_PATH))
        )

        customer_df = spark.read.parquet(str(CUSTOMER_SILVER_PATH))
        staff_df = spark.read.parquet(str(STAFF_SILVER_PATH))
        rental_df = spark.read.parquet(str(RENTAL_SILVER_PATH))

        source_count = raw_df.count()

        print(f"Raw Payment records: {source_count}")
        print(f"Customer dependency: {CUSTOMER_SILVER_PATH}")
        print(f"Staff dependency: {STAFF_SILVER_PATH}")
        print(f"Rental dependency: {RENTAL_SILVER_PATH}")

        print(f"Customer Silver records: {customer_df.count()}")
        print(f"Staff Silver records: {staff_df.count()}")
        print(f"Rental Silver records: {rental_df.count()}")

        processor = PaymentProcessor()

        valid_df, rejected_df = processor.process(
            source_df=raw_df,
            customer_df=customer_df,
            staff_df=staff_df,
            rental_df=rental_df,
        )

        valid_count = valid_df.count()
        rejected_count = rejected_df.count()

        print("\n========== PROCESSING SUMMARY ==========")
        print(f"Source records: {source_count}")
        print(f"Silver records: {valid_count}")
        print(f"Quarantine records: {rejected_count}")
        print(f"Total processed: {valid_count + rejected_count}")

        print("\n========== REJECTION REASONS ==========")

        (
            rejected_df.groupBy("rejection_reason")
            .count()
            .orderBy(F.desc("count"))
            .show(100, truncate=False)
        )

        print("\n========== PAYMENT SILVER SAMPLE ==========")
        valid_df.orderBy("payment_id").show(50, truncate=False)

        print("\n========== PAYMENT QUARANTINE SAMPLE ==========")
        rejected_df.orderBy("payment_id").show(100, truncate=False)

        # -------------------------------------------------
        # 1. General data-quality assertions
        # -------------------------------------------------

        duplicate_payment_ids = (
            valid_df.groupBy("payment_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        invalid_amounts = valid_df.filter(
            F.col("amount").isNull()
            | (F.col("amount") <= 0)
        ).count()

        invalid_payment_dates = valid_df.filter(
            F.col("payment_date").isNull()
        ).count()

        invalid_foreign_keys = sum(
            valid_df.filter(
                F.col(column) != "VALID"
            ).count()
            for column in [
                "customer_reference_quality",
                "staff_reference_quality",
                "rental_reference_quality",
            ]
        )

        print("\n========== GENERAL DATA QUALITY CHECKS ==========")
        print(f"Duplicate Payment IDs: {duplicate_payment_ids}")
        print(f"Invalid/non-positive amounts: {invalid_amounts}")
        print(f"Invalid payment dates: {invalid_payment_dates}")
        print(f"Invalid foreign-key statuses: {invalid_foreign_keys}")

        assert source_count == valid_count + rejected_count, (
            "Source records do not reconcile with Silver + Quarantine."
        )

        assert duplicate_payment_ids == 0
        assert invalid_amounts == 0
        assert invalid_payment_dates == 0
        assert invalid_foreign_keys == 0

        # All retained Silver records must have valid quality statuses.
        for column in [
            "amount_quality",
            "payment_date_quality",
            "customer_reference_quality",
            "staff_reference_quality",
            "rental_reference_quality",
        ]:
            invalid_status_count = valid_df.filter(
                F.col(column) != "VALID"
            ).count()

            assert invalid_status_count == 0, (
                f"Unexpected non-VALID status in {column}."
            )

        # -------------------------------------------------
        # 2. Conflicting duplicate Payment ID assertions
        # -------------------------------------------------

        print("\n========== CONFLICTING DUPLICATE CHECKS ==========")

        # ID 7:
        # Staff ID is missing in one version and 3 in another.
        # Staff ID 3 is not present in the Staff Silver dependency.
        # Neither version may be retained.
        assert valid_df.filter(
            F.col("payment_id") == "7"
        ).count() == 0

        payment_7 = rejected_df.filter(
            F.col("payment_id").cast("string") == "7"
        )

        assert payment_7.count() == 2, (
            "Payment ID 7 must have exactly two quarantined source versions."
        )

        assert payment_7.filter(
            F.col("rejection_reason").contains(
                "CONFLICTING_DUPLICATE_PAYMENT_ID"
            )
        ).count() == 0, (
            "Payment ID 7 should not be marked as conflicting because "
            "one Staff ID is genuinely missing and is treated as a "
            "completable gap by the current business rule."
        )

        assert payment_7.filter(
            F.col("rejection_reason").contains("INVALID_STAFF_ID")
        ).count() == 1, (
            "Exactly one Payment ID 7 version should have INVALID_STAFF_ID."
        )

        assert payment_7.filter(
            F.col("rejection_reason").contains("INVALID_STAFF_REFERENCE")
        ).count() == 1, (
            "Exactly one Payment ID 7 version should have "
            "INVALID_STAFF_REFERENCE because Staff ID 3 is absent "
            "from Staff Silver."
        )

        assert valid_df.filter(
            F.col("payment_id").cast("string") == "7"
        ).count() == 0, (
            "Payment ID 7 must not enter Silver while its Staff reference "
            "is invalid."
        )

        print("Payment ID 7: PASS")

        # ID 28:
        # Amounts differ: 1,000.50 versus 11.11.
        # Both versions must be quarantined.
        assert valid_df.filter(
            F.col("payment_id") == "28"
        ).count() == 0

        assert_quarantined_twice(
            rejected_df,
            28,
            "CONFLICTING_DUPLICATE_PAYMENT_ID",
        )

        print("Payment ID 28: PASS")

        # ID 363:
        # Customer ID ABC is malformed, not missing.
        # It must not be completed using Customer ID 182.
        assert valid_df.filter(
            F.col("payment_id") == "363"
        ).count() == 0

        assert_quarantined_twice(
            rejected_df,
            363,
            "CONFLICTING_DUPLICATE_PAYMENT_ID",
        )

        assert rejected_df.filter(
            (F.col("payment_id") == "363")
            & F.col("rejection_reason").contains("INVALID_CUSTOMER_ID")
        ).count() == 1

        print("Payment ID 363: PASS")

        # ID 723:
        # Amounts differ: 11.81 versus 1,000.50.
        # Both versions must be quarantined.
        assert valid_df.filter(
            F.col("payment_id") == "723"
        ).count() == 0

        assert_quarantined_twice(
            rejected_df,
            723,
            "CONFLICTING_DUPLICATE_PAYMENT_ID",
        )

        print("Payment ID 723: PASS")

        # -------------------------------------------------
        # 3. Exact duplicate Payment ID assertion
        # -------------------------------------------------

        print("\n========== EXACT DUPLICATE CHECKS ==========")

        # ID 27 occurs three times with identical source values.
        # One copy proceeds to validation; two additional copies
        # must be quarantined as exact duplicates.
        id_27_rejected = rejected_df.filter(
            F.col("payment_id") == "27"
        )

        assert id_27_rejected.count() == 3, (
            "Payment ID 27 should have three rejected source rows "
            "under the current dependency validation."
        )

        duplicate_copies = id_27_rejected.filter(
            F.col("rejection_reason").contains(
                "DUPLICATE_PAYMENT_RECORD"
            )
        ).count()

        assert duplicate_copies == 2, (
            "Payment ID 27 should have exactly two duplicate copies."
        )

        print("Payment ID 27: PASS")

        # -------------------------------------------------
        # 4. Final reconciliation
        # -------------------------------------------------

        print("\n========== FINAL ACCEPTANCE CHECKS ==========")

        assert source_count == 952, (
            f"Expected 952 raw records, found {source_count}."
        )

        assert valid_count + rejected_count == 952

        print("Raw source records: 952")
        print(f"Silver records: {valid_count}")
        print(f"Quarantine records: {rejected_count}")
        print("Source reconciliation: PASS")
        print("Conflicting duplicate checks: PASS")
        print("Exact duplicate checks: PASS")
        print("Foreign-key checks: PASS")
        print("Amount and date checks: PASS")

        print("\nAll Payment Silver acceptance checks passed.")

        # -------------------------------------------------
        # 5. Write output only after all assertions pass
        # -------------------------------------------------

        writer = SilverWriter(
            silver_base_path=str(SILVER_PATH),
            quarantine_base_path=str(QUARANTINE_PATH),
        )

        result = writer.write(
            valid_df=valid_df,
            rejected_df=rejected_df,
            table_name="payment",
            batch_id=BATCH_ID,
            ingestion_date=INGESTION_DATE,
        )

        print("\n========== PAYMENT WRITE SUMMARY ==========")
        print(result)

        print("\nPayment Silver processing completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
