from pyspark.sql import SparkSession, functions as F

RAW_BASE = "data/raw"
SILVER_BASE = "data/processed/silver"
QUARANTINE_BASE = "data/processed/quarantine"

# Exact accepted partitions only. Do not read table roots: they may contain
# older test batches and would inflate counts.
BATCHES = {
    "actor": ("2026-11-22", "20261122_actor_test_01"),
    "address": ("2026-11-22", "20261122_address_test_01"),
    "category": ("2026-11-22", "20261122_category_test_01"),
    "city": ("2026-11-22", "20261122_city_test_01"),
    "country": ("2026-11-22", "20261122_country_test_01"),
    "customer": ("2026-11-22", "20261122_customer_test_02"),
    "film": ("2026-11-22", "20261122_film_test_01"),
    "film_actor": ("2026-11-23", "20261123_film_actor_test_01"),
    "film_category": ("2026-11-23", "20261123_film_category_test_01"),
    "inventory": ("2026-11-22", "20261122_inventory_test_02"),
    "language": ("2026-11-22", "20261122_language_test"),
    "payment": ("2026-11-23", "20261123_payment_test_03"),
    "rental": ("2026-11-23", "20261123_rental_test_01"),
    "staff": ("2026-11-22", "20261122_staff_test_01"),
    "store": ("2026-11-22", "20261122_store_test_01"),
}

PRIMARY_KEYS = {
    "actor": ["actor_id"],
    "address": ["address_id"],
    "category": ["category_id"],
    "city": ["city_id"],
    "country": ["country_id"],
    "customer": ["customer_id"],
    "film": ["film_id"],
    "film_actor": ["actor_id", "film_id"],
    "film_category": ["film_id", "category_id"],
    "inventory": ["inventory_id"],
    "language": ["language_id"],
    "payment": ["payment_id"],
    "rental": ["rental_id"],
    "staff": ["staff_id"],
    "store": ["store_id"],
}

# (source table, source FK, reference table, reference PK, policy)
# VERIFIED: missing positive reference is a failure.
# UNRESOLVED: missing positive reference is reported separately and allowed
# for the currently provisional/circular relationships.
FOREIGN_KEYS = [
    ("address", "city_id", "city", "city_id", "VERIFIED"),
    ("city", "country_id", "country", "country_id", "VERIFIED"),
    ("customer", "address_id", "address", "address_id", "VERIFIED"),
    ("customer", "store_id", "store", "store_id", "UNRESOLVED"),
    ("film", "language_id", "language", "language_id", "VERIFIED"),
    ("inventory", "film_id", "film", "film_id", "VERIFIED"),
    ("inventory", "store_id", "store", "store_id", "UNRESOLVED"),
    ("rental", "inventory_id", "inventory", "inventory_id", "VERIFIED"),
    ("rental", "customer_id", "customer", "customer_id", "VERIFIED"),
    ("rental", "staff_id", "staff", "staff_id", "UNRESOLVED"),
    ("payment", "customer_id", "customer", "customer_id", "VERIFIED"),
    ("payment", "staff_id", "staff", "staff_id", "VERIFIED"),
    ("payment", "rental_id", "rental", "rental_id", "VERIFIED"),
    ("film_actor", "actor_id", "actor", "actor_id", "VERIFIED"),
    ("film_actor", "film_id", "film", "film_id", "VERIFIED"),
    ("film_category", "film_id", "film", "film_id", "VERIFIED"),
    ("film_category", "category_id", "category", "category_id", "VERIFIED"),
    ("staff", "address_id", "address", "address_id", "VERIFIED"),
    ("staff", "store_id", "store", "store_id", "VERIFIED"),
    ("store", "address_id", "address", "address_id", "VERIFIED"),
    ("store", "manager_staff_id", "staff", "staff_id", "UNRESOLVED"),
]


def read_batch(spark, base, table, date, batch):
    path = (
        f"{base}/{table}/"
        f"ingestion_date={date}/batch_id={batch}"
    )
    try:
        return spark.read.parquet(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read expected {table} partition: {path}\n{exc}"
        ) from exc


def numeric_key(col_name):
    return F.expr(f"try_cast(trim(cast(`{col_name}` as string)) AS BIGINT)")


def main():
    spark = (
        SparkSession.builder
        .appName("DataForgeSilverConsolidatedAudit")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    silver = {}
    quarantine = {}
    raw_counts = {}
    silver_counts = {}
    quarantine_counts = {}
    failures = []
    warnings = []

    try:
        print("\n========== 1. LOAD EXACT ACCEPTED BATCHES ==========")
        for table, (date, batch) in BATCHES.items():
            raw_path = f"{RAW_BASE}/{table}.csv"
            raw_df = (
                spark.read.option("header", True)
                .option("inferSchema", False)
                .csv(raw_path)
            )
            silver_df = read_batch(
                spark, SILVER_BASE, table, date, batch
            )
            quarantine_df = read_batch(
                spark, QUARANTINE_BASE, table, date, batch
            )

            raw_counts[table] = raw_df.count()
            silver_counts[table] = silver_df.count()
            quarantine_counts[table] = quarantine_df.count()
            silver[table] = silver_df
            quarantine[table] = quarantine_df

            print(
                f"{table:15} raw={raw_counts[table]:4} "
                f"silver={silver_counts[table]:4} "
                f"quarantine={quarantine_counts[table]:4} "
                f"batch={batch}"
            )

        print("\n========== 2. SOURCE RECONCILIATION ==========")
        reconciliation_total = 0
        for table in BATCHES:
            expected = raw_counts[table]
            actual = silver_counts[table] + quarantine_counts[table]
            passed = expected == actual
            reconciliation_total += expected
            print(
                f"{table:15} source={expected:4} "
                f"silver+quarantine={actual:4} "
                f"{'PASS' if passed else 'FAIL'}"
            )
            if not passed:
                failures.append(
                    f"{table}: source reconciliation {expected} != {actual}"
                )

        print("Total raw source rows:", reconciliation_total)
        print(
            "Total accepted Silver rows:",
            sum(silver_counts.values()),
        )
        print(
            "Total Quarantine rows:",
            sum(quarantine_counts.values()),
        )

        print("\n========== 3. PRIMARY / COMPOSITE KEY AUDIT ==========")
        for table, keys in PRIMARY_KEYS.items():
            df = silver[table]
            missing_cols = [c for c in keys if c not in df.columns]
            if missing_cols:
                failures.append(
                    f"{table}: missing primary-key columns {missing_cols}"
                )
                print(f"{table:15} FAIL missing columns {missing_cols}")
                continue

            invalid_condition = None
            for key in keys:
                key_value = numeric_key(key)
                condition = key_value.isNull() | (key_value <= 0)
                invalid_condition = (
                    condition if invalid_condition is None
                    else invalid_condition | condition
                )

            invalid_count = df.filter(invalid_condition).count()
            duplicate_count = (
                df.groupBy(*keys)
                .count()
                .filter(F.col("count") > 1)
                .count()
            )

            status = "PASS" if invalid_count == 0 and duplicate_count == 0 else "FAIL"
            print(
                f"{table:15} invalid_key_rows={invalid_count:4} "
                f"duplicate_keys={duplicate_count:4} {status}"
            )
            if invalid_count:
                failures.append(f"{table}: {invalid_count} invalid primary-key rows")
            if duplicate_count:
                failures.append(f"{table}: {duplicate_count} duplicate primary keys")

        print("\n========== 4. FOREIGN-KEY AUDIT ==========")
        fk_totals = {}
        for src_table, src_key, ref_table, ref_key, policy in FOREIGN_KEYS:
            src_df = silver[src_table]
            ref_df = silver[ref_table]

            if src_key not in src_df.columns or ref_key not in ref_df.columns:
                message = (
                    f"{src_table}.{src_key} -> {ref_table}.{ref_key}: "
                    "required column missing"
                )
                failures.append(message)
                print("FAIL", message)
                continue

            refs = (
                ref_df.select(numeric_key(ref_key).alias("_ref_key"))
                .filter(F.col("_ref_key").isNotNull() & (F.col("_ref_key") > 0))
                .distinct()
                .withColumn("_ref_exists", F.lit(True))
            )

            checked = (
                src_df
                .withColumn("_fk_value", numeric_key(src_key))
                .join(
                    F.broadcast(refs),
                    F.col("_fk_value") == F.col("_ref_key"),
                    "left",
                )
            )

            invalid_id = checked.filter(
                F.col("_fk_value").isNull() | (F.col("_fk_value") <= 0)
            ).count()

            missing_positive = checked.filter(
                (F.col("_fk_value") > 0) & F.col("_ref_exists").isNull()
            ).count()

            if policy == "VERIFIED":
                invalid_reference = missing_positive
                unresolved = 0
            else:
                invalid_reference = 0
                unresolved = missing_positive

            key = f"{src_table}.{src_key}->{ref_table}.{ref_key}"
            fk_totals[key] = (
                invalid_id, invalid_reference, unresolved
            )

            status = (
                "PASS"
                if invalid_id == 0 and invalid_reference == 0
                else "FAIL"
            )
            print(
                f"{key:48} invalid_id={invalid_id:3} "
                f"invalid_ref={invalid_reference:3} "
                f"unresolved={unresolved:3} "
                f"policy={policy:10} {status}"
            )

            if invalid_id:
                failures.append(f"{key}: {invalid_id} invalid FK IDs")
            if invalid_reference:
                failures.append(
                    f"{key}: {invalid_reference} missing verified references"
                )
            if unresolved:
                warnings.append(
                    f"{key}: {unresolved} unresolved provisional references"
                )

        
        print("\n========== 5. TIMESTAMP QUALITY SUMMARY ==========")

        for table, df in silver.items():
            quality_columns = [
                c for c in df.columns
                if c.endswith("_quality")
                and any(
                    keyword in c.lower()
                    for keyword in (
                        "date",
                        "time",
                        "timestamp",
                    )
                )
            ]

            if not quality_columns:
                print(f"{table:15} No timestamp quality columns found")
                continue

            for col in sorted(quality_columns):
                counts = {
                    row[col] if row[col] is not None else "NULL": row["count"]
                    for row in df.groupBy(col).count().collect()
                }

                print(f"{table:15} {col}: {counts}")

        print("\n========== 6. PASSWORD / SENSITIVE FIELD CHECK ==========")
        for table in ["staff"]:
            for dataset_name, df in [
                ("Silver", silver[table]),
                ("Quarantine", quarantine[table]),
            ]:
                sensitive = [
                    c for c in df.columns
                    if "password" in c.lower()
                ]
                print(
                    f"{table} {dataset_name}: "
                    f"password_columns={sensitive}"
                )
                if sensitive:
                    failures.append(
                        f"{table} {dataset_name}: password columns present"
                    )

        print("\n========== 7. FINAL AUDIT SUMMARY ==========")
        print("Tables audited:", len(BATCHES))
        print("Source rows:", reconciliation_total)
        print("Silver rows:", sum(silver_counts.values()))
        print("Quarantine rows:", sum(quarantine_counts.values()))
        print("Failures:", len(failures))
        print("Unresolved-reference warnings:", len(warnings))

        if warnings:
            print("\n--- UNRESOLVED REFERENCES (REVIEW) ---")
            for warning in warnings:
                print("WARNING:", warning)

        if failures:
            print("\n--- AUDIT FAILURES ---")
            for failure in failures:
                print("FAIL:", failure)
            raise AssertionError(
                f"Silver audit failed with {len(failures)} issue(s). "
                "Review the details above."
            )

        print("\nPASS: Consolidated Silver audit checks completed.")
        if warnings:
            print(
                "NOTE: Unresolved provisional references remain. "
                "They are reported, not treated as verified relationships."
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
