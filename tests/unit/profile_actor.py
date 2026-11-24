
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# 1. Load raw Actor dataset
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
file_path = PROJECT_ROOT / "data" / "raw" / "actor.csv"

df = pd.read_csv(
    file_path,
    dtype=str,
    keep_default_na=False,
)

print("\n--- ACTOR DATASET OVERVIEW ---")
print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}")

# ---------------------------------------------------------
# 2. Display sample records
# ---------------------------------------------------------
print("\n--- SAMPLE RECORDS ---")
print(df.head(30).to_string(index=False))

# ---------------------------------------------------------
# 3. Missing / empty values
# ---------------------------------------------------------
print("\n--- MISSING / EMPTY VALUES ---")

for column in df.columns:
    empty_count = (
        df[column]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )
    print(f"{column}: {empty_count}")

# ---------------------------------------------------------
# 4. Distinct value counts
# ---------------------------------------------------------
print("\n--- DISTINCT VALUE COUNTS ---")

for column in df.columns:
    print(f"{column}: {df[column].nunique()}")

# ---------------------------------------------------------
# 5. Duplicate Actor IDs
# ---------------------------------------------------------
print("\n--- DUPLICATE ACTOR IDs ---")

if "actor_id" in df.columns:
    duplicate_ids = df[
        df.duplicated(subset=["actor_id"], keep=False)
    ].sort_values("actor_id")

    if duplicate_ids.empty:
        print("No duplicate actor IDs found.")
    else:
        print(duplicate_ids.to_string(index=False))

# ---------------------------------------------------------
# 6. Actor ID distribution
# ---------------------------------------------------------
print("\n--- ACTOR ID DISTRIBUTION ---")

if "actor_id" in df.columns:
    print(
        df["actor_id"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

# ---------------------------------------------------------
# 7. First name variations
# ---------------------------------------------------------
print("\n--- FIRST NAME VARIATIONS ---")

if "first_name" in df.columns:
    print(
        df["first_name"]
        .value_counts(dropna=False)
        .to_string()
    )

# ---------------------------------------------------------
# 8. Last name variations
# ---------------------------------------------------------
print("\n--- LAST NAME VARIATIONS ---")

if "last_name" in df.columns:
    print(
        df["last_name"]
        .value_counts(dropna=False)
        .to_string()
    )

# ---------------------------------------------------------
# 9. Duplicate actor names
# ---------------------------------------------------------
print("\n--- DUPLICATE ACTOR NAMES ---")

if {"first_name", "last_name"}.issubset(df.columns):
    df["normalized_first_name"] = (
        df["first_name"].str.strip().str.lower()
    )
    df["normalized_last_name"] = (
        df["last_name"].str.strip().str.lower()
    )

    duplicate_names = df[
        df.duplicated(
            subset=[
                "normalized_first_name",
                "normalized_last_name",
            ],
            keep=False,
        )
    ].sort_values(
        ["normalized_last_name", "normalized_first_name"]
    )

    if duplicate_names.empty:
        print("No duplicate actor names found.")
    else:
        print(
            duplicate_names[
                ["actor_id", "first_name", "last_name", "last_update"]
            ].to_string(index=False)
        )

# ---------------------------------------------------------
# 10. Timestamp variations
# ---------------------------------------------------------
print("\n--- TIMESTAMP VARIATIONS ---")

if "last_update" in df.columns:
    print(
        df["last_update"]
        .value_counts(dropna=False)
        .to_string()
    )

# ---------------------------------------------------------
# 11. Null-like values
# ---------------------------------------------------------
print("\n--- INVALID OR NULL-LIKE VALUES ---")

null_like_values = [
    "",
    "NULL",
    "NA",
    "N/A",
    "UNKNOWN",
]

for column in df.columns:
    null_mask = (
        df[column]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(null_like_values)
    )

    null_records = df[null_mask]

    print(f"\n{column}: {len(null_records)} null-like records")

    if not null_records.empty:
        print(null_records.to_string(index=False))

# ---------------------------------------------------------
# 12. Whitespace and casing anomalies
# ---------------------------------------------------------
print("\n--- WHITESPACE / CASING ANOMALIES ---")

for column in ["first_name", "last_name"]:
    if column not in df.columns:
        continue

    values = df[column].astype(str)

    anomalies = df[
        values.ne(values.str.strip())
        | values.ne(values.str.title())
    ]

    print(f"\n{column}: {len(anomalies)} records")

    if not anomalies.empty:
        print(anomalies.to_string(index=False))

# ---------------------------------------------------------
# 13. Actor ID numeric validation
# ---------------------------------------------------------
print("\n--- ACTOR ID VALIDATION ---")

if "actor_id" in df.columns:
    numeric_ids = pd.to_numeric(
        df["actor_id"].str.strip(),
        errors="coerce",
    )

    invalid_ids = df[
        numeric_ids.isna()
        | (numeric_ids <= 0)
        | (numeric_ids % 1 != 0)
    ]

    print(f"Invalid Actor IDs: {len(invalid_ids)}")

    if not invalid_ids.empty:
        print(invalid_ids.to_string(index=False))

print("\n--- ACTOR PROFILING COMPLETED ---")
