
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = PROJECT_ROOT / "data" / "raw" / "address.csv"

NULL_VALUES = {"", "NULL", "NA", "N/A", "UNKNOWN", "NONE"}

print("\nLoading Address dataset...")

if not FILE_PATH.exists():
    raise FileNotFoundError(f"Address dataset not found: {FILE_PATH}")

df = pd.read_csv(
    FILE_PATH,
    dtype=str,
    keep_default_na=False
)

print("\n--- ADDRESS DATASET OVERVIEW ---")
print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\n--- SAMPLE RECORDS ---")
print(df.head(30).to_string(index=False))

print("\n--- MISSING / NULL-LIKE VALUES ---")
for column in df.columns:
    count = df[column].astype(str).str.strip().str.upper().isin(
        NULL_VALUES
    ).sum()
    print(f"{column}: {count}")

print("\n--- DISTINCT VALUE COUNTS ---")
for column in df.columns:
    print(f"{column}: {df[column].nunique(dropna=False)}")

# Primary-key profiling
if "address_id" in df.columns:
    print("\n--- DUPLICATE ADDRESS IDs ---")

    duplicate_mask = df["address_id"].duplicated(keep=False)

    duplicates = df[duplicate_mask].sort_values("address_id")

    if duplicates.empty:
        print("No duplicate address IDs found.")
    else:
        print(duplicates.to_string(index=False))

    print("\n--- ADDRESS ID VALIDATION ---")

    numeric_ids = pd.to_numeric(
        df["address_id"].str.strip(),
        errors="coerce"
    )

    invalid_mask = (
        numeric_ids.isna()
        | (numeric_ids <= 0)
        | (numeric_ids % 1 != 0)
    )

    print(f"Invalid address IDs: {invalid_mask.sum()}")

    if invalid_mask.any():
        print(df.loc[invalid_mask].to_string(index=False))

# String-column profiling
print("\n--- STRING VALUE DISTRIBUTIONS ---")

for column in df.columns:
    if column in {"address_id", "last_update"}:
        continue

    print(f"\n{column}:")

    values = df[column].value_counts(dropna=False)

    print(values.head(30).to_string())

# Foreign-key profiling
for column in ["city_id"]:
    if column in df.columns:
        print(f"\n--- {column.upper()} VALIDATION ---")

        numeric_values = pd.to_numeric(
            df[column].str.strip(),
            errors="coerce"
        )

        invalid_mask = (
            numeric_values.isna()
            | (numeric_values <= 0)
            | (numeric_values % 1 != 0)
        )

        print(f"Invalid {column} values: {invalid_mask.sum()}")

        if invalid_mask.any():
            print(
                df.loc[invalid_mask, ["address_id", column]]
                .to_string(index=False)
            )

# Timestamp profiling
for column in ["last_update"]:
    if column in df.columns:
        print(f"\n--- {column.upper()} VARIATIONS ---")

        print(
            df[column]
            .value_counts(dropna=False)
            .to_string()
        )

# Whitespace and casing inspection
print("\n--- WHITESPACE / CASING ANOMALIES ---")

for column in df.columns:
    if column in {"address_id", "last_update"}:
        continue

    values = df[column].astype(str)

    anomalies = df[
        values.ne(values.str.strip())
        | values.ne(values.str.title())
    ]

    if not anomalies.empty:
        print(f"\n{column}: {len(anomalies)} potential anomalies")
        print(anomalies.head(30).to_string(index=False))

print("\n--- ADDRESS PROFILING COMPLETED ---")
