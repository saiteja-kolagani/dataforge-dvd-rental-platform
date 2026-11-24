
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "country.csv"

df = pd.read_csv(
    SOURCE_FILE,
    dtype=str,
    keep_default_na=False,
)

print("\n--- COUNTRY DATASET OVERVIEW ---")
print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\n--- SAMPLE RECORDS ---")
print(df.head(30).to_string(index=False))

print("\n--- MISSING / EMPTY VALUES ---")
for column in df.columns:
    empty_count = (
        df[column].astype(str).str.strip().eq("").sum()
    )
    print(f"{column}: {empty_count}")

print("\n--- DISTINCT VALUE COUNTS ---")
for column in df.columns:
    print(f"{column}: {df[column].nunique(dropna=False)}")

print("\n--- DUPLICATE COUNTRY IDs ---")
if "country_id" in df.columns:
    duplicates = df[
        df.duplicated(subset=["country_id"], keep=False)
    ].sort_values("country_id")

    print(duplicates.to_string(index=False))

print("\n--- COUNTRY ID DISTRIBUTION ---")
if "country_id" in df.columns:
    print(
        df["country_id"]
        .value_counts(dropna=False)
        .sort_index()
    )

print("\n--- COUNTRY NAME VARIATIONS ---")
if "country" in df.columns:
    print(
        df["country"]
        .value_counts(dropna=False)
        .to_string()
    )

print("\n--- TIMESTAMP VARIATIONS ---")
if "last_update" in df.columns:
    print(
        df["last_update"]
        .value_counts(dropna=False)
        .to_string()
    )

print("\n--- DUPLICATE COUNTRY NAMES ---")
if "country" in df.columns:
    normalized_names = df["country"].str.strip().str.lower()

    duplicate_names = df[
        normalized_names.duplicated(keep=False)
    ].copy()

    duplicate_names["normalized_country"] = (
        duplicate_names["country"].str.strip().str.lower()
    )

    print(
        duplicate_names
        .sort_values("normalized_country")
        .to_string(index=False)
    )
