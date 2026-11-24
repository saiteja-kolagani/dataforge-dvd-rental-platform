
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "category.csv"

df = pd.read_csv(SOURCE_FILE, dtype=str, keep_default_na=False)

print("\n--- CATEGORY DATASET OVERVIEW ---")
print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\n--- SAMPLE RECORDS ---")
print(df.head(20).to_string(index=False))

print("\n--- MISSING / EMPTY VALUES ---")
for column in df.columns:
    empty_count = (
        df[column].astype(str).str.strip().eq("").sum()
    )
    print(f"{column}: {empty_count}")

print("\n--- DISTINCT VALUE COUNTS ---")
for column in df.columns:
    print(f"{column}: {df[column].nunique(dropna=False)}")

print("\n--- DUPLICATE CATEGORY IDs ---")
if "category_id" in df.columns:
    duplicates = df[
        df.duplicated(subset=["category_id"], keep=False)
    ].sort_values("category_id")

    print(duplicates.to_string(index=False))

print("\n--- CATEGORY ID DISTRIBUTION ---")
if "category_id" in df.columns:
    print(df["category_id"].value_counts(dropna=False).sort_index())

print("\n--- NAME VARIATIONS ---")
if "name" in df.columns:
    print(
        df["name"]
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
