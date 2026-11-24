
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "city.csv"

df = pd.read_csv(
    SOURCE_FILE,
    dtype=str,
    keep_default_na=False,
)

print("\n--- CITY DATASET OVERVIEW ---")
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

print("\n--- DUPLICATE CITY IDs ---")
if "city_id" in df.columns:
    duplicates = df[
        df.duplicated(subset=["city_id"], keep=False)
    ].sort_values("city_id")

    print(duplicates.to_string(index=False))

print("\n--- CITY ID DISTRIBUTION ---")
if "city_id" in df.columns:
    print(
        df["city_id"]
        .value_counts(dropna=False)
        .sort_index()
    )

print("\n--- CITY NAME VARIATIONS ---")
if "city" in df.columns:
    print(
        df["city"]
        .value_counts(dropna=False)
        .to_string()
    )

print("\n--- COUNTRY ID DISTRIBUTION ---")
if "country_id" in df.columns:
    print(
        df["country_id"]
        .value_counts(dropna=False)
        .sort_index()
    )

print("\n--- TIMESTAMP VARIATIONS ---")
if "last_update" in df.columns:
    print(
        df["last_update"]
        .value_counts(dropna=False)
        .to_string()
    )

print("\n--- DUPLICATE CITY NAMES ---")
if "city" in df.columns:
    normalized_names = df["city"].str.strip().str.lower()

    duplicate_names = df[
        normalized_names.duplicated(keep=False)
    ].copy()

    duplicate_names["normalized_city"] = (
        duplicate_names["city"].str.strip().str.lower()
    )

    print(
        duplicate_names
        .sort_values("normalized_city")
        .to_string(index=False)
    )

print("\n--- INVALID OR NULL-LIKE VALUES ---")
for column in ["city_id", "city", "country_id", "last_update"]:
    if column in df.columns:
        null_like = df[
            df[column].str.strip().str.upper().isin(
                ["", "NULL", "NA", "N/A", "UNKNOWN"]
            )
        ]

        print(f"\n{column}: {len(null_like)} null-like records")
        if not null_like.empty:
            print(null_like.to_string(index=False))
