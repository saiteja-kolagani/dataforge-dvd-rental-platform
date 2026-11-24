
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# 1. Load raw Customer dataset
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
file_path = PROJECT_ROOT / "data" / "raw" / "customer.csv"

df = pd.read_csv(
    file_path,
    dtype=str,
    keep_default_na=False,
)

NULL_LIKE_VALUES = ["", "NULL", "NA", "N/A", "UNKNOWN", "NONE"]

print("\n--- CUSTOMER DATASET OVERVIEW ---")
print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}")

# ---------------------------------------------------------
# 2. Display sample records
# ---------------------------------------------------------
print("\n--- SAMPLE RECORDS ---")
print(df.head(30).to_string(index=False))

# ---------------------------------------------------------
# 3. Missing / empty / null-like values
# ---------------------------------------------------------
print("\n--- MISSING / NULL-LIKE VALUES ---")

for column in df.columns:
    null_mask = (
        df[column]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(NULL_LIKE_VALUES)
    )

    print(f"{column}: {null_mask.sum()}")

# ---------------------------------------------------------
# 4. Distinct value counts
# ---------------------------------------------------------
print("\n--- DISTINCT VALUE COUNTS ---")

for column in df.columns:
    print(f"{column}: {df[column].nunique()}")

# ---------------------------------------------------------
# 5. Duplicate Customer IDs
# ---------------------------------------------------------
print("\n--- DUPLICATE CUSTOMER IDs ---")

if "customer_id" in df.columns:
    duplicate_ids = df[
        df.duplicated(
            subset=["customer_id"],
            keep=False,
        )
    ].sort_values("customer_id")

    if duplicate_ids.empty:
        print("No duplicate customer IDs found.")
    else:
        print(duplicate_ids.to_string(index=False))

# ---------------------------------------------------------
# 6. Customer ID validation
# ---------------------------------------------------------
print("\n--- CUSTOMER ID VALIDATION ---")

if "customer_id" in df.columns:
    numeric_ids = pd.to_numeric(
        df["customer_id"].str.strip(),
        errors="coerce",
    )

    invalid_ids = df[
        numeric_ids.isna()
        | (numeric_ids <= 0)
        | (numeric_ids % 1 != 0)
    ]

    print(f"Invalid Customer IDs: {len(invalid_ids)}")

    if not invalid_ids.empty:
        print(invalid_ids.to_string(index=False))

# ---------------------------------------------------------
# 7. Customer name variations
# ---------------------------------------------------------
for column in ["first_name", "last_name"]:
    if column not in df.columns:
        continue

    print(f"\n--- {column.upper()} VARIATIONS ---")

    print(
        df[column]
        .value_counts(dropna=False)
        .head(40)
        .to_string()
    )

# ---------------------------------------------------------
# 8. Duplicate customer names
# ---------------------------------------------------------
print("\n--- DUPLICATE CUSTOMER NAMES ---")

if {"first_name", "last_name"}.issubset(df.columns):
    temp = df.copy()

    temp["normalized_first_name"] = (
        temp["first_name"].str.strip().str.lower()
    )

    temp["normalized_last_name"] = (
        temp["last_name"].str.strip().str.lower()
    )

    duplicate_names = temp[
        temp.duplicated(
            subset=[
                "normalized_first_name",
                "normalized_last_name",
            ],
            keep=False,
        )
    ]

    if duplicate_names.empty:
        print("No duplicate customer names found.")
    else:
        print(
            duplicate_names[
                ["customer_id", "first_name", "last_name"]
            ].to_string(index=False)
        )

# ---------------------------------------------------------
# 9. Email profiling
# ---------------------------------------------------------
print("\n--- EMAIL PROFILING ---")

if "email" in df.columns:
    email = df["email"].astype(str).str.strip()

    email_missing = (
        email.str.upper().isin(NULL_LIKE_VALUES)
    )

    email_pattern = (
        r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
        r"@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$"
    )

    email_valid = email.str.match(
        email_pattern,
        na=False,
    )

    invalid_email_mask = (~email_missing) & (~email_valid)

    print(f"Missing emails: {email_missing.sum()}")
    print(f"Valid-format emails: {(~email_missing & email_valid).sum()}")
    print(f"Invalid-format emails: {invalid_email_mask.sum()}")

    if invalid_email_mask.any():
        print("\nInvalid email records:")
        print(
            df.loc[
                invalid_email_mask,
                ["customer_id", "email"],
            ].to_string(index=False)
        )

    # Duplicate email addresses
    normalized_email = email.str.lower()

    duplicate_email_mask = (
        normalized_email.ne("")
        & normalized_email.ne("NULL")
        & normalized_email.duplicated(keep=False)
    )

    print(
        f"\nRecords with duplicate email addresses: "
        f"{duplicate_email_mask.sum()}"
    )

    if duplicate_email_mask.any():
        print(
            df.loc[
                duplicate_email_mask,
                ["customer_id", "email"],
            ].sort_values("email").to_string(index=False)
        )

# ---------------------------------------------------------
# 10. Foreign-key profiling
# ---------------------------------------------------------
for column in ["store_id", "address_id"]:
    if column not in df.columns:
        continue

    print(f"\n--- {column.upper()} DISTRIBUTION ---")

    print(
        df[column]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    numeric_values = pd.to_numeric(
        df[column].str.strip(),
        errors="coerce",
    )

    invalid_fk = df[
        numeric_values.isna()
        | (numeric_values <= 0)
        | (numeric_values % 1 != 0)
    ]

    print(f"Invalid {column} values: {len(invalid_fk)}")

    if not invalid_fk.empty:
        print(
            invalid_fk[
                ["customer_id", column]
            ].to_string(index=False)
        )

# ---------------------------------------------------------
# 11. Active / inactive status profiling
# ---------------------------------------------------------
for column in ["active", "activebool"]:
    if column not in df.columns:
        continue

    print(f"\n--- {column.upper()} DISTRIBUTION ---")

    print(
        df[column]
        .value_counts(dropna=False)
        .to_string()
    )

# ---------------------------------------------------------
# 12. Date / timestamp profiling
# ---------------------------------------------------------
for column in ["create_date", "last_update"]:
    if column not in df.columns:
        continue

    print(f"\n--- {column.upper()} VARIATIONS ---")

    print(
        df[column]
        .value_counts(dropna=False)
        .head(30)
        .to_string()
    )

# ---------------------------------------------------------
# 13. Whitespace and casing anomalies
# ---------------------------------------------------------
print("\n--- WHITESPACE / CASING ANOMALIES ---")

for column in ["first_name", "last_name", "email"]:
    if column not in df.columns:
        continue

    values = df[column].astype(str)

    anomalies = df[
        values.ne(values.str.strip())
    ]

    print(f"\n{column}: {len(anomalies)} whitespace anomalies")

    if not anomalies.empty:
        print(
            anomalies[
                ["customer_id", column]
            ].head(30).to_string(index=False)
        )

# ---------------------------------------------------------
# 14. Final profiling summary
# ---------------------------------------------------------
print("\n--- CUSTOMER PROFILING COMPLETED ---")
