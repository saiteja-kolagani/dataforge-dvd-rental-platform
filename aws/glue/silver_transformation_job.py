
import sys
from datetime import date

from awsglue.utils import getResolvedOptions
from pyspark.sql import functions as F

from src.silver.actor_processor import ActorProcessor
from src.silver.address_processor import AddressProcessor
from src.silver.category_processor import CategoryProcessor
from src.silver.city_processor import CityProcessor
from src.silver.country_processor import CountryProcessor
from src.silver.customer_processor import CustomerProcessor
from src.silver.film_actor_processor import FilmActorProcessor
from src.silver.film_category_processor import FilmCategoryProcessor
from src.silver.film_processor import FilmProcessor
from src.silver.inventory_processor import InventoryProcessor
from src.silver.language_processor import LanguageProcessor
from src.silver.payment_processor import PaymentProcessor
from src.silver.rental_processor import RentalProcessor
from src.silver.staff_processor import StaffProcessor
from src.silver.store_processor import StoreProcessor


# ---------------------------------------------------------
# 1. JOB PARAMETERS
# ---------------------------------------------------------

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "BRONZE_BUCKET",
        "SILVER_BUCKET",
        "QUARANTINE_BUCKET",
        "BATCH_ID",
        "INGESTION_DATE",
    ],
)

BRONZE_BUCKET = args["BRONZE_BUCKET"]
SILVER_BUCKET = args["SILVER_BUCKET"]
QUARANTINE_BUCKET = args["QUARANTINE_BUCKET"]
BATCH_ID = args["BATCH_ID"]
INGESTION_DATE = date.fromisoformat(
    args["INGESTION_DATE"]
).isoformat()


# ---------------------------------------------------------
# 2. SPARK SESSION
# ---------------------------------------------------------

from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

sc = SparkContext.getOrCreate()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

job = Job(glue_context)
job.init(args["JOB_NAME"], args)


# ---------------------------------------------------------
# 3. JOB SUMMARY
# ---------------------------------------------------------

summary = []


# ---------------------------------------------------------
# 4. READ BRONZE CSV
# ---------------------------------------------------------

def read_bronze(table_name):
    path = (
        f"s3://{BRONZE_BUCKET}/bronze/{table_name}/"
        f"ingestion_date={INGESTION_DATE}/"
        f"batch_id={BATCH_ID}/{table_name}.csv"
    )

    print(f"Reading Bronze: {path}")

    return (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(path)
    )


# ---------------------------------------------------------
# 5. WRITE SILVER AND QUARANTINE
# ---------------------------------------------------------

def write_output(df, bucket, layer, table_name):
    output_path = (
        f"s3://{bucket}/{layer}/{table_name}/"
        f"ingestion_date={INGESTION_DATE}/"
        f"batch_id={BATCH_ID}"
    )

    output_df = (
        df
        .withColumn("batch_id", F.lit(BATCH_ID))
        .withColumn(
            "ingestion_date",
            F.lit(INGESTION_DATE).cast("date"),
        )
    )

    # Each batch has its own output directory.
    # Reprocessing the same batch replaces only that directory.
    output_df.write.mode("overwrite").parquet(output_path)

    return output_path


def process_and_write(table_name, result):
    valid_df, rejected_df = result

    valid_count = valid_df.count()
    rejected_count = rejected_df.count()

    silver_path = write_output(
        valid_df,
        SILVER_BUCKET,
        "silver",
        table_name,
    )

    quarantine_path = write_output(
        rejected_df,
        QUARANTINE_BUCKET,
        "quarantine",
        table_name,
    )

    table_summary = {
        "table": table_name,
        "source_records": valid_count + rejected_count,
        "silver_records": valid_count,
        "quarantine_records": rejected_count,
        "silver_path": silver_path,
        "quarantine_path": quarantine_path,
    }

    summary.append(table_summary)

    print(f"Completed: {table_name}")
    print(f"Silver records: {valid_count}")
    print(f"Quarantine records: {rejected_count}")

    return valid_df


# ---------------------------------------------------------
# 6. PROCESS TABLES IN DEPENDENCY ORDER
# ---------------------------------------------------------

try:

    # Reference tables
    country = process_and_write(
        "country",
        CountryProcessor().process(read_bronze("country")),
    )

    language = process_and_write(
        "language",
        LanguageProcessor().process(read_bronze("language")),
    )

    category = process_and_write(
        "category",
        CategoryProcessor().process(read_bronze("category")),
    )

    actor = process_and_write(
        "actor",
        ActorProcessor().process(read_bronze("actor")),
    )

    # Geographic tables
    city = process_and_write(
        "city",
        CityProcessor().process(
            read_bronze("city"),
            country,
        ),
    )

    address = process_and_write(
        "address",
        AddressProcessor().process(
            read_bronze("address"),
            city,
        ),
    )

    # Store and staff relationship
    store = process_and_write(
        "store",
        StoreProcessor().process(
            read_bronze("store"),
            address,
        ),
    )

    staff = process_and_write(
        "staff",
        StaffProcessor().process(
            read_bronze("staff"),
            address,
            store,
        ),
    )

    # Customer and film
    customer = process_and_write(
        "customer",
        CustomerProcessor().process(
            read_bronze("customer"),
            store,
            address,
        ),
    )

    film = process_and_write(
        "film",
        FilmProcessor().process(
            read_bronze("film"),
            language,
        ),
    )

    # Transaction-related tables
    inventory = process_and_write(
        "inventory",
        InventoryProcessor().process(
            read_bronze("inventory"),
            film,
            store,
        ),
    )

    rental = process_and_write(
        "rental",
        RentalProcessor().process(
            read_bronze("rental"),
            inventory,
            customer,
            staff,
        ),
    )

    # Many-to-many relationship tables
    process_and_write(
        "film_actor",
        FilmActorProcessor().process(
            read_bronze("film_actor"),
            actor,
            film,
        ),
    )

    process_and_write(
        "film_category",
        FilmCategoryProcessor().process(
            read_bronze("film_category"),
            film,
            category,
        ),
    )

    # Payment depends on rental, customer and staff
    process_and_write(
        "payment",
        PaymentProcessor().process(
            read_bronze("payment"),
            customer,
            staff,
            rental,
        ),
    )

    # -----------------------------------------------------
    # 7. FINAL JOB SUMMARY
    # -----------------------------------------------------

    print("\n========== DATAFORGE SILVER JOB SUMMARY ==========")
    print(f"Batch ID: {BATCH_ID}")
    print(f"Ingestion Date: {INGESTION_DATE}")
    print(f"Tables processed: {len(summary)}")

    total_source = sum(x["source_records"] for x in summary)
    total_silver = sum(x["silver_records"] for x in summary)
    total_quarantine = sum(x["quarantine_records"] for x in summary)

    for item in summary:
        print(item)

    print(f"Total source records: {total_source}")
    print(f"Total Silver records: {total_silver}")
    print(f"Total Quarantine records: {total_quarantine}")

    if total_source != total_silver + total_quarantine:
        raise RuntimeError("Record reconciliation failed.")

    print("Silver transformation completed successfully.")

    job.commit()

except Exception:
    print("Silver transformation job failed.")
    raise
