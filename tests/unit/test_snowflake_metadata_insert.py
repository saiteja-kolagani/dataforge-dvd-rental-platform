
from datetime import date, datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from src.utils.snowflake_client import SnowflakeClient
from src.ingestion.ingestion_service import insert_ingestion_metadata, METADATA_TABLE


def test_snowflake_metadata_insert():

    snowflake_client = SnowflakeClient()

    batch_id = f"TEST_{uuid4().hex[:12]}"

    batch = SimpleNamespace(
        batch_id=batch_id,
        ingestion_timestamp=datetime.now(timezone.utc),
        ingestion_date=date.today(),
    )

    file = SimpleNamespace(
        file_name="test_actor.csv",
        table_name="actor",
        file_size_bytes=1024,
    )

    try:
        snowflake_client.connect()

        # Insert a temporary test record.
        insert_ingestion_metadata(
            snowflake_client=snowflake_client,
            batch=batch,
            file=file,
            s3_key=(
                f"bronze/actor/ingestion_date={batch.ingestion_date}/"
                f"batch_id={batch_id}/test_actor.csv"
            ),
            status="SUCCESS",
        )

        # Verify the inserted record.
        query = f"""
            SELECT COUNT(*)
            FROM {METADATA_TABLE}
            WHERE BATCH_ID = %s
        """

        result = snowflake_client.execute_query(
            query=query,
            params=(batch_id,),
        )

        assert result[0][0] == 1, "Metadata record was not found!"

        print("Snowflake metadata insertion test passed!")

    finally:

        try:
            # Remove the temporary test record.
            snowflake_client.execute_query(
                query=f"""
                    DELETE FROM {METADATA_TABLE}
                    WHERE BATCH_ID = %s
                """,
                params=(batch_id,),
            )

            print("Temporary test metadata cleaned up.")

        finally:
            snowflake_client.close()


if __name__ == "__main__":
    test_snowflake_metadata_insert()
