
from src.utils.snowflake_client import SnowflakeClient


client = SnowflakeClient()

try:
    client.connect()

    result = client.execute_query("""
        SELECT
            CURRENT_USER(),
            CURRENT_ROLE(),
            CURRENT_WAREHOUSE(),
            CURRENT_DATABASE(),
            CURRENT_SCHEMA()
    """)

    print("\nSnowflake connection successful!")
    print("Connection details:", result[0])

finally:
    client.close()
