
import snowflake.connector

from src.config.settings import (
    SNOWFLAKE_ACCOUNT,
    SNOWFLAKE_USER,
    SNOWFLAKE_PASSWORD,
    SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_SCHEMA,
    SNOWFLAKE_ROLE,
)
from src.utils.setup_logger import logger


class SnowflakeClient:
    """Manages Snowflake database connectivity and SQL execution."""

    def __init__(self):
        self.connection = None

    def connect(self):
        """Establish a connection to Snowflake."""

        required_settings = {
            "SNOWFLAKE_ACCOUNT": SNOWFLAKE_ACCOUNT,
            "SNOWFLAKE_USER": SNOWFLAKE_USER,
            "SNOWFLAKE_PASSWORD": SNOWFLAKE_PASSWORD,
            "SNOWFLAKE_WAREHOUSE": SNOWFLAKE_WAREHOUSE,
            "SNOWFLAKE_DATABASE": SNOWFLAKE_DATABASE,
            "SNOWFLAKE_SCHEMA": SNOWFLAKE_SCHEMA,
            "SNOWFLAKE_ROLE": SNOWFLAKE_ROLE,
        }

        missing_settings = [
            name
            for name, value in required_settings.items()
            if not value
        ]

        if missing_settings:
            raise ValueError(
                f"Missing Snowflake settings: {', '.join(missing_settings)}"
            )

        try:
            self.connection = snowflake.connector.connect(
                account=SNOWFLAKE_ACCOUNT,
                user=SNOWFLAKE_USER,
                password=SNOWFLAKE_PASSWORD,
                warehouse=SNOWFLAKE_WAREHOUSE,
                database=SNOWFLAKE_DATABASE,
                schema=SNOWFLAKE_SCHEMA,
                role=SNOWFLAKE_ROLE,
                login_timeout=30,
                network_timeout=60,
                client_session_keep_alive=False,
            )

            logger.info("Snowflake connection established successfully.")
            return self.connection

        except snowflake.connector.Error:
            logger.exception("Failed to establish Snowflake connection.")
            raise

    def execute_query(self, query: str, params=None):
        """
        Execute SQL and return rows for result-producing statements.
        Return row count for statements without result sets.
        """

        if self.connection is None:
            raise ConnectionError(
                "Snowflake connection is not established."
            )

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)

                if cursor.description is not None:
                    return cursor.fetchall()

                return cursor.rowcount

        except snowflake.connector.Error:
            logger.exception("Snowflake query execution failed.")
            raise

    def close(self):
        """Close the Snowflake connection safely."""

        if self.connection is not None:
            try:
                self.connection.close()
                self.connection = None
                logger.info("Snowflake connection closed successfully.")

            except snowflake.connector.Error:
                logger.exception("Failed to close Snowflake connection.")
                raise
