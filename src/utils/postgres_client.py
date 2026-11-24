
import psycopg2

from src.utils.setup_logger import logger


class PostgreSQLClient:

    def __init__(
        self,
        host: str,
        port: str,
        database: str,
        user: str,
        password: str,
        connect_timeout: int = 10,
        sslmode: str = "prefer",
    ):
        try:
            self.connection = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                connect_timeout=connect_timeout,
                sslmode=sslmode,
            )

            logger.info("PostgreSQL connection established successfully.")

        except psycopg2.Error:
            logger.exception("Failed to establish PostgreSQL connection.")
            raise

    def insert_ingestion_metadata(
        self,
        batch_id,
        ingestion_timestamp,
        ingestion_date,
        source_system,
        source_file,
        source_table,
        s3_bucket,
        s3_key,
        file_size_bytes,
        status,
        error_message=None,
    ):

        if self.connection.closed:
            raise ConnectionError(
                "PostgreSQL connection is already closed."
            )

        query = """
            INSERT INTO dvd_rental_ingestion_metadata (
                batch_id,
                ingestion_timestamp,
                ingestion_date,
                source_system,
                source_file,
                source_table,
                s3_bucket,
                s3_key,
                file_size_bytes,
                status,
                error_message
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            );
        """

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        batch_id,
                        ingestion_timestamp,
                        ingestion_date,
                        source_system,
                        source_file,
                        source_table,
                        s3_bucket,
                        s3_key,
                        file_size_bytes,
                        status,
                        error_message,
                    ),
                )

            self.connection.commit()

            logger.info(
                f"Metadata recorded successfully for "
                f"file {source_file}, batch {batch_id}, status {status}"
            )

        except Exception:
            try:
                if not self.connection.closed:
                    self.connection.rollback()
            except Exception:
                logger.exception(
                    "PostgreSQL rollback failed."
                )

            logger.exception(
                f"Failed to insert ingestion metadata "
                f"for file {source_file}, batch {batch_id}"
            )
            raise

    def close(self):

        if getattr(self, "connection", None) and not self.connection.closed:
            try:
                self.connection.close()
                logger.info("PostgreSQL connection closed.")

            except psycopg2.Error:
                logger.exception("Failed to close PostgreSQL connection.")
                raise
