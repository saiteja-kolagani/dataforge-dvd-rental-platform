
import os
import sys

# Use the active virtual environment's Python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("DataForge-Silver-Development")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.pyspark.python", sys.executable)
    .config("spark.pyspark.driver.python", sys.executable)
    .getOrCreate()
)

try:
    print("Spark Version:", spark.version)
    print("Python Executable:", sys.executable)
    print("Application:", spark.sparkContext.appName)

    data = [
        (1, "Film A", "PG"),
        (2, "Film B", "G"),
        (3, "Film C", "R"),
    ]

    df = spark.createDataFrame(
        data,
        ["film_id", "title", "rating"]
    )

    df.show()

finally:
    spark.stop()
