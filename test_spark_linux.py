import os
import sys
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = (
    SparkSession.builder
    .appName("DataForge-Linux-Test")
    .master("local[2]")
    .getOrCreate()
)

print("Spark version:", spark.version)

data = [(1, "Silver"), (2, "Quarantine")]
df = spark.createDataFrame(data, ["id", "layer"])

df.show()

spark.stop()