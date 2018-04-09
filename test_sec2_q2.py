"""
Author:
Date:
TOpic:
"""

###########################################################################################3
# COFIGURATION
###########################################################################################3

from pyspark.sql import SparkSession
from pyspark.mllib.regression import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import sys
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GBTRegressor
from math import sqrt

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("test") \
    .getOrCreate()

sc = spark.sparkContext

sc.setLogLevel("ERROR")
sc.setCheckpointDir('C://Users/Ravi/PycharmProjects/WeblogChallenge/checkpoint/')

# print(sc)

REPARTITION_FACTOR = int(sc._jsc.sc().getExecutorMemoryStatus().size()) * 10
# print(REPARTITION_FACTOR)

# UTILS FUNCTIONS
_minutesLambda = lambda i: i * 60

###########################################################################################3


###########################################################################################3
# DATA INGESTION
###########################################################################################3

raw_data = spark.read.option("delimiter", " ").csv("C://Users/Ravi/PycharmProjects/WeblogChallenge/data") \
    .sample(False, 0.001, 42)

# print(raw_data.count())

print("Ingesting Data...\n ")

raw_data_w_cols = raw_data \
    .withColumnRenamed("_c0", "timestamp") \
    .withColumnRenamed("_c1", "elb") \
    .withColumnRenamed("_c2", "client") \
    .withColumnRenamed("_c3", "backend") \
    .withColumnRenamed("_c4", "request_processing_time") \
    .withColumnRenamed("_c5", "backend_processing_time") \
    .withColumnRenamed("_c6", "response_processing_time") \
    .withColumnRenamed("_c7", "elb_status_code") \
    .withColumnRenamed("_c8", "backend_status_code") \
    .withColumnRenamed("_c9", "received_bytes") \
    .withColumnRenamed("_c10", "sent_bytes") \
    .withColumnRenamed("_c11", "request") \
    .withColumnRenamed("_c12", "user_agent") \
    .withColumnRenamed("_c13", "ssl_cipher") \
    .withColumnRenamed("_c14", "ssl_protocol") \
    .repartition(REPARTITION_FACTOR)

# print(raw_data_w_cols.select(col("ssl_protocol")).distinct().count())

raw_data_w_cols_clean = raw_data_w_cols \
    .withColumn("request_processing_time_clean",
                when(col("request_processing_time") == "-1", None).otherwise(col("request_processing_time"))) \
    .withColumn("backend_processing_time_clean",
                when(col("backend_processing_time") == "-1", None).otherwise(col("backend_processing_time"))) \
    .withColumn("response_processing_time_clean",
                when(col("response_processing_time") == "-1", None).otherwise(col("response_processing_time"))) \
    .withColumn("elb_status_code_clean", concat_ws("_", lit("elb_status"), col("elb_status_code"))) \
    .drop("elb_status_code") \
    .withColumnRenamed("elb_status_code_clean", "elb_status_code") \
    .withColumn("backend_status_code_clean", concat_ws("_", lit("backend_status"), col("backend_status_code"))) \
    .drop("backend_status_code") \
    .withColumnRenamed("backend_status_code_clean", "backend_status_code")

# print(raw_data_w_cols_clean.select(col("user_agent")).distinct().show())


print("Pre-processing Data...")
_pre_proc = raw_data_w_cols_clean \
    .withColumn("IP", split(col("client"), ":").getItem(0)) \
    .withColumn("request_split", split(col("request"), " ")) \
    .withColumn("request_type", col("request_split").getItem(0)) \
    .withColumn("URL",
                lower(split(split(col("request_split").getItem(1), "/").getItem(2), ":").getItem(0).cast(StringType()))) \
    .withColumn("http_version", col("request_split").getItem(2)) \
    .withColumn("unix_tmpstmp", to_utc_timestamp(col("timestamp"), "ISO 8601")) \
    .withColumn("date", to_date(col("unix_tmpstmp"))) \
    .withColumn("hour", hour(col("unix_tmpstmp"))) \
    .withColumn("minute", minute(col("unix_tmpstmp"))) \
    .withColumn("dummy_count", lit(1.0)) \
    .replace(["\"GET"], ["GET"], "request_type") \
    .withColumn("lagged_tmpstmp", lag(col("unix_tmpstmp"), 1).over(Window.partitionBy("IP").orderBy("unix_tmpstmp"))) \
    .withColumn("new_session",
                coalesce(unix_timestamp(col("unix_tmpstmp")) - unix_timestamp(col("lagged_tmpstmp")), lit(0.0)).cast(
                    FloatType()) > 1800.0) \
    .withColumn("sessionized_temp", sum(when(col("new_session") == False, 0.0).otherwise(1.0)).over(
    Window.partitionBy("IP").orderBy("unix_tmpstmp"))) \
    .withColumn("sessionized", concat_ws("_", col("IP").cast(StringType()), lit("Session"),
                                         col("sessionized_temp").cast(StringType()))) \
    .drop("new_session") \
    .drop("sessionized_temp") \
    .drop("lagged_tmpstmp") \
    .repartition(REPARTITION_FACTOR)

# print("_pre_proc schema:")
# _pre_proc.printSchema()

# creating feature set 1

model_input_1 = _pre_proc \
    .withColumn("IP_URL_visit", count("URL").over(
    Window.partitionBy("IP").orderBy("unix_tmpstmp").rangeBetween(Window.unboundedPreceding,
                                                                  Window.unboundedFollowing)))
# .groupby("sessionized","IP_URL_visit")\
# .agg(min(col("unix_tmpstmp")).alias("session_start_hour"))


model_input_1.select(col("IP_URL_visit")).distinct().orderBy(["IP_URL_visit"], ascending=[0, 1]).show(5)
