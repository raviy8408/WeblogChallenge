from pyspark.sql import SparkSession
from pyspark.mllib.regression import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import sys

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("test") \
    .getOrCreate()

sc = spark.sparkContext

sc.setLogLevel("ERROR")

REPARTITION_FACTOR = int(sc._jsc.sc().getExecutorMemoryStatus().size()) * 10
print(REPARTITION_FACTOR)

raw_data = spark.read.option("delimiter", " ").csv("C://Users/Ravi/PycharmProjects/WeblogChallenge/data")
# .sample(False, 0.01, 42)

# print(raw_data.count())

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

# print(raw_data_w_cols.select(col("elb_status_code")).distinct().show())

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

print(raw_data_w_cols_clean.select(col("user_agent")).take(3))

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
    .replace(["\"GET"], ["GET"], "request_type")

# _pre_proc.select(min(col("unix_tmpstmp")),max(col("unix_tmpstmp"))).show()
# _pre_proc.select(col("hour"), col("minute")).distinct().orderBy("hour").show(110)

_model_input_1 = _pre_proc \
    .withColumn("row_count", count(col("date")).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize))) \
    .withColumn("avg_request_processing_time", mean(col("request_processing_time_clean").cast(FloatType())).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize))) \
    .withColumn("avg_backend_processing_time", mean(col("backend_processing_time_clean").cast(FloatType())).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize))) \
    .withColumn("avg_response_processing_time", mean(col("response_processing_time_clean").cast(FloatType())).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize))) \
    .groupBy(["date", "hour", "minute", "row_count", "avg_request_processing_time", "avg_backend_processing_time",
              "avg_response_processing_time"]) \
    .pivot("request_type").sum("dummy_count") \
    .na.fill(0.0)

_model_input_2 = _pre_proc \
    .groupBy(["date", "hour", "minute"]) \
    .pivot("elb_status_code").sum("dummy_count") \
    .na.fill(0.0)

_model_input_3 = _pre_proc \
    .groupBy(["date", "hour", "minute"]) \
    .pivot("backend_status_code").sum("dummy_count") \
    .na.fill(0.0)

_model_input_4 = _pre_proc \
    .select(col("date"), col("hour"), col("minute"), col("received_bytes"), col("sent_bytes")) \
    .groupBy(["date", "hour", "minute"]) \
    .agg(mean(col("received_bytes").cast(FloatType())).alias("avg_received_bytes"),
         mean(col("sent_bytes").cast(FloatType())).alias("avg_sent_bytes"))

#
# _model_input_4\
#     .describe()\
#     .show(10)
