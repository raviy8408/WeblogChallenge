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

# raw_data_w_cols.show()
raw_data_w_cols \
    .replace(["-1"], []) \
    .select(col("request_processing_time").cast(FloatType()),
            col("backend_processing_time").cast(FloatType()),
            col("response_processing_time").cast(FloatType())
            ) \
    .describe() \
    .show()

_requests = raw_data_w_cols \
    .select(col("request")) \
    .withColumn("temp", split(col("request"), " ")) \
    .select(col("temp").getItem(0).alias("request_type"),
            col("temp").getItem(1).alias("request_address"),
            col("temp").getItem(2).alias("http_version"))

_website = _requests \
    .select(col("request_address")) \
    .withColumn("temp", split(col("request_address"), "/").getItem(2)) \
    .withColumn("website", split(col("temp"), ":").getItem(0)) \
    .select(col("website")) \
    .distinct()

# raw_data_w_cols.show(10)
# print(_requests.take(1))

# _requests.show(10)



_timestamp = raw_data_w_cols \
    .withColumn("IP", split(col("client"), ":").getItem(0)) \
    .withColumn("request_split", split(col("request"), " ")) \
    .withColumn("request_type", col("request_split").getItem(0)) \
    .withColumn("URL",
                lower(split(split(col("request_split").getItem(1), "/").getItem(2), ":").getItem(0).cast(StringType()))) \
    .withColumn("http_version", col("request_split").getItem(2)) \
    .withColumn("temp_tmpstmp", to_utc_timestamp(col("timestamp"), "ISO 8601")) \
    .withColumn("date", to_date(col("temp_tmpstmp"))) \
    .withColumn("hour", hour(col("temp_tmpstmp"))) \
    .withColumn("minute", minute(col("temp_tmpstmp"))) \
    .withColumn("dummy_count", lit(1.0)) \
    .replace(["\"GET"], ["GET"], "request_type") \
    .withColumn("row_count", count(col("date")).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize))) \
    .withColumn("avg_request_processing_time", mean(col("request_processing_time").cast(FloatType())).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize))) \
    .withColumn("avg_backend_processing_time", mean(col("backend_processing_time").cast(FloatType())).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize))) \
    .withColumn("avg_response_processing_time", mean(col("response_processing_time").cast(FloatType())).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize))) \
    .groupBy(["date", "hour", "minute", "row_count", "avg_request_processing_time", "avg_backend_processing_time",
              "avg_response_processing_time"]) \
    .pivot("request_type").sum("dummy_count") \
    .na.fill(0.0)
#
# _timestamp.show(15)
# _timestamp.describe().show()
# _timestamp.select(col("request_type_clean")).distinct().show()
# print(_timestamp.take(1))


_client = raw_data_w_cols \
    .select(col("client"),
            split(col("client"), ":").getItem(0).alias("IP"),
            col("timestamp"),
            col("backend"),
            col("request"),
            col("user_agent")) \
    .withColumn("temp_tmpstmp", to_utc_timestamp(col("timestamp"), "ISO 8601")) \
    .drop(col("timestamp")) \
    .orderBy(["IP", "temp_tmpstmp"])

# _client.show(50)

_sessionized = raw_data_w_cols \
    .withColumn("IP", split(col("client"), ":").getItem(0)) \
    .withColumn("request_split", split(col("request"), " ")) \
    .withColumn("request_type", col("request_split").getItem(0)) \
    .withColumn("URL",
                lower(split(split(col("request_split").getItem(1), "/").getItem(2), ":").getItem(0).cast(StringType()))) \
    .withColumn("http_version", col("request_split").getItem(2)) \
    .withColumn("temp_tmpstmp", to_utc_timestamp(col("timestamp"), "ISO 8601")) \
    .withColumn("lagged_tmpstmp", lag(col("temp_tmpstmp"), 1).over(Window.partitionBy("IP").orderBy("temp_tmpstmp"))) \
    .withColumn("new_session",
                coalesce(unix_timestamp(col("temp_tmpstmp")) - unix_timestamp(col("lagged_tmpstmp")), lit(0.0)).cast(
                    FloatType()) > 1800.0) \
    .withColumn("sessionized_temp", sum(when(col("new_session") == False, 0.0).otherwise(1.0)).over(
    Window.partitionBy("IP").orderBy("temp_tmpstmp"))) \
    .withColumn("sessionized", concat_ws("_", col("IP").cast(StringType()), lit("Session"),
                                         col("sessionized_temp").cast(StringType()))) \
    .orderBy(["IP", "temp_tmpstmp"])

# .groupBy(["IP"]) \
#     .agg(max(col("successive_time_diff")).alias("max_diff"),
#          min(col("successive_time_diff")).alias("min_diff"))

_session_time = _sessionized \
    .repartition(REPARTITION_FACTOR) \
    .groupBy(["IP", "sessionized"]) \
    .agg(max(col("temp_tmpstmp")).alias("session_end_time"),
         min(col("temp_tmpstmp")).alias("session_start_time")) \
    .withColumn("session_time", (unix_timestamp("session_end_time") - unix_timestamp("session_start_time")) / 60.0)

_URL_visit_count = _sessionized \
    .repartition(REPARTITION_FACTOR) \
    .groupBy(["sessionized"]) \
    .agg(countDistinct("URL").alias("URL_count"))

# _session_time.show(50)

# _session_time.groupBy(["IP"]).agg(mean("session_time").alias("avg_time")).orderBy(["avg_time"], ascending=[0]).show()
# _session_time.describe(['session_time']).show()
# _URL_visit_count.show(10)

# print(raw_data_w_cols.select("elb").distinct().count())
