"""
Author: Ravi Kumar Yadav
Date: 08/04/2018
Topic: Section 1 Solution
"""

###########################################################################################3
# COFIGURATION
###########################################################################################3
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("test") \
    .getOrCreate()

sc = spark.sparkContext

sc.setLogLevel("ERROR")

REPARTITION_FACTOR = int(sc._jsc.sc().getExecutorMemoryStatus().size()) * 10
# print(REPARTITION_FACTOR)
###########################################################################################3

###########################################################################################3
# DATA INGESTION
###########################################################################################3
raw_data = spark.read.option("delimiter", " ").csv("C://Users/Ravi/PycharmProjects/WeblogChallenge/data") \
    .sample(False, 0.01, 42)

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
##################################################################################

###########################################################################################
# SECTION 1 : Processing & Analytical goals
###########################################################################################

###############################################################################
# Q1: Sessionize the web log by IP. Sessionize = aggregrate all page hits by
#       visitor/IP during a fixed time window.
# Solution: Per IP a session is defined as time gap between two consecutive
#           requests should be less than 30 min. The column "sessionize"
#           is the final column indicating the session for all the requests.
###############################################################################
_sessionized = raw_data_w_cols \
    .withColumn("IP", split(col("client"), ":").getItem(0)) \
    .withColumn("request_split", split(col("request"), " ")) \
    .withColumn("request_type", col("request_split").getItem(0)) \
    .withColumn("URL",
                lower(split(split(col("request_split").getItem(1), "/").getItem(2), ":").getItem(0).cast(StringType()))) \
    .withColumn("http_version", col("request_split").getItem(2)) \
    .withColumn("unix_tmpstmp", to_utc_timestamp(col("timestamp"), "ISO 8601")) \
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
    .repartition(REPARTITION_FACTOR)

print("SECTION 1: \n")
print("Solution 1:")
_sessionized.show(10)
print("##################################################\n")

###############################################################################
# Q2: Determine the average session time
# Solution: Calculate session time from session start and end time. Print
#           variable summary.
###############################################################################

_session_time = _sessionized \
    .groupBy(["IP", "sessionized"]) \
    .agg(max(col("unix_tmpstmp")).alias("session_end_time"),
         min(col("unix_tmpstmp")).alias("session_start_time")) \
    .withColumn("session_time", (unix_timestamp("session_end_time") - unix_timestamp("session_start_time")) / 60.0)

print("Solution 2: Average session time")
_session_time.describe(['session_time']).show()
print("##################################################\n")

###############################################################################
# Q3: Determine unique URL visits per session. To clarify, count a hit to a
#       unique URL only once per session.
# Solution: Count distinct URL per session visit
###############################################################################

_URL_visit_count = _sessionized \
    .groupBy(["sessionized"]) \
    .agg(countDistinct("URL").alias("URL_count"))

print("Solution 3:")
_URL_visit_count.show(10)
print("##################################################\n")

###############################################################################
# Q4: Find the most engaged users, ie the IPs with the longest session times
# Solution: Calculate avg session time for all the IPs. Table is displayed in
#           descending order of avg session time
###############################################################################

_user_avg_session_time = _session_time.groupBy(["IP"]) \
    .agg(mean("session_time").alias("avg_session_time")) \
    .orderBy(["avg_session_time"], ascending=[0])

print("Solution 4:")
_user_avg_session_time.show(10)
print("##################################################\n")

###############################################################################
