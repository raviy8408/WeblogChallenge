"""
Author: Ravi Kumar Yadav
Date: 10/04/2018
Topic: Section 2 question 3 Solution
"""
###########################################################################################
# SECTION 2 : Additional questions for Machine Learning Engineer (MLE) candidates:
# Q1: Predict the expected load (requests/second) in the next minute
###########################################################################################

###########################################################################################
# COFIGURATION
###########################################################################################

from pyspark.sql import SparkSession
from pyspark.mllib.regression import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import sys
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import Normalizer
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

###########################################################################################3

###########################################################################################3
# DATA INGESTION
###########################################################################################3

raw_data = spark.read.option("delimiter", " ").csv("C://Users/Ravi/PycharmProjects/WeblogChallenge/data")
# .sample(False, 0.0001, 42)

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

# print(raw_data_w_cols.select(col("elb_status_code")).distinct().show())

raw_data_w_cols_clean = raw_data_w_cols \
    .withColumn("request_processing_time_clean",
                when(col("request_processing_time") == "-1", None).otherwise(
                    col("request_processing_time").cast(FloatType()))) \
    .withColumn("backend_processing_time_clean",
                when(col("backend_processing_time") == "-1", None).otherwise(
                    col("backend_processing_time").cast(FloatType()))) \
    .withColumn("response_processing_time_clean",
                when(col("response_processing_time") == "-1", None).otherwise(
                    col("response_processing_time").cast(FloatType()))) \
    .withColumn("elb_status_code_clean", concat_ws("_", lit("elb_status"), col("elb_status_code"))) \
    .drop("elb_status_code") \
    .withColumnRenamed("elb_status_code_clean", "elb_status_code") \
    .withColumn("backend_status_code_clean", concat_ws("_", lit("backend_status"), col("backend_status_code"))) \
    .drop("backend_status_code") \
    .withColumnRenamed("backend_status_code_clean", "backend_status_code") \
    .drop("request_processing_time") \
    .drop("backend_processing_time") \
    .drop("response_processing_time")

# print(raw_data_w_cols_clean.select(col("user_agent")).distinct().show())

#############################################################################
# -- MODEL INPUT DATA PREP
#############################################################################
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
    .drop("request_split") \
    .repartition(REPARTITION_FACTOR)

# _pre_proc.printSchema()

# _pre_proc.select(min(col("unix_tmpstmp")),max(col("unix_tmpstmp"))).show()
# _pre_proc.select(col("hour"), col("minute")).distinct().orderBy("hour").show(110)

#############################################################################
# -- FEATURE SET 1
#############################################################################

print("Creating feature set 1...")
_model_input_1 = _pre_proc \
    .withColumn("session_start_time", min("unix_tmpstmp").over(
    Window.partitionBy("sessionized").orderBy("unix_tmpstmp").rangeBetween(Window.unboundedPreceding,
                                                                           Window.unboundedFollowing))) \
    .withColumn("session_end_time", max("unix_tmpstmp").over(
    Window.partitionBy("sessionized").orderBy("unix_tmpstmp").rangeBetween(Window.unboundedPreceding,
                                                                           Window.unboundedFollowing))) \
    .withColumn("session_time", (unix_timestamp("session_end_time") - unix_timestamp("session_start_time")) / 60.0) \
    .withColumn("session_received_bytes", sum("received_bytes").over(
    Window.partitionBy("sessionized").orderBy("unix_tmpstmp").rangeBetween(Window.unboundedPreceding,
                                                                           Window.unboundedFollowing))) \
    .withColumn("session_sent_bytes", sum("sent_bytes").over(
    Window.partitionBy("sessionized").orderBy("unix_tmpstmp").rangeBetween(Window.unboundedPreceding,
                                                                           Window.unboundedFollowing))) \
    .withColumn("session_request_processing_time", sum("request_processing_time_clean").over(
    Window.partitionBy("sessionized").orderBy("unix_tmpstmp").rangeBetween(Window.unboundedPreceding,
                                                                           Window.unboundedFollowing))) \
    .withColumn("session_backend_processing_time", sum("backend_processing_time_clean").over(
    Window.partitionBy("sessionized").orderBy("unix_tmpstmp").rangeBetween(Window.unboundedPreceding,
                                                                           Window.unboundedFollowing))) \
    .withColumn("session_response_processing_time", sum("response_processing_time_clean").over(
    Window.partitionBy("sessionized").orderBy("unix_tmpstmp").rangeBetween(Window.unboundedPreceding,
                                                                           Window.unboundedFollowing))) \
    .groupBy("IP") \
    .agg(countDistinct("URL").alias("unique_URL_visit"),
         countDistinct("sessionized").alias("session_count"),
         sum("session_time").alias("avg_session_time"),
         sum("session_received_bytes").alias("avg_session_received_bytes"),
         sum("session_sent_bytes").alias("avg_session_sent_bytes"),
         sum("session_request_processing_time").alias("avg_request_processing_time"),
         sum("session_backend_processing_time").alias("avg_backend_processing_time"),
         sum("session_response_processing_time").alias("avg_response_processing_time")
         ) \
    .na.fill(0.0)

# _model_input_1.describe().show()

feature_columns_1 = list(set(_model_input_1.columns) - set(["IP", "unique_URL_visit"]))

##############################################################################

assembler_1 = VectorAssembler(inputCols=feature_columns_1,
                              outputCol="feature_1")

_model_input_1_feature_set = assembler_1.transform(_model_input_1) \
    .select("IP", "unique_URL_visit", "feature_1")

# _model_input_1_feature_set.show(5)

#############################################################################
# -- FEATURE SET 2
#############################################################################
print("Creating feature set 2...")
_model_input_2 = _pre_proc \
    .groupBy("IP") \
    .pivot("elb_status_code").sum("dummy_count") \
    .repartition(REPARTITION_FACTOR) \
    .na.fill(0.0)

# _model_input_2.describe().show()

feature_columns_2 = list(set(_model_input_2.columns) - set(["IP"]))

##############################################################################

assembler_2 = VectorAssembler(inputCols=feature_columns_2,
                              outputCol="feature_2")

_model_input_2_feature_set = assembler_2.transform(_model_input_2) \
    .select("IP", "feature_2")

# _model_input_2_feature_set.show(5)

#############################################################################
# -- FEATURE SET 3
#############################################################################
print("Creating feature set 3...")
_model_input_3 = _pre_proc \
    .groupBy("IP") \
    .pivot("backend_status_code").sum("dummy_count") \
    .repartition(REPARTITION_FACTOR) \
    .na.fill(0.0)

# _model_input_3.describe().show()

feature_columns_3 = list(set(_model_input_3.columns) - set(["IP"]))

##############################################################################

assembler_3 = VectorAssembler(inputCols=feature_columns_3,
                              outputCol="feature_3")

_model_input_3_feature_set = assembler_3.transform(_model_input_3) \
    .select("IP", "feature_3")

# _model_input_3_feature_set.show(5)

#############################################################################
# -- FEATURE SET 4
#############################################################################
print("Creating feature set 4...")
_model_input_4 = _pre_proc \
    .groupBy("IP") \
    .pivot("request_type").sum("dummy_count") \
    .repartition(REPARTITION_FACTOR) \
    .na.fill(0.0)

# _model_input_3.describe().show()

feature_columns_4 = list(set(_model_input_4.columns) - set(["IP"]))

##############################################################################

assembler_4 = VectorAssembler(inputCols=feature_columns_4,
                              outputCol="feature_4")

_model_input_4_feature_set = assembler_4.transform(_model_input_4) \
    .select("IP", "feature_4")

# _model_input_3_feature_set.show(5)

##################################################################################
# -- FINAL MODEL INPUT DATA COMBINING FEATURE SET 1,2,3
##################################################################################
print("Combining feature sets...\n")
_complete_model_input = _model_input_1_feature_set \
    .join(_model_input_2_feature_set, ["IP"])

# .repartition(REPARTITION_FACTOR)

# _complete_model_input.show(5)

_complete_feature_list = list(set(_complete_model_input.columns) - set(["IP", "unique_URL_visit"]))

assembler = VectorAssembler(inputCols=_complete_feature_list,
                            outputCol="feature")

_model_input_all_feature = assembler.transform(_complete_model_input) \
    .select("IP", "unique_URL_visit", "feature") \
    .repartition(REPARTITION_FACTOR)

# print("_model_input_all_feature SCHEMA")
# _model_input_all_feature.printSchema()
# _model_input_all_feature.show(2)

########################################################################################
print("Scaling features...")
scaler = StandardScaler(inputCol="feature", outputCol="scaledFeatures",
                        withStd=True, withMean=True)

scalerModel = scaler.fit(_model_input_all_feature)

_model_input_all_feature_scaled = scalerModel.transform(_model_input_all_feature) \
    .drop("feature") \
    .repartition(REPARTITION_FACTOR)

# _model_input_all_feature_scaled.show()

########################################################################################

########################################################################################
# --MODEL BUILDING
########################################################################################
print("Model Training...\n")
splits = _model_input_all_feature_scaled.randomSplit([0.7, 0.3])
trainingData = splits[0]
testData = splits[1]

# trainingData.show(3)
#
# lr = LinearRegression()\
#     .setLabelCol("load")\
#     .setFeaturesCol("feature")\
#     .setMaxIter(10)\
#     .setRegParam(1.0)\
#     .setElasticNetParam(1.0)
#
# lrModel = lr.fit(trainingData)
# _test_pred = lrModel.transform(testData).select("feature", "load", "prediction")

gbt = GBTRegressor(maxIter=3, maxDepth=3, seed=42, maxMemoryInMB=2048) \
    .setLabelCol("unique_URL_visit") \
    .setFeaturesCol("scaledFeatures")

gbtModel = gbt.fit(trainingData)
_test_pred = gbtModel.transform(testData).select("scaledFeatures", "unique_URL_visit", "prediction")

# print("_train_pred SCHEMA")
# _test_pred.printSchema()
# _test_pred.catch()
# _test_pred.show(10)

testMSE = _test_pred.rdd.map(lambda lp: (lp[1] - lp[2]) * (lp[1] - lp[2])).sum() / \
          float(_test_pred.count())

print("\n####################################################################\n")
print('Test Root Mean Squared Error = ' + str(sqrt(testMSE)))
print("\n####################################################################\n")

###########################################################################
