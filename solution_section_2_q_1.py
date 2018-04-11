"""
Author: Ravi Kumar Yadav
Date: 10/04/2018
TOpic: Section 2 question 1 Solution
"""
###########################################################################################
# SECTION 2 : Additional questions for Machine Learning Engineer (MLE) candidates:
# Q1: Predict the expected load (requests/second) in the next minute
###########################################################################################

###########################################################################################
# COFIGURATION
###########################################################################################

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import sys
from pyspark.ml.feature import VectorAssembler, StandardScaler
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

REPARTITION_FACTOR = int(sc._jsc.sc().getExecutorMemoryStatus().size()) * 10
# print(REPARTITION_FACTOR)

# UTILS FUNCTIONS
_minutesLambda = lambda i: i * 60
###########################################################################################3

###########################################################################################3
# DATA INGESTION
###########################################################################################3
raw_data = spark.read.option("delimiter", " ").csv("C://Users/Ravi/PycharmProjects/WeblogChallenge/data")
# .sample(False, 0.00001, 42)

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
    .replace(["\"GET"], ["GET"], "request_type")

# _pre_proc.select(min(col("unix_tmpstmp")),max(col("unix_tmpstmp"))).show()
# _pre_proc.select(col("hour"), col("minute")).distinct().orderBy("hour").show(110)

#############################################################################
# -- FEATURE SET 1
#############################################################################
print("Creating feature set 1...")
_model_input_1 = _pre_proc \
    .withColumn("row_count", count(col("date")).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize)).cast(
    FloatType())) \
    .withColumn("avg_request_processing_time", mean(col("request_processing_time_clean").cast(FloatType())).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize))) \
    .withColumn("avg_backend_processing_time", mean(col("backend_processing_time_clean").cast(FloatType())).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize))) \
    .withColumn("avg_response_processing_time", mean(col("response_processing_time_clean").cast(FloatType())).over(
    Window.partitionBy(["date", "hour", "minute"]).orderBy("date").rangeBetween(-sys.maxsize, sys.maxsize))) \
    .groupBy(["date", "hour", "minute", "row_count", "avg_request_processing_time", "avg_backend_processing_time",
              "avg_response_processing_time"]) \
    .pivot("request_type").sum("dummy_count") \
    .repartition(REPARTITION_FACTOR) \
    .na.fill(0.0) \
    .withColumn("time", to_timestamp(concat(col("date"), lit(" "), col("hour"), lit(":"), col("minute")),
                                     format="yyyy-MM-dd HH:mm")) \
    .withColumn("load", col("row_count") / lit(60.0)) \
    .drop("row_count")

_initial_column_set_1 = set(_model_input_1.columns) - set(
    ["load", "date", "hour", "minute"])  # these are the redundant columns

# print("_model_input_1 SCHEMA")
# _model_input_1.printSchema()

_model_input_1.select(mean(col("load"))).show()
####################################################################################

for col_name in list(set(_model_input_1.columns) - set(["date", "hour", "minute", "time"])):
    # time_lag_in_mins = 15
    # while time_lag_in_mins <= 60:
    for time_lag_in_mins in [1, 15, 60, 120]:
        _model_input_1 = _model_input_1 \
            .withColumn(col_name + "_cum_" + str(time_lag_in_mins) + "_minutes",
                        sum(col_name)
                        .over(Window.partitionBy("date")
                              .orderBy(col("time").cast("timestamp").cast("long"))
                              .rangeBetween(- _minutesLambda(time_lag_in_mins), -1)
                              )
                        ) \
            .na.fill(0.0)

        # time_lag_in_mins += 15

# print("_model_input_1 SCHEMA")
# _model_input_1.printSchema()
# _model_input_1.show(10)
##########################################################################

_final_column_set_1 = set(_model_input_1.columns) - _initial_column_set_1

_model_input_1_feature_set_to_assembler = _model_input_1 \
    .drop("time") \
    .select(list(_final_column_set_1))

feature_columns_1 = list(set(_model_input_1_feature_set_to_assembler.columns) - set(["date", "hour", "minute", "load"]))

##############################################################################

assembler_1 = VectorAssembler(inputCols=feature_columns_1,
                              outputCol="feature_1")

_model_input_1_feature_set = assembler_1.transform(_model_input_1_feature_set_to_assembler) \
    .select("date", "hour", "minute", "load", "feature_1")

# _model_input_1_feature_set.show(5)
#############################################################################
# -- FEATURE SET 2
#############################################################################
print("Creating feature set 2...")

_model_input_2 = _pre_proc \
    .groupBy(["date", "hour", "minute"]) \
    .pivot("elb_status_code").sum("dummy_count") \
    .repartition(REPARTITION_FACTOR) \
    .na.fill(0.0) \
    .withColumn("time", to_timestamp(concat(col("date"), lit(" "), col("hour"), lit(":"), col("minute")),
                                     format="yyyy-MM-dd HH:mm"))

_initial_column_set_2 = set(_model_input_2.columns) - set(["date", "hour", "minute"])

# print("_model_input_2 SCHEMA")
# _model_input_2.printSchema()
###############################################################################

for col_name in list(set(_model_input_2.columns) - set(["date", "hour", "minute", "time"])):

    for time_lag_in_mins in [1, 15]:
        _model_input_2 = _model_input_2 \
            .withColumn(col_name + "_cum_" + str(time_lag_in_mins) + "_minutes",
                        sum(col_name)
                        .over(Window.partitionBy("date")
                              .orderBy(col("time").cast("timestamp").cast("long"))
                              .rangeBetween(- _minutesLambda(time_lag_in_mins), -1)
                              )
                        ) \
            .na.fill(0.0)

###############################################################################

_final_column_set_2 = set(_model_input_2.columns) - _initial_column_set_2

_model_input_2_feature_set_to_assembler = _model_input_2 \
    .drop("time") \
    .select(list(_final_column_set_2))

feature_columns_2 = list(set(_model_input_2_feature_set_to_assembler.columns) - set(["date", "hour", "minute"]))

#############################################################################33

assembler_2 = VectorAssembler(inputCols=feature_columns_2,
                              outputCol="feature_2")

_model_input_2_feature_set = assembler_2.transform(_model_input_2_feature_set_to_assembler) \
    .select("date", "hour", "minute", "feature_2")

# _model_input_2_feature_set.show(5)
#############################################################################
# -- FEATURE SET 3
#############################################################################
print("Creating feature set 3...")
_model_input_3 = _pre_proc \
    .groupBy(["date", "hour", "minute"]) \
    .pivot("backend_status_code").sum("dummy_count") \
    .repartition(REPARTITION_FACTOR) \
    .na.fill(0.0) \
    .withColumn("time", to_timestamp(concat(col("date"), lit(" "), col("hour"), lit(":"), col("minute")),
                                     format="yyyy-MM-dd HH:mm"))

_initial_column_set_3 = set(_model_input_3.columns) - set(["date", "hour", "minute"])

# print("_model_input_3 SCHEMA")
# _model_input_3.printSchema()

##############################################################################

for col_name in list(set(_model_input_3.columns) - set(["date", "hour", "minute", "time"])):

    for time_lag_in_mins in [1, 15]:
        _model_input_3 = _model_input_3 \
            .withColumn(col_name + "_cum_" + str(time_lag_in_mins) + "_minutes",
                        sum(col_name)
                        .over(Window.partitionBy("date")
                              .orderBy(col("time").cast("timestamp").cast("long"))
                              .rangeBetween(- _minutesLambda(time_lag_in_mins), -1)
                              )
                        ) \
            .na.fill(0.0)

# print("_model_input_3 SCHEMA")
# _model_input_3.printSchema()
#############################################################################

_final_column_set_3 = set(_model_input_3.columns) - _initial_column_set_3

_model_input_3_feature_set_to_assembler = _model_input_3 \
    .drop("time") \
    .select(list(_final_column_set_3))

feature_columns_3 = list(set(_model_input_3_feature_set_to_assembler.columns) - set(["date", "hour", "minute"]))
##############################################################################

assembler_3 = VectorAssembler(inputCols=feature_columns_3,
                              outputCol="feature_3")

_model_input_3_feature_set = assembler_3.transform(_model_input_3_feature_set_to_assembler) \
    .select("date", "hour", "minute", "feature_3")

# _model_input_3_feature_set.show(5)
#############################################################################
# -- FEATURE SET 4
#############################################################################
print("Creating feature set 4...")
_model_input_4 = _pre_proc \
    .select(col("date"), col("hour"), col("minute"), col("received_bytes"), col("sent_bytes")) \
    .groupBy(["date", "hour", "minute"]) \
    .agg(sum(col("received_bytes").cast(FloatType())).alias("avg_received_bytes"),
         sum(col("sent_bytes").cast(FloatType())).alias("avg_sent_bytes")) \
    .repartition(REPARTITION_FACTOR) \
    .withColumn("time", to_timestamp(concat(col("date"), lit(" "), col("hour"), lit(":"), col("minute")),
                                     format="yyyy-MM-dd HH:mm"))

_initial_column_set_4 = set(_model_input_4.columns) - set(["date", "hour", "minute"])

# print("_model_input_4 SCHEMA")
# _model_input_4.printSchema()
#################################################################################

for col_name in list(set(_model_input_4.columns) - set(["date", "hour", "minute", "time"])):

    for time_lag_in_mins in [1, 15]:
        _model_input_4 = _model_input_4 \
            .withColumn(col_name + "_cum_" + str(time_lag_in_mins) + "_minutes",
                        sum(col_name)
                        .over(Window.partitionBy("date")
                              .orderBy(col("time").cast("timestamp").cast("long"))
                              .rangeBetween(- _minutesLambda(time_lag_in_mins), -1)
                              )
                        ) \
            .na.fill(0.0)

# print("_model_input_4 SCHEMA")
# _model_input_4.printSchema()
#############################################################################

_final_column_set_4 = set(_model_input_4.columns) - _initial_column_set_4

_model_input_4_feature_set_to_assembler = _model_input_4 \
    .drop("time") \
    .select(list(_final_column_set_4))

feature_columns_4 = list(set(_model_input_4_feature_set_to_assembler.columns) - set(["date", "hour", "minute"]))

#############################################################################33

assembler_4 = VectorAssembler(inputCols=feature_columns_4,
                              outputCol="feature_4")

_model_input_4_feature_set = assembler_4.transform(_model_input_4_feature_set_to_assembler) \
    .select("date", "hour", "minute", "feature_4")

# _model_input_4_feature_set.show(5)

##################################################################################
# -- FINAL MODEL INPUT DATA COMBINING FEATURE SET 1,2,3,4
##################################################################################
print("Combining feature sets...\n")
_complete_model_input = _model_input_1_feature_set \
    .join(_model_input_2_feature_set, ["date", "hour", "minute"]) \
    .join(_model_input_3_feature_set, ["date", "hour", "minute"]) \
    .join(_model_input_4_feature_set, ["date", "hour", "minute"]) \
    .repartition(REPARTITION_FACTOR)

# _complete_model_input.show(5)

_complete_feature_list = list(set(_complete_model_input.columns) - set(["date", "hour", "minute", "load"]))

assembler = VectorAssembler(inputCols=_complete_feature_list,
                            outputCol="feature")

_model_input_all_feature = assembler.transform(_complete_model_input) \
    .select("date", "hour", "minute", "load", "feature")

# print("_model_input_all_feature SCHEMA")
# _model_input_all_feature.printSchema()
# _model_input_all_feature.show(2)

########################################################################################

scaler = StandardScaler(inputCol="feature", outputCol="scaledFeatures",
                        withStd=True, withMean=True)

scalerModel = scaler.fit(_model_input_all_feature)

_model_input_all_feature_scaled = scalerModel.transform(_model_input_all_feature)

# _model_input_all_feature_normalized.show()
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
    .setLabelCol("load") \
    .setFeaturesCol("scaledFeatures")

gbtModel = gbt.fit(trainingData)
_test_pred = gbtModel.transform(testData).select("scaledFeatures", "load", "prediction")

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
