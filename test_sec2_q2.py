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
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GBTRegressor
from math import sqrt

# from pyspark.sql import HiveContext

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("test") \
    .enableHiveSupport() \
    .getOrCreate()

sc = spark.sparkContext
# sqlContext = HiveContext(sc)

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

#############################################################################
# -- FEATURE SET 1
#############################################################################
print("Creating feature set 1...")

_model_input_1 = _pre_proc \
    .withColumn("IP_URL_visit", size(collect_set("URL").over(
    Window.partitionBy("IP").orderBy("unix_tmpstmp").rangeBetween(Window.unboundedPreceding,
                                                                  Window.unboundedFollowing)))) \
    .groupBy(["sessionized", "IP_URL_visit"]) \
    .agg(max(col("unix_tmpstmp")).alias("session_end_time"),
         min(col("unix_tmpstmp")).alias("session_start_time"),
         hour(min(col("unix_tmpstmp"))).alias("session_start_hour")) \
    .withColumn("session_time", (unix_timestamp("session_end_time") - unix_timestamp("session_start_time")) / 60.0) \
    .select("sessionized", "session_start_hour", "session_time", "IP_URL_visit")

# print("_model_input_1 SCHEMA:")
# _model_input_1.printSchema()

# _model_input_1.select(mean(col("session_time"))).show(2)
# print(_model_input_1.select(col("session_start_hour")).distinct().count())

########################################################################

stringIndexer = StringIndexer(inputCol="session_start_hour", outputCol="session_start_hour_indexed")
indexer = stringIndexer.fit(_model_input_1)
_model_input_1_st_indexed = indexer.transform(_model_input_1)
encoder = OneHotEncoder(inputCol="session_start_hour_indexed", outputCol="session_start_hour_encoded")
_model_input_1_encoded = encoder.transform(_model_input_1_st_indexed) \
    .drop("session_start_hour") \
    .drop("session_start_hour_indexed") \
 \
# _model_input_1_encoded.show(2)

_feature_column_set_1 = ["IP_URL_visit"]
#########################################################################

assembler_1 = VectorAssembler(inputCols=_feature_column_set_1,
                              outputCol="feature_1")

_model_input_1_feature_set = assembler_1.transform(_model_input_1_encoded) \
    .select("sessionized", "session_time", "session_start_hour_encoded", "feature_1")

_model_input_1_feature_set.select(mean(col("session_time"))).show()

########################################################################

#############################################################################
# -- FEATURE SET 3
#############################################################################
print("Creating feature set 2...")

_model_input_2_temp_1 = _pre_proc \
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

_initial_column_set_2_temp_1 = set(_model_input_2_temp_1.columns) - set(
    ["date", "hour", "minute"])  # these are the redundant columns

# print("_model_input_2_temp_1 SCHEMA")
# _model_input_2_temp_1.printSchema()

####################################################################################

for col_name in list(set(_model_input_2_temp_1.columns) - set(["date", "hour", "minute", "time"])):
    # time_lag_in_mins = 15
    # while time_lag_in_mins <= 60:
    for time_lag_in_mins in [15]:
        _model_input_2_temp_1 = _model_input_2_temp_1 \
            .withColumn(col_name + "_cum_" + str(time_lag_in_mins) + "_minutes",
                        sum(col_name)
                        .over(Window.partitionBy("date")
                              .orderBy(col("time").cast("timestamp").cast("long"))
                              .rangeBetween(- _minutesLambda(time_lag_in_mins), -1)
                              )
                        ) \
            .na.fill(0.0)

        # time_lag_in_mins += 15

# print("_model_input_2_temp_1 SCHEMA")
# _model_input_2_temp_1.printSchema()
# _model_input_1.show(10)
##########################################################################

_final_column_set_2_temp_1 = set(_model_input_2_temp_1.columns) - _initial_column_set_2_temp_1

_model_input_2_temp_1_feature_set_to_assembler = _model_input_2_temp_1 \
    .select(list(_final_column_set_2_temp_1)) \
    .drop("time")

feature_columns_2_temp_1 = list(
    set(_model_input_2_temp_1_feature_set_to_assembler.columns) - set(["date", "hour", "minute"]))

# print(feature_columns_2_temp_1)

###########################################################################

assembler_2_temp_1 = VectorAssembler(inputCols=feature_columns_2_temp_1,
                                     outputCol="feature_2")

_model_input_2_temp_1_feature_set = assembler_2_temp_1.transform(_model_input_2_temp_1_feature_set_to_assembler) \
    .withColumn("date_hour_min", concat_ws("_", concat_ws("_", col("date"), col("hour")), col("minute"))) \
    .select("date_hour_min", "feature_2") \
 \
# _model_input_2_temp_1_feature_set.printSchema()

#########################################################################

_session_start_date_hour_min = _pre_proc \
    .groupBy("sessionized") \
    .agg(min("unix_tmpstmp").alias("session_start_time")) \
    .withColumn("date", to_date(col("session_start_time"))) \
    .withColumn("hour", hour(col("session_start_time"))) \
    .withColumn("minute", minute(col("session_start_time"))) \
    .withColumn("date_hour_min", concat_ws("_", concat_ws("_", col("date"), col("hour")), col("minute"))) \
    .select("sessionized", "date_hour_min")

# print("_model_input_2_temp_2 SCHEMA:")
# _model_input_2_temp_2.printSchema()

_model_input_2_feature_set = _session_start_date_hour_min \
    .join(_model_input_2_temp_1_feature_set, "date_hour_min", how="left") \
    .drop("date_hour_min")

# print("_model_input_2 SCHEMA:")
# _model_input_2.printSchema()
# _model_input_2.show(10)

############################################################################

#############################################################################
# -- FEATURE SET 3
#############################################################################
print("Creating feature set 3...")

_model_input_3_temp_1 = _pre_proc \
    .groupBy(["date", "hour", "minute"]) \
    .pivot("elb_status_code").sum("dummy_count") \
    .repartition(REPARTITION_FACTOR) \
    .na.fill(0.0) \
    .withColumn("time", to_timestamp(concat(col("date"), lit(" "), col("hour"), lit(":"), col("minute")),
                                     format="yyyy-MM-dd HH:mm"))

_initial_column_set_3_temp_1 = set(_model_input_3_temp_1.columns) - set(["date", "hour", "minute"])

# print("_model_input_3_temp_1 SCHEMA")
# _model_input_3_temp_1.printSchema()
###############################################################################

for col_name in list(set(_model_input_3_temp_1.columns) - set(["date", "hour", "minute", "time"])):

    for time_lag_in_mins in [15]:
        _model_input_3_temp_1 = _model_input_3_temp_1 \
            .withColumn(col_name + "_cum_" + str(time_lag_in_mins) + "_minutes",
                        sum(col_name)
                        .over(Window.partitionBy("date")
                              .orderBy(col("time").cast("timestamp").cast("long"))
                              .rangeBetween(- _minutesLambda(time_lag_in_mins), -1)
                              )
                        ) \
            .na.fill(0.0)

###############################################################################

_final_column_set_3_temp_1 = set(_model_input_3_temp_1.columns) - _initial_column_set_3_temp_1

_model_input_3_temp_1_feature_set_to_assembler = _model_input_3_temp_1 \
    .drop("time") \
    .select(list(_final_column_set_3_temp_1))

feature_columns_3_temp_1 = list(
    set(_model_input_3_temp_1_feature_set_to_assembler.columns) - set(["date", "hour", "minute"]))

#############################################################################33

assembler_3_temp_1 = VectorAssembler(inputCols=feature_columns_3_temp_1,
                                     outputCol="feature_3")

_model_input_3_temp_1_feature_set = assembler_3_temp_1.transform(_model_input_3_temp_1_feature_set_to_assembler) \
    .withColumn("date_hour_min", concat_ws("_", concat_ws("_", col("date"), col("hour")), col("minute"))) \
    .select("date_hour_min", "feature_3")

################################################################################

_model_input_3_feature_set = _session_start_date_hour_min \
    .join(_model_input_3_temp_1_feature_set, "date_hour_min", how="left") \
    .drop("date_hour_min")

################################################################################

#############################################################################
# -- FEATURE SET 4
#############################################################################


print("Creating feature set 4...")
_model_input_4_temp_1 = _pre_proc \
    .groupBy(["date", "hour", "minute"]) \
    .pivot("backend_status_code").sum("dummy_count") \
    .repartition(REPARTITION_FACTOR) \
    .na.fill(0.0) \
    .withColumn("time", to_timestamp(concat(col("date"), lit(" "), col("hour"), lit(":"), col("minute")),
                                     format="yyyy-MM-dd HH:mm"))

_initial_column_set_4_temp_1 = set(_model_input_4_temp_1.columns) - set(["date", "hour", "minute"])

# print("_model_input_4_temp_1 SCHEMA")
# _model_input_4_temp_1.printSchema()

#################################4_temp_1

for col_name in list(set(_model_input_4_temp_1.columns) - set(["date", "hour", "minute", "time"])):

    for time_lag_in_mins in [15]:
        _model_input_4_temp_1 = _model_input_4_temp_1 \
            .withColumn(col_name + "_cum_" + str(time_lag_in_mins) + "_minutes",
                        sum(col_name)
                        .over(Window.partitionBy("date")
                              .orderBy(col("time").cast("timestamp").cast("long"))
                              .rangeBetween(- _minutesLambda(time_lag_in_mins), -1)
                              )
                        ) \
            .na.fill(0.0)

# print("_model_input_4_temp_1 SCHEMA")
# _model_input_4_temp_1.printSchema()

#############################################################################

_final_column_set_4_temp_1 = set(_model_input_4_temp_1.columns) - _initial_column_set_4_temp_1

_model_input_4_temp_1_feature_set_to_assembler = _model_input_4_temp_1 \
    .drop("time") \
    .select(list(_final_column_set_4_temp_1))

feature_columns_4_temp_1 = list(
    set(_model_input_4_temp_1_feature_set_to_assembler.columns) - set(["date", "hour", "minute"]))

#############################################################################

assembler_4_temp_1 = VectorAssembler(inputCols=feature_columns_4_temp_1,
                                     outputCol="feature_4")

_model_input_4_temp_1_feature_set = assembler_4_temp_1.transform(_model_input_4_temp_1_feature_set_to_assembler) \
    .withColumn("date_hour_min", concat_ws("_", concat_ws("_", col("date"), col("hour")), col("minute"))) \
    .select("date_hour_min", "feature_4")

# _model_input_4_temp_1_feature_set.show(5)

################################################################################################

################################################################################

_model_input_4_feature_set = _session_start_date_hour_min \
    .join(_model_input_4_temp_1_feature_set, "date_hour_min", how="left") \
    .drop("date_hour_min")

################################################################################


#############################################################################
# -- FEATURE SET 5
#############################################################################
print("Creating feature set 5...")
_model_input_5_temp_1 = _pre_proc \
    .select(col("date"), col("hour"), col("minute"), col("received_bytes"), col("sent_bytes")) \
    .groupBy(["date", "hour", "minute"]) \
    .agg(mean(col("received_bytes").cast(FloatType())).alias("avg_received_bytes"),
         mean(col("sent_bytes").cast(FloatType())).alias("avg_sent_bytes")) \
    .repartition(REPARTITION_FACTOR) \
    .withColumn("time", to_timestamp(concat(col("date"), lit(" "), col("hour"), lit(":"), col("minute")),
                                     format="yyyy-MM-dd HH:mm"))

_initial_column_set_5_temp_1 = set(_model_input_5_temp_1.columns) - set(["date", "hour", "minute"])

# print("_model_input_5_temp_1 SCHEMA")
# _model_input_5_temp_1.printSchema()
#################################################################################

for col_name in list(set(_model_input_5_temp_1.columns) - set(["date", "hour", "minute", "time"])):

    for time_lag_in_mins in [15]:
        _model_input_5_temp_1 = _model_input_5_temp_1 \
            .withColumn(col_name + "_cum_" + str(time_lag_in_mins) + "_minutes",
                        sum(col_name)
                        .over(Window.partitionBy("date")
                              .orderBy(col("time").cast("timestamp").cast("long"))
                              .rangeBetween(- _minutesLambda(time_lag_in_mins), -1)
                              )
                        ) \
            .na.fill(0.0)

# print("_model_input_5_temp_1 SCHEMA")
# _model_input_5_temp_1.printSchema()
#############################################################################

_final_column_set_5_temp_1 = set(_model_input_5_temp_1.columns) - _initial_column_set_5_temp_1

_model_input_5_temp_1_feature_set_to_assembler = _model_input_5_temp_1 \
    .drop("time") \
    .select(list(_final_column_set_5_temp_1))

feature_columns_5_temp_1 = list(
    set(_model_input_5_temp_1_feature_set_to_assembler.columns) - set(["date", "hour", "minute"]))

#############################################################################33

assembler_5_temp_1 = VectorAssembler(inputCols=feature_columns_5_temp_1,
                                     outputCol="feature_5")

_model_input_5_temp_1_feature_set = assembler_5_temp_1.transform(_model_input_5_temp_1_feature_set_to_assembler) \
    .withColumn("date_hour_min", concat_ws("_", concat_ws("_", col("date"), col("hour")), col("minute"))) \
    .select("date_hour_min", "feature_5")

# _model_input_5_temp_1_feature_set.show(5)

################################################################################

_model_input_5_feature_set = _session_start_date_hour_min \
    .join(_model_input_5_temp_1_feature_set, "date_hour_min", how="left") \
    .drop("date_hour_min")

################################################################################

##################################################################################
# -- FINAL MODEL INPUT DATA COMBINING FEATURE SET 1,2,3,4,5
##################################################################################
print("Combining feature sets...\n")
_complete_model_input = _model_input_1_feature_set \
    .join(_model_input_2_feature_set, ["sessionized"]) \
    .join(_model_input_3_feature_set, ["sessionized"]) \
    .join(_model_input_4_feature_set, ["sessionized"]) \
    .join(_model_input_5_feature_set, ["sessionized"]) \
    .repartition(REPARTITION_FACTOR)

# _complete_model_input.show(5)

_complete_feature_list_continuous_var = list(
    set(_complete_model_input.columns) - set(["sessionized", "session_time", "session_start_hour_encoded"]))

assembler_continuous_var = VectorAssembler(inputCols=_complete_feature_list_continuous_var,
                                           outputCol="feature_continuous")

_model_input_all_feature = assembler_continuous_var.transform(_complete_model_input) \
    .select("sessionized", "session_time", "session_start_hour_encoded", "feature_continuous")

# print("_model_input_all_feature SCHEMA")
# _model_input_all_feature.printSchema()
# _model_input_all_feature.show(2)

########################################################################################
scaler = StandardScaler(inputCol="feature_continuous", outputCol="scaledFeatures",
                        withStd=True, withMean=True)

scalerModel = scaler.fit(_model_input_all_feature)

_model_input_continuous_feature_scaled = scalerModel.transform(_model_input_all_feature)

# _model_input_continuous_feature_scaled.printSchema()

_complete_feature_list = ["scaledFeatures", "session_start_hour_encoded"]

########################################################################################

assembler = VectorAssembler(inputCols=_complete_feature_list,
                            outputCol="feature")

_model_input_all_feature_vectorized = assembler.transform(_model_input_continuous_feature_scaled) \
    .select("sessionized", "session_time", "feature")

# _model_input_all_feature_vectorized.printSchema()

########################################################################################
# --MODEL BUILDING
########################################################################################
print("Model Training...\n")
splits = _model_input_all_feature_vectorized.randomSplit([0.7, 0.3])
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
    .setLabelCol("session_time") \
    .setFeaturesCol("feature")

gbtModel = gbt.fit(trainingData)
_test_pred = gbtModel.transform(testData).select("feature", "session_time", "prediction")

# print("_train_pred SCHEMA")
# _test_pred.printSchema()
# _test_pred.catch()
_test_pred.show(30)

# testMSE = _test_pred.rdd.map(lambda lp: (lp[1] - lp[2]) * (lp[1] - lp[2])).sum() / \
#           float(_test_pred.count())
#
# print("\n####################################################################\n")
# print('Test Root Mean Squared Error = ' + str(sqrt(testMSE)))
# print("\n####################################################################\n")

###########################################################################
