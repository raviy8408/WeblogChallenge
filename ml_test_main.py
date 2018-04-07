from pyspark.sql import SparkSession
from pyspark.mllib.regression import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import sys
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GBTRegressor

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("test") \
    .getOrCreate()

sc = spark.sparkContext

sc.setLogLevel("ERROR")
sc.setCheckpointDir('C://Users/Ravi/PycharmProjects/WeblogChallenge/checkpoint/')

# print(sc)

REPARTITION_FACTOR = int(sc._jsc.sc().getExecutorMemoryStatus().size()) * 10
print(REPARTITION_FACTOR)

# UTILS FUNCTIONS
_minutesLambda = lambda i: i * 60

raw_data = spark.read.option("delimiter", " ").csv("C://Users/Ravi/PycharmProjects/WeblogChallenge/data")
# .sample(False, 0.0001, 42)

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

# print(raw_data_w_cols_clean.select(col("user_agent")).distinct().show())


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


##############################################################################

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
    .withColumnRenamed("row_count", "load")

_initial_column_set_1 = set(_model_input_1.columns) - set(
    ["load", "date", "hour", "minute"])  # these are the redundant columns

# print("_model_input_1 SCHEMA")
# _model_input_1.printSchema()

####################################

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

_final_column_set_1 = set(_model_input_1.columns) - _initial_column_set_1

_model_input_1_feature_set_to_assembler = _model_input_1 \
    .drop("time") \
    .select(list(_final_column_set_1))

feature_columns_1 = list(set(_model_input_1_feature_set_to_assembler.columns) - set(["date", "hour", "minute", "load"]))

###############################################################################

##############################################################################
# VECTOR ASSEMBLER 1

assembler_1 = VectorAssembler(inputCols=feature_columns_1,
                              outputCol="feature_1")

_model_input_1_feature_set = assembler_1.transform(_model_input_1_feature_set_to_assembler) \
    .select("date", "hour", "minute", "load", "feature_1")

# _model_input_1_feature_set.show(5)

# print("_model_input_1_feature_set SCHEMA")
# _model_input_1_feature_set.printSchema()
# _model_input_1_feature_set.select(col("feature_1")).show(10)


## TODO: Needs to be deleted
# lr = LinearRegression()\
#     .setLabelCol("load")\
#     .setFeaturesCol("feature_1")\
#     .setMaxIter(10)\
#     .setRegParam(1.0)\
#     .setElasticNetParam(1.0)
#
# lrModel = lr.fit(_model_input_1_feature_set)
# lrModel.transform(_model_input_1_feature_set).select("feature_1", "load", "prediction").show()


# gbt = GBTRegressor(maxIter=5, maxDepth=2, seed=42)\
#     .setLabelCol("load")\
#     .setFeaturesCol("feature_1")
#
# gbtModel = gbt.fit(_model_input_1_feature_set)
# _train_pred = gbtModel.transform(_model_input_1_feature_set).select("feature_1", "load", "prediction")
# # _train_pred.show()
#
# trainMSE = _train_pred.rdd.map(lambda lp: (lp[1] - lp[2]) * (lp[1] - lp[2])).sum() /\
#     float(_train_pred.count())
# print('Test Mean Squared Error = ' + str(trainMSE))

##############################################################################

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

    for time_lag_in_mins in [1, 15, 60, 120]:
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

# _model_input_1.show(10)

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
##############################################################################

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

#################################3

for col_name in list(set(_model_input_3.columns) - set(["date", "hour", "minute", "time"])):

    for time_lag_in_mins in [1, 15, 60, 120]:
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
# _model_input_1.show(10)

_final_column_set_3 = set(_model_input_3.columns) - _initial_column_set_3

_model_input_3_feature_set_to_assembler = _model_input_3 \
    .drop("time") \
    .select(list(_final_column_set_3))

feature_columns_3 = list(set(_model_input_3_feature_set_to_assembler.columns) - set(["date", "hour", "minute"]))

#############################################################################33

assembler_3 = VectorAssembler(inputCols=feature_columns_3,
                              outputCol="feature_3")

_model_input_3_feature_set = assembler_3.transform(_model_input_3_feature_set_to_assembler) \
    .select("date", "hour", "minute", "feature_3")

# _model_input_3_feature_set.show(5)

#############################################################################


_model_input_4 = _pre_proc \
    .select(col("date"), col("hour"), col("minute"), col("received_bytes"), col("sent_bytes")) \
    .groupBy(["date", "hour", "minute"]) \
    .agg(mean(col("received_bytes").cast(FloatType())).alias("avg_received_bytes"),
         mean(col("sent_bytes").cast(FloatType())).alias("avg_sent_bytes")) \
    .repartition(REPARTITION_FACTOR) \
    .withColumn("time", to_timestamp(concat(col("date"), lit(" "), col("hour"), lit(":"), col("minute")),
                                     format="yyyy-MM-dd HH:mm"))

_initial_column_set_4 = set(_model_input_4.columns) - set(["date", "hour", "minute"])

# print("_model_input_3 SCHEMA")
# _model_input_3.printSchema()

#################################3

for col_name in list(set(_model_input_4.columns) - set(["date", "hour", "minute", "time"])):

    for time_lag_in_mins in [1, 15, 60, 120]:
        _model_input_4 = _model_input_4 \
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
# _model_input_1.show(10)

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

print("_model_input_all_feature SCHEMA")
# _model_input_all_feature.printSchema()

_model_input_all_feature.show(2)

# gbt = GBTRegressor(maxIter=10, maxDepth=4, seed=42, maxMemoryInMB=1024)\
#     .setLabelCol("load")\
#     .setFeaturesCol("feature")
#
# lr = LinearRegression()\
#     .setLabelCol("load")\
#     .setFeaturesCol("feature")\
#     .setMaxIter(10)\
#     .setRegParam(1.0)\
#     .setElasticNetParam(1.0)
#
# lrModel = lr.fit(_model_input_all_feature)
# _train_pred = lrModel.transform(_model_input_all_feature).select("feature", "load", "prediction")

# gbtModel = gbt.fit(_model_input_all_feature)
# _train_pred = gbtModel.transform(_model_input_all_feature).select("feature", "load", "prediction")

# print("_train_pred SCHEMA")
# _train_pred.printSchema()
# _train_pred.catch()
# _train_pred.show(10)

# trainMSE = _train_pred.rdd.map(lambda lp: (lp[1] - lp[2]) * (lp[1] - lp[2])).sum() /\
#            float(_train_pred.count())
# print('Test Mean Squared Error = ' + str(trainMSE))
