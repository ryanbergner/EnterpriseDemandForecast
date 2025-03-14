# COMMAND ----------
# MAGIC %md
# MAGIC # Forecasting Example w/ Neural Network

# COMMAND ----------
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import (
    count,
    lag,
    last_day,
    expr,
    explode,
    sequence,
    date_format,
    col,
    last,
    lead,
    when,
    months_between,
    month,
    year,
    avg,
    to_timestamp,
    to_date,
    mean,
    unix_timestamp
)
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType
)
from typing import Optional, List, Any, Dict
from math import sqrt as math_sqrt
import datetime
import mlflow

# Temporary
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline, PipelineModel

from src.preprocessing.preprocess import aggregate_sales_data, retrieve_sales_data
from src.feature_engineering.feature_engineering import add_features
from src.model_training.ml_models import (
    train_sparkML_models as train_sparkML_model,
    evaluate_sparkML_models as evaluate_SparkML_model
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Checking Mounted Directories

# COMMAND ----------
mounts = dbutils.fs.mounts()
display(mounts)

# COMMAND ----------
source_path = "dbfs:/mnt/m5"
date_column = "OrderDate"
product_id_column = "item_id"
quantity_column = "Quantity"
month_end_column = "MonthEndDate"

# COMMAND ----------
dbutils.fs.mounts()

# COMMAND ----------
files = dbutils.fs.ls("/mnt/m5")
display(files)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load & Transform M5 Dataset

# COMMAND ----------
m5_sdf = spark.read.csv("/mnt/m5/sales_train_validation.csv", header=True).limit(100000)

from pyspark.sql.functions import col, expr, explode, row_number

# Create a window specification partitioned by item_id
window_spec = Window.partitionBy("item_id").orderBy(expr("monotonically_increasing_id()"))
day_columns = [f"d_{i}" for i in range(1, 1901)]

# Unpivot the DataFrame
m5_sdf = (
    m5_sdf
    .select(
        col("item_id"),
        explode(expr(f"array({','.join(day_columns)})")).alias("Quantity")
    )
    .withColumn("day", (row_number().over(window_spec) - 1) % 1900 + 1)
    .withColumn("OrderDate", expr("date_add('2015-01-01', day - 1)"))
)

display(m5_sdf.tail(5))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Aggregate & Feature Engineering

# COMMAND ----------
df_agg = aggregate_sales_data(m5_sdf, date_column, product_id_column, quantity_column, month_end_column)
df_feat = add_features(df_agg, month_end_column, product_id_column, quantity_column)
display(df_feat.tail(5))

# COMMAND ----------
df_feat.write.mode("overwrite").option("header", "true").csv("/mnt/m5/data/df_feat.csv")

# COMMAND ----------
df_feat.write.mode("overwrite").saveAsTable("main.default.df_feat")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Neural Network Training with TensorFlow/Keras

# COMMAND ----------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

# Select relevant columns for the model
input_columns = [
    "Quantity",
    "month",
    "year",
    "months_since_last_order",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_5",
]
target_column = "lead_month_1"

# Convert Spark DataFrame to Pandas DataFrame for Keras
df_feat_pd = df_feat.select(*input_columns, target_column).toPandas()
df_feat_pd.dropna(inplace=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Simple Model Training

# COMMAND ----------
# Adjust input columns for a quick example
input_columns = ["Quantity", "month", "year", "months_since_last_order", "lag_1", "lag_11"]
target_column = "lead_month_1"

X = df_feat_pd[input_columns].values
y = df_feat_pd[target_column].values

X[0], y[0]  # Checking first example

# COMMAND ----------
# Define the neural network model (simplest example)
model = Sequential()
model.add(InputLayer(input_shape=(len(input_columns),)))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["R2Score", "RootMeanSquaredError"])

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Display the model summary
model.summary()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Slightly More Complex NN -> Multiple Layers & Dropout

# COMMAND ----------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# Normalize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the neural network model
model = Sequential()
model.add(InputLayer(input_shape=(len(input_columns),)))
model.add(Dense(8, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))

# Compile the model
model.compile(
    optimizer="adam", loss="mse", metrics=["R2Score", "RootMeanSquaredError"]
)

# Early stopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train the model with a validation split
model.fit(X_scaled, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

model.summary()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Hyperparameter Tuning with Keras Tuner

# COMMAND ----------
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from kerastuner.tuners import RandomSearch

# Normalize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def build_model(hp):
    model = Sequential()
    model.add(InputLayer(input_shape=(len(input_columns),)))

    # Tune the number of units in the first Dense layer
    model.add(Dense(
        units=hp.Int("units_1", min_value=8, max_value=128, step=8),
        activation="relu"
    ))
    model.add(Dropout(
        rate=hp.Float("dropout_1", min_value=0.0, max_value=0.5, step=0.1)
    ))

    # Tune the number of units in the second Dense layer
    model.add(Dense(
        units=hp.Int("units_2", min_value=8, max_value=128, step=8),
        activation="relu"
    ))
    model.add(Dropout(
        rate=hp.Float("dropout_2", min_value=0.0, max_value=0.5, step=0.1)
    ))

    model.add(Dense(1))

    # Tune the learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3)
        ),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.R2Score()]
    )
    return model

tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1
    # directory="my_dir",
    # project_name="hyperparam_tuning"
)

early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

tuner.search(X_scaled, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

best_hps = tuner.get_best_hyperparameters(num_trials=10)[0]
model = tuner.hypermodel.build(best_hps)

model.fit(X_scaled, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
model.summary()

# COMMAND ----------
# MAGIC %md
# MAGIC ### CNN Network with Hyperparameter Tuning

# COMMAND ----------
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from kerastuner.tuners import RandomSearch

# Normalize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the input data to be 3D for Conv1D => (samples, time steps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

def build_cnn_model(hp):
    model = Sequential()
    model.add(InputLayer(input_shape=(X_scaled.shape[1], 1)))

    # First Conv1D layer
    model.add(Conv1D(
        filters=hp.Int("filters_1", min_value=16, max_value=128, step=16),
        kernel_size=hp.Int("kernel_size_1", min_value=2, max_value=5, step=1),
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=hp.Float("dropout_1", min_value=0.0, max_value=0.5, step=0.1)))

    # Second Conv1D layer
    model.add(Conv1D(
        filters=hp.Int("filters_2", min_value=16, max_value=128, step=16),
        kernel_size=hp.Int("kernel_size_2", min_value=2, max_value=5, step=1),
        activation="relu"
    ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=hp.Float("dropout_2", min_value=0.0, max_value=0.5, step=0.1)))

    model.add(Flatten())
    model.add(Dense(1))

    # Tune learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3)
        ),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.R2Score()]
    )
    return model

cnn_tuner = RandomSearch(
    build_cnn_model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1
    # directory="my_dir",
    # project_name="hyperparam_tuning"
)

early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

cnn_tuner.search(X_scaled, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

best_hps = cnn_tuner.get_best_hyperparameters(num_trials=10)[0]
model = cnn_tuner.hypermodel.build(best_hps)

model.fit(X_scaled, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
model.summary()

# COMMAND ----------
# MAGIC %pip install keras-tuner

# COMMAND ----------
# MAGIC %md
# MAGIC ## XGBoost Model for Benchmark

# COMMAND ----------
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5
)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
display({"Mean Squared Error": mse, "R2 Score": r2})

# COMMAND ----------
# MAGIC %md
# MAGIC ## Correlation Matrix

# COMMAND ----------
correlation_matrix = df_feat.corr()
display(correlation_matrix)
