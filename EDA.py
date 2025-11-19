# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------
# MAGIC %md
# MAGIC # Comprehensive Exploratory Data Analysis (EDA) for M5 Dataset
# MAGIC 
# MAGIC This notebook provides a complete EDA workflow including:
# MAGIC - Data loading and preprocessing
# MAGIC - Feature engineering
# MAGIC - Statistical analysis
# MAGIC - Visualization
# MAGIC - Model performance comparison (Spark ML, StatsForecast, Neural Networks)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------
import warnings
warnings.filterwarnings("ignore")

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, lag, lead, sum as spark_sum, stddev_samp,
    month, year, lit, to_date, unix_timestamp, avg as spark_avg,
    rand, coalesce, abs as spark_abs, broadcast, count, last_day,
    expr, explode, sequence, date_format, last, months_between,
    mean, monotonically_increasing_id, row_number
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType,
    DoubleType, IntegerType, DecimalType
)

# Spark ML imports
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# StatsForecast imports
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, HoltWinters, ARIMA, CrostonClassic,
    SeasonalExponentialSmoothingOptimized
)

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# TabPFN imports
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("TabPFN not available. Install with: pip install tabpfn")

# Visualization imports
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import numpy as np

# MLflow imports
import mlflow
import mlflow.sklearn
import mlflow.pyspark.ml
from mlflow.models.signature import infer_signature
try:
    import mlflavors
    import mlflavors.statsforecast
except ImportError:
    print("mlflavors not available. Install with: pip install mlflavors")

# Project imports
from src.preprocessing.preprocess import aggregate_sales_data, retrieve_sales_data
from src.feature_engineering.feature_engineering import add_features
from src.model_training.ml_models import (
    train_sparkML_models,
    evaluate_sparkML_models
)
from src.model_training.stats_models import (
    train_stats_models,
    evaluate_stats_models
)

# Standard library imports
from typing import Optional, List, Any, Dict, Tuple
from math import sqrt as math_sqrt, isclose
import datetime

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Load M5 Dataset

# COMMAND ----------
# Define column mappings for M5 dataset
date_column = "OrderDate"
product_id_column = "item_id"
quantity_column = "Quantity"
month_end_column = "MonthEndDate"

# Load M5 dataset
# For local: use local file path
# For Databricks: use DBFS or mounted blob storage path
try:
    # Try loading from mounted storage first (Databricks)
    df = spark.read.csv("/mnt/m5/sales_train_validation.csv", header=True)
    print("Loaded M5 dataset from mounted storage")
except:
    # Fallback to local path
    try:
        df = spark.read.csv("data/m5/sales_train_validation.csv", header=True)
        print("Loaded M5 dataset from local path")
    except:
        print("ERROR: Could not load M5 dataset. Please check the path.")
        print("Expected locations:")
        print("  - Databricks: /mnt/m5/sales_train_validation.csv")
        print("  - Local: data/m5/sales_train_validation.csv")

# Display basic info
print(f"Dataset shape: {df.count()} rows, {len(df.columns)} columns")
display(df.limit(5))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Data Transformation and Preprocessing
# MAGIC
# MAGIC The M5 dataset comes in a wide format (d_1, d_2, ... d_1913).
# MAGIC We need to unpivot it to long format for time series analysis.

# COMMAND ----------
# Unpivot the M5 dataset
day_columns = [f"d_{i}" for i in range(1, 1914)]
window_spec = Window.partitionBy("item_id").orderBy(monotonically_increasing_id())

df_unpivoted = (
    df.select(
        col("item_id"),
        col("store_id"),
        col("cat_id"),
        col("dept_id"),
        col("state_id"),
        explode(expr(f"array({','.join(day_columns)})")).alias(quantity_column)
    )
    .withColumn("day_num", row_number().over(window_spec))
    .withColumn(date_column, expr("date_add('2011-01-29', day_num - 1)"))  # M5 starts at 2011-01-29
    .withColumn(quantity_column, col(quantity_column).cast(DoubleType()))
)

print(f"Unpivoted dataset shape: {df_unpivoted.count()} rows, {len(df_unpivoted.columns)} columns")
display(df_unpivoted.limit(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Feature Engineering

# COMMAND ----------
# Aggregate sales data by month
df_agg = aggregate_sales_data(
    df=df_unpivoted,
    date_column=date_column,
    product_id_column=product_id_column,
    quantity_column=quantity_column,
    month_end_column=month_end_column
)

print(f"Aggregated dataset: {df_agg.count()} rows")
display(df_agg.limit(10))

# COMMAND ----------
# Add engineered features
df_feat = add_features(
    df_agg,
    month_end_column=month_end_column,
    product_id_column=product_id_column,
    quantity_column=quantity_column
)

# Filter out products with insufficient data
df_feat = df_feat.filter(col("total_orders") >= 5)

print(f"Feature-engineered dataset: {df_feat.count()} rows, {len(df_feat.columns)} columns")
print(f"\nAvailable features: {df_feat.columns}")
display(df_feat.limit(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Exploratory Data Analysis

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5.1 Basic Statistics

# COMMAND ----------
# Summary statistics
df_feat.select(quantity_column, "month", "year", "total_orders").describe().show()

# Distribution of products by category (if available)
if "product_category" in df_feat.columns:
    df_feat.groupBy("product_category").count().orderBy(col("count").desc()).show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5.2 Time Series Patterns

# COMMAND ----------
# Convert to Pandas for visualization
sample_products = df_feat.select(product_id_column).distinct().limit(5).rdd.flatMap(lambda x: x).collect()
df_sample = df_feat.filter(col(product_id_column).isin(sample_products)).toPandas()

# Plot time series for sample products
plt.figure(figsize=(15, 8))
for i, product in enumerate(sample_products, 1):
    plt.subplot(2, 3, i)
    product_data = df_sample[df_sample[product_id_column] == product].sort_values(month_end_column)
    plt.plot(product_data[month_end_column], product_data[quantity_column])
    plt.title(f"Product: {product}")
    plt.xlabel("Date")
    plt.ylabel("Quantity")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### 5.3 Correlation Analysis

# COMMAND ----------
# Select numeric columns for correlation
numeric_cols = [
    quantity_column, "ma_4_month", "ma_8_month", "ma_12_month",
    "cov_quantity", "avg_demand_quantity", "months_since_last_order",
    "last_order_quantity", "month", "year", "lead_month_1"
]

# Filter to columns that exist
available_numeric_cols = [c for c in numeric_cols if c in df_feat.columns]

# Get correlation matrix
correlation_df = df_feat.select(available_numeric_cols).limit(10000).toPandas()
correlation_matrix = correlation_df.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Model Training and Comparison

# COMMAND ----------
# MAGIC %md
# MAGIC ### 6.1 Prepare Training and Test Sets

# COMMAND ----------
# Define feature lists
list_of_features = [
    quantity_column,
    "ma_4_month",
    "ma_8_month",
    "ma_12_month",
    "avg_demand_quantity",
    "avg_demand_interval",
    "months_since_last_order",
    "last_order_quantity",
    "month"
]

# Filter to available features
list_of_features = [f for f in list_of_features if f in df_feat.columns]
target_column = "lead_month_1"

print(f"Using features: {list_of_features}")
print(f"Target column: {target_column}")

# Create train/test split based on date (80/20)
df_feat = df_feat.withColumn("unix_time", unix_timestamp(col(month_end_column)))
df_feat = df_feat.filter(col("unix_time").isNotNull())

approx_list = df_feat.approxQuantile("unix_time", [0.8], 0)
if approx_list:
    cutoff_ts = approx_list[0]
    cutoff_date = datetime.datetime.fromtimestamp(cutoff_ts)
    print(f"Train/Test split cutoff date: {cutoff_date}")
    
    train_spark = df_feat.filter(col("unix_time") < cutoff_ts).dropna()
    test_spark = df_feat.filter(col("unix_time") >= cutoff_ts).dropna()
    
    print(f"Training set: {train_spark.count()} rows")
    print(f"Test set: {test_spark.count()} rows")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 6.2 Spark ML Models

# COMMAND ----------
# Define Spark ML models
gbt = GBTRegressor(labelCol=target_column)
GBTParamGrid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [2, 3, 5])
    .addGrid(gbt.maxIter, [10])
    .build()
)

rf = RandomForestRegressor(labelCol=target_column)
rfParamGrid = (
    ParamGridBuilder()
    .addGrid(rf.maxDepth, [2, 3, 5])
    .addGrid(rf.numTrees, [10, 50])
    .build()
)

lr = LinearRegression(labelCol=target_column)

spark_ml_models = [
    {"alias": "LinearRegression", "model": lr, "param_grid": None},
    {"alias": "RandomForest", "model": rf, "param_grid": rfParamGrid},
    {"alias": "GradientBoostedTrees", "model": gbt, "param_grid": GBTParamGrid}
]

# Train and evaluate each model
spark_results = {}
for model_dict in spark_ml_models:
    print(f"\n=== Training {model_dict['alias']} ===")
    try:
        pipeline_model = train_sparkML_models(
            model=model_dict['model'],
            train_df=train_spark.select(*list_of_features, target_column),
            featuresCols=list_of_features,
            labelCol=target_column,
            paramGrid=model_dict['param_grid']
        )
        
        metrics = evaluate_sparkML_models(
            model=pipeline_model,
            test_df=test_spark.select(*list_of_features, target_column),
            features_cols=list_of_features,
            label_col=target_column,
            requirements_path=None,
            model_alias=model_dict['alias']
        )
        
        spark_results[model_dict['alias']] = metrics
        print(f"Results: {metrics}")
    except Exception as e:
        print(f"Error training {model_dict['alias']}: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 6.3 Statistical Forecasting Models

# COMMAND ----------
# Define StatsForecast models
stats_models = [
    {"alias": "SeasonalES", "model": SeasonalExponentialSmoothingOptimized(season_length=12)},
    {"alias": "CrostonClassic", "model": CrostonClassic()}
]

stats_results = {}
for stats_info in stats_models:
    print(f"\n=== Training {stats_info['alias']} ===")
    try:
        trained_stats = train_stats_models(
            models=[stats_info['model']],
            train_df=train_spark,
            month_end_column=month_end_column,
            product_id_column=product_id_column,
            target_column=target_column
        )
        
        # Note: evaluate_stats_models logs to MLflow
        # For EDA purposes, we'll just note it was trained
        print(f"{stats_info['alias']} trained successfully")
        stats_results[stats_info['alias']] = "Trained"
    except Exception as e:
        print(f"Error training {stats_info['alias']}: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 6.4 TabPFN Model (if available)

# COMMAND ----------
if TABPFN_AVAILABLE:
    print("=== Training TabPFN ===")
    
    # Convert to pandas and limit to 10k rows for TabPFN
    train_pdf = train_spark.select(*list_of_features, target_column).limit(10000).toPandas().dropna()
    test_pdf = test_spark.select(*list_of_features, target_column).limit(2000).toPandas().dropna()
    
    if len(train_pdf) > 0 and len(test_pdf) > 0:
        X_train = train_pdf[list_of_features].values
        y_train = train_pdf[target_column].values
        X_test = test_pdf[list_of_features].values
        y_test = test_pdf[target_column].values
        
        try:
            reg = TabPFNRegressor(device="auto")
            reg.fit(X_train, y_train)
            
            preds = reg.predict(X_test)
            mse_val = mean_squared_error(y_test, preds)
            rmse_val = float(np.sqrt(mse_val))
            mae_val = float(mean_absolute_error(y_test, preds))
            r2_val = float(r2_score(y_test, preds))
            
            print(f"TabPFN Results:")
            print(f"  MSE: {mse_val:.3f}")
            print(f"  RMSE: {rmse_val:.3f}")
            print(f"  MAE: {mae_val:.3f}")
            print(f"  R²: {r2_val:.3f}")
        except Exception as e:
            print(f"Error training TabPFN: {e}")
else:
    print("TabPFN not available. Skipping.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### 6.5 Neural Network Models

# COMMAND ----------
print("=== Training Neural Network ===")

# Prepare data for neural network
train_pdf = train_spark.select(*list_of_features, target_column).limit(50000).toPandas().dropna()
test_pdf = test_spark.select(*list_of_features, target_column).limit(10000).toPandas().dropna()

if len(train_pdf) > 0 and len(test_pdf) > 0:
    X_train = train_pdf[list_of_features].values
    y_train = train_pdf[target_column].values
    X_test = test_pdf[list_of_features].values
    y_test = test_pdf[target_column].values
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build simple neural network
    model = Sequential([
        InputLayer(input_shape=(len(list_of_features),)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.R2Score()]
    )
    
    # Train with early stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    nn_preds = model.predict(X_test_scaled).flatten()
    nn_mse = mean_squared_error(y_test, nn_preds)
    nn_rmse = np.sqrt(nn_mse)
    nn_mae = mean_absolute_error(y_test, nn_preds)
    nn_r2 = r2_score(y_test, nn_preds)
    
    print(f"\nNeural Network Results:")
    print(f"  MSE: {nn_mse:.3f}")
    print(f"  RMSE: {nn_rmse:.3f}")
    print(f"  MAE: {nn_mae:.3f}")
    print(f"  R²: {nn_r2:.3f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, nn_preds, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Model Comparison Summary

# COMMAND ----------
print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)

# Spark ML results
if spark_results:
    print("\nSpark ML Models:")
    for model_name, metrics in spark_results.items():
        print(f"  {model_name}: {metrics}")

# Stats models
if stats_results:
    print("\nStatistical Models:")
    for model_name, status in stats_results.items():
        print(f"  {model_name}: {status}")

print("\n" + "="*60)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Conclusions and Recommendations
# MAGIC 
# MAGIC Based on the EDA and model comparisons:
# MAGIC 
# MAGIC 1. **Data Quality**: Review the data quality metrics and handle missing values appropriately
# MAGIC 2. **Feature Importance**: Identify which features contribute most to predictions
# MAGIC 3. **Model Selection**: Choose the best performing model for your use case
# MAGIC 4. **Hyperparameter Tuning**: Further optimize the best performing models
# MAGIC 5. **Production Deployment**: Consider ensemble methods for improved robustness
