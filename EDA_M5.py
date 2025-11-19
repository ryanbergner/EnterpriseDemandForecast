# Databricks notebook source
# MAGIC %md
# MAGIC # Comprehensive EDA for M5 Forecasting Dataset
# MAGIC 
# MAGIC This notebook consolidates exploratory data analysis, feature engineering exploration, and model comparison for the M5 forecasting dataset.

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Loading and Initial Exploration

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, count, sum as spark_sum, avg as spark_avg,
    min as spark_min, max as spark_max, stddev_samp, 
    expr, explode, row_number, date_add, month, year
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import project modules
from src.preprocessing.preprocess import aggregate_sales_data
from src.feature_engineering.feature_engineering import add_features

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load M5 Dataset
# MAGIC 
# MAGIC The M5 dataset contains sales data with columns: item_id, d_1, d_2, ..., d_1913
# MAGIC We need to unpivot the day columns into rows.

# COMMAND ----------

# For local execution, adjust path as needed
# For Databricks, use blob storage path configured in azure_config
try:
    # Try Databricks environment first
    m5_path = dbutils.widgets.get("m5_data_path") if 'dbutils' in globals() else None
    if m5_path is None:
        m5_path = "/mnt/m5/sales_train_validation.csv"  # Default Databricks path
except:
    m5_path = "data/m5/sales_train_validation.csv"  # Default local path

# Read M5 data
m5_sdf = spark.read.format("csv").option("header", True).load(m5_path)

# Limit for initial exploration (remove limit for full dataset)
sample_size = 100000  # Adjust as needed
m5_sdf = m5_sdf.limit(sample_size)

display(m5_sdf.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unpivot Day Columns to Time Series Format

# COMMAND ----------

# Create window specification partitioned by item_id
window_spec = Window.partitionBy("item_id").orderBy(expr("monotonically_increasing_id()"))
day_columns = [f"d_{i}" for i in range(1, 1914)]  # M5 has days 1-1913

# Unpivot the DataFrame
m5_sdf = (
    m5_sdf
    .select(
        col("item_id"),
        explode(expr(f"array({','.join(day_columns)})")).alias("Quantity")
    )
    .withColumn("day", (row_number().over(window_spec) - 1) % 1913 + 1)
    .withColumn("OrderDate", expr("date_add('2015-01-01', day - 1)"))
    .withColumn("Quantity", col("Quantity").cast(DoubleType()))
)

display(m5_sdf.tail(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic Statistics

# COMMAND ----------

print(f"Total records: {m5_sdf.count()}")
print(f"Unique items: {m5_sdf.select('item_id').distinct().count()}")
print(f"Date range: {m5_sdf.agg(spark_min('OrderDate'), spark_max('OrderDate')).collect()[0]}")

# Summary statistics
m5_sdf.select("Quantity").describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Aggregation and Feature Engineering

# COMMAND ----------

# Define column mappings for M5 dataset
date_column = "OrderDate"
product_id_column = "item_id"
quantity_column = "Quantity"
month_end_column = "MonthEndDate"

# Aggregate to monthly level
df_agg = aggregate_sales_data(
    m5_sdf, 
    date_column, 
    product_id_column, 
    quantity_column, 
    month_end_column
)

display(df_agg.head(10))

# COMMAND ----------

# Add comprehensive features
df_feat = add_features(
    df_agg, 
    month_end_column, 
    product_id_column, 
    quantity_column
)

print(f"Feature-engineered dataset shape: {df_feat.count()} rows, {len(df_feat.columns)} columns")
display(df_feat.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Summary

# COMMAND ----------

print("Available features:")
for col_name in sorted(df_feat.columns):
    print(f"  - {col_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Product Category Distribution

# COMMAND ----------

category_dist = (
    df_feat
    .groupBy("product_category")
    .agg(
        count("*").alias("count"),
        spark_avg(quantity_column).alias("avg_demand"),
        spark_avg("cov_quantity").alias("avg_cov"),
        spark_avg("avg_demand_interval").alias("avg_interval")
    )
    .orderBy("count", ascending=False)
)

display(category_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Quality Analysis

# COMMAND ----------

# Check for missing values
missing_stats = {}
for col_name in df_feat.columns:
    null_count = df_feat.filter(col(col_name).isNull()).count()
    total_count = df_feat.count()
    missing_stats[col_name] = {
        "null_count": null_count,
        "null_percentage": (null_count / total_count * 100) if total_count > 0 else 0
    }

missing_df = pd.DataFrame(missing_stats).T
missing_df = missing_df[missing_df["null_count"] > 0].sort_values("null_count", ascending=False)
if len(missing_df) > 0:
    print("Columns with missing values:")
    display(missing_df)
else:
    print("No missing values found!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Zero Demand Analysis

# COMMAND ----------

zero_demand_stats = (
    df_feat
    .groupBy(product_id_column)
    .agg(
        count("*").alias("total_months"),
        spark_sum(when(col(quantity_column) == 0, 1).otherwise(0)).alias("zero_months")
    )
    .withColumn("zero_rate", col("zero_months") / col("total_months"))
)

display(zero_demand_stats.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Time Series Visualization

# COMMAND ----------

# Sample a few products for visualization
sample_products = (
    df_feat
    .select(product_id_column)
    .distinct()
    .limit(5)
    .collect()
)

sample_product_ids = [row[product_id_column] for row in sample_products]

# Convert to Pandas for visualization
df_viz = (
    df_feat
    .filter(col(product_id_column).isin(sample_product_ids))
    .select(product_id_column, month_end_column, quantity_column)
    .orderBy(product_id_column, month_end_column)
    .toPandas()
)

# Plot time series
fig, axes = plt.subplots(len(sample_product_ids), 1, figsize=(15, 4 * len(sample_product_ids)))
if len(sample_product_ids) == 1:
    axes = [axes]

for idx, product_id in enumerate(sample_product_ids):
    product_data = df_viz[df_viz[product_id_column] == product_id]
    axes[idx].plot(product_data[month_end_column], product_data[quantity_column])
    axes[idx].set_title(f"Time Series for {product_id}")
    axes[idx].set_xlabel("Date")
    axes[idx].set_ylabel("Quantity")
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature Correlation Analysis

# COMMAND ----------

# Select numeric features for correlation
numeric_features = [
    quantity_column, "month", "year", "months_since_last_order",
    "last_order_quantity", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
    "ma_4_month", "ma_8_month", "ma_12_month", "cov_quantity",
    "avg_demand_interval", "total_orders"
]

# Sample data for correlation (convert to Pandas)
df_corr = (
    df_feat
    .select(*numeric_features)
    .limit(10000)  # Sample for performance
    .toPandas()
    .dropna()
)

correlation_matrix = df_corr.corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
display(plt.gcf())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Performance Comparison Setup

# COMMAND ----------

# MAGIC %md
# MAGIC This section prepares the data for model training and comparison.
# MAGIC Actual model training should be done using the main training scripts:
# MAGIC - `main_train_m5_local.py` for local execution
# MAGIC - `main_train_m5_databricks.py` for Azure Databricks

# COMMAND ----------

# Prepare train/test split for evaluation
from pyspark.sql.functions import unix_timestamp

df_feat = df_feat.withColumn("unix_time", unix_timestamp(col(month_end_column)))
df_feat = df_feat.filter(col("unix_time").isNotNull())

# Calculate 80th percentile for train/test split
approx_list = df_feat.approxQuantile("unix_time", [0.8], 0)
if approx_list:
    cutoff_ts = approx_list[0]
    cutoff_date = datetime.fromtimestamp(cutoff_ts)
    print(f"Train/Test split cutoff date: {cutoff_date}")
    
    train_df = df_feat.filter(col("unix_time") < cutoff_ts)
    test_df = df_feat.filter(col("unix_time") >= cutoff_ts)
    
    print(f"Training set size: {train_df.count()}")
    print(f"Test set size: {test_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary and Next Steps

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Findings:
# MAGIC 1. Dataset contains sales data for multiple items over time
# MAGIC 2. Features have been engineered including lags, moving averages, and product categories
# MAGIC 3. Data is ready for model training
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC 1. Run training scripts to train models:
# MAGIC    - Local: `python main_train_m5_local.py`
# MAGIC    - Databricks: Use `main_train_m5_databricks.py` notebook
# MAGIC 2. Compare model performance using MLflow
# MAGIC 3. Generate predictions using `main_inference.py`
