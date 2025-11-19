# Databricks notebook source
# MAGIC %md
# MAGIC # M5 Dataset Training on Azure Databricks
# MAGIC 
# MAGIC This notebook trains forecasting models on the M5 dataset stored in Azure Blob Storage.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# Import Azure configuration
try:
    from azure_config import (
        get_blob_storage_path,
        get_blob_storage_mount_path,
        validate_azure_config,
        print_config_status,
        MLFLOW_EXPERIMENT_PREFIX
    )
    
    # Print configuration status
    print_config_status()
    
    if not validate_azure_config():
        raise ValueError("Azure configuration is incomplete. Please set required variables in azure_config.py or as environment variables.")
        
except ImportError:
    print("Warning: azure_config.py not found. Using default paths.")
    def get_blob_storage_path():
        return dbutils.widgets.get("blob_path") if 'dbutils' in globals() else "/mnt/m5/sales_train_validation.csv"
    def get_blob_storage_mount_path():
        return "/mnt/m5"
    MLFLOW_EXPERIMENT_PREFIX = "/M5_Forecasting_Databricks"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Mount Blob Storage (if not already mounted)

# COMMAND ----------

try:
    from azure_config import (
        AZURE_STORAGE_ACCOUNT_NAME,
        AZURE_STORAGE_ACCOUNT_KEY,
        AZURE_STORAGE_CONTAINER_NAME
    )
    
    mount_point = get_blob_storage_mount_path()
    
    # Check if already mounted
    mounts = dbutils.fs.mounts()
    if not any(mount.mountPoint == mount_point for mount in mounts):
        print(f"Mounting blob storage to {mount_point}...")
        
        if AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY:
            dbutils.fs.mount(
                source=f"wasbs://{AZURE_STORAGE_CONTAINER_NAME}@{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
                mount_point=mount_point,
                extra_configs={
                    f"fs.azure.account.key.{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net": AZURE_STORAGE_ACCOUNT_KEY
                }
            )
            print(f"Successfully mounted to {mount_point}")
        else:
            print("Warning: Storage account credentials not configured. Using existing mount or direct path.")
    else:
        print(f"Blob storage already mounted at {mount_point}")
        
except Exception as e:
    print(f"Mounting skipped or failed: {e}")
    print("Using direct blob storage path instead.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load M5 Dataset

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, explode, row_number, date_add
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

# Get data path
try:
    m5_path = f"{get_blob_storage_mount_path()}/sales_train_validation.csv"
except:
    m5_path = get_blob_storage_path()

print(f"Loading M5 data from: {m5_path}")

# Read M5 CSV
m5_df = spark.read.format("csv").option("header", True).load(m5_path)

# Display schema
m5_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Transform M5 Data to Time Series Format

# COMMAND ----------

# Unpivot day columns (d_1, d_2, ..., d_1913) to time series format
window_spec = Window.partitionBy("item_id").orderBy(expr("monotonically_increasing_id()"))
day_columns = [f"d_{i}" for i in range(1, 1914)]  # M5 has days 1-1913

m5_transformed = (
    m5_df
    .select(
        col("item_id"),
        explode(expr(f"array({','.join(day_columns)})")).alias("Quantity")
    )
    .withColumn("day", (row_number().over(window_spec) - 1) % 1913 + 1)
    .withColumn("OrderDate", expr("date_add('2015-01-01', day - 1)"))
    .withColumn("Quantity", col("Quantity").cast(DoubleType()))
    .drop("day")
)

print(f"Transformed dataset: {m5_transformed.count()} records")
display(m5_transformed.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Training

# COMMAND ----------

# Import main training function
from main_train import main_train

# Save transformed data temporarily
temp_table = "m5_transformed_temp"
m5_transformed.write.mode("overwrite").saveAsTable(temp_table)

# Read back and convert to CSV for main_train
temp_csv_path = "/tmp/m5_for_training"
m5_transformed.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_csv_path)

# Get actual CSV file path
csv_files = dbutils.fs.ls(temp_csv_path)
csv_file = [f for f in csv_files if f.name.endswith('.csv')]
if csv_file:
    actual_csv_path = csv_file[0].path
else:
    actual_csv_path = temp_csv_path

# Run training
main_train(
    data_path=actual_csv_path,
    date_col="OrderDate",
    product_id_col="item_id",
    quantity_col="Quantity",
    month_end_col="MonthEndDate",
    experiment_prefix=MLFLOW_EXPERIMENT_PREFIX
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Training Complete
# MAGIC 
# MAGIC Check MLflow UI for experiment results and model artifacts.
