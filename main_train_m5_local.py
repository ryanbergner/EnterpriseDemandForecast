#!/usr/bin/env python
# main_train_m5_local.py
# Training script for M5 dataset on local environment

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, explode, row_number, date_add
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

# Import main training function
from main_train import main_train

def load_m5_data_local(spark: SparkSession, data_path: str):
    """
    Load and transform M5 dataset from local CSV file.
    
    Args:
        spark: SparkSession instance
        data_path: Path to M5 sales_train_validation.csv file
        
    Returns:
        DataFrame with columns: item_id, OrderDate, Quantity
    """
    print(f"Loading M5 data from: {data_path}")
    
    # Read M5 CSV (has columns: item_id, d_1, d_2, ..., d_1913)
    m5_df = spark.read.format("csv").option("header", True).load(data_path)
    
    # Unpivot day columns to time series format
    window_spec = Window.partitionBy("item_id").orderBy(expr("monotonically_increasing_id()"))
    day_columns = [f"d_{i}" for i in range(1, 1914)]  # M5 has days 1-1913
    
    m5_unpivoted = (
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
    
    print(f"Loaded {m5_unpivoted.count()} records for {m5_unpivoted.select('item_id').distinct().count()} items")
    
    return m5_unpivoted


def main():
    """Main entry point for local M5 training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train forecasting models on M5 dataset (local)")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/m5/sales_train_validation.csv",
        help="Path to M5 sales_train_validation.csv file"
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default="/M5_Forecasting_Local",
        help="MLflow experiment name prefix"
    )
    
    args = parser.parse_args()
    
    # Create Spark session
    spark = SparkSession.builder.appName("M5_Local_Training").getOrCreate()
    
    try:
        # Load and transform M5 data
        df = load_m5_data_local(spark, args.data_path)
        
        # Save to temporary location for main_train to read
        temp_path = "/tmp/m5_transformed"
        df.write.mode("overwrite").parquet(temp_path)
        
        # Read back and call main_train
        df_final = spark.read.parquet(temp_path)
        
        # Convert to CSV temporarily for main_train (or modify main_train to accept DataFrame)
        csv_temp = "/tmp/m5_for_training.csv"
        df_final.coalesce(1).write.mode("overwrite").option("header", True).csv(csv_temp)
        
        # Get the actual CSV file path (Spark creates a directory)
        csv_files = [f for f in os.listdir(csv_temp) if f.endswith('.csv')]
        if csv_files:
            actual_csv_path = os.path.join(csv_temp, csv_files[0])
        else:
            # Try subdirectory
            for root, dirs, files in os.walk(csv_temp):
                for file in files:
                    if file.endswith('.csv'):
                        actual_csv_path = os.path.join(root, file)
                        break
        
        # Call main training function
        main_train(
            data_path=actual_csv_path,
            date_col="OrderDate",
            product_id_col="item_id",
            quantity_col="Quantity",
            month_end_col="MonthEndDate",
            experiment_prefix=args.experiment_prefix
        )
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
