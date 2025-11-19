#!/usr/bin/env python
"""
main_train.py

Generic training script for forecasting models

This is a flexible training script that can work with M5 dataset or custom datasets.
It trains both Spark ML and StatsForecast models on the provided data.

Usage:
    python main_train.py --data-path /path/to/dataset.csv --config config.py
"""

import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import mlflow.pyspark.ml

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from statsforecast.models import SeasonalExponentialSmoothingOptimized, CrostonClassic

# Local project imports
from src.preprocessing.preprocess import aggregate_sales_data
from src.feature_engineering.feature_engineering import add_features
from src.model_training.ml_models import train_sparkML_models, evaluate_sparkML_models
from src.model_training.stats_models import train_stats_models, evaluate_stats_models


def main_train(
    data_path="data/m5/sales_train_validation.csv",
    date_column="OrderDate",
    product_id_column="item_id",
    quantity_column="Quantity",
    month_end_column="MonthEndDate",
    experiment_name="Forecasting_Experiment",
    max_items=100
):
    """
    Main function to train machine learning and statistical models for forecasting.

    Args:
        data_path: Path to input CSV data file
        date_column: Name of the date column in the dataset
        product_id_column: Name of the product identifier column
        quantity_column: Name of the quantity column
        month_end_column: Name of the column indicating the end of the month
        experiment_name: Name for MLflow experiment
        max_items: Maximum number of items to train on
    """
    print("="*80)
    print("FORECASTING MODEL TRAINING")
    print("="*80)
    
    # 1) Create Spark session
    print("\n[1/5] Creating Spark session...")
    spark = SparkSession.builder.appName("Forecasting_Training").getOrCreate()
    
    # 2) Read data
    print(f"\n[2/5] Reading data from: {data_path}")
    try:
        df = spark.read.format("csv").option("header", True).load(data_path)
        
        # Cast columns to appropriate types
        df = (
            df.withColumn(date_column, col(date_column).cast("date"))
              .withColumn(quantity_column, col(quantity_column).cast("double"))
        )
        
        print(f"[INFO] Loaded dataset: {df.count()} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        spark.stop()
        return
    
    # 3) Aggregate and feature engineering
    print("\n[3/5] Aggregating and engineering features...")
    df_agg = aggregate_sales_data(
        df,
        date_column,
        product_id_column,
        quantity_column,
        month_end_column
    )
    
    df_feat = add_features(
        df_agg,
        month_end_column,
        product_id_column,
        quantity_column
    )
    
    # Filter out products with insufficient data
    df_feat = df_feat.filter(col("total_orders") >= 5)
    print(f"[INFO] Feature-engineered dataset: {df_feat.count()} rows")
    
    # Sample items if needed
    distinct_items = df_feat.select(product_id_column).distinct()
    total_items = distinct_items.count()
    print(f"[INFO] Total distinct items: {total_items}")
    
    if total_items > max_items:
        print(f"[INFO] Sampling {max_items} items for training...")
        fraction = max_items / float(total_items)
        sampled_items = distinct_items.sample(withReplacement=False, fraction=fraction, seed=42)
        df_feat = df_feat.join(sampled_items, on=product_id_column, how="inner")
    
    # 4) Prepare train/test split
    print("\n[4/5] Preparing train/test split (80/20)...")
    df_feat = df_feat.withColumn("unix_timestamp", unix_timestamp(col(month_end_column)))
    df_with_unix = df_feat.filter(col("unix_timestamp").isNotNull()).dropna()
    
    # BUG FIX: Check if approxQuantile returns empty list before accessing [0]
    percentile_list = df_with_unix.approxQuantile("unix_timestamp", [0.8], 0)
    if not percentile_list:
        print("[ERROR] Could not compute quantile for train/test split. Exiting.")
        spark.stop()
        return
    
    percentile_timestamp = percentile_list[0]
    percentile_date = datetime.datetime.fromtimestamp(percentile_timestamp).strftime("%Y-%m-%d")
    print(f"[INFO] Split cutoff date: {percentile_date}")
    
    train_df = df_with_unix.filter(col(month_end_column) < percentile_date).dropna()
    test_df = df_with_unix.filter(col(month_end_column) >= percentile_date).dropna()
    
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"[INFO] Training set: {train_count} rows")
    print(f"[INFO] Test set: {test_count} rows")
    
    if train_count == 0 or test_count == 0:
        print("[ERROR] Empty train or test set. Exiting.")
        spark.stop()
        return
    
    # Define features and target
    target_col = "lead_month_1"
    list_of_features = [
        quantity_column,
        "months_since_last_order",
        "last_order_quantity",
        "month",
        "year"
    ]
    
    # Filter to available features
    available_features = [f for f in list_of_features if f in df_feat.columns]
    print(f"[INFO] Using features: {available_features}")
    
    # 5) Define and train models
    print("\n[5/5] Training models...")
    
    # Create MLflow experiment
    experiment_name_full = f"/Users/{experiment_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    try:
        experiment_id = mlflow.create_experiment(experiment_name_full)
    except mlflow.exceptions.MlflowException:
        exp = mlflow.get_experiment_by_name(experiment_name_full)
        experiment_id = exp.experiment_id if exp else None
    
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"[INFO] MLflow experiment: {experiment_name_full}")
    
    # Define Spark ML models
    lr = LinearRegression(labelCol=target_col)
    rf = RandomForestRegressor(labelCol=target_col)
    gbt = GBTRegressor(labelCol=target_col)
    
    rfParamGrid = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, [2, 3, 5])
        .addGrid(rf.numTrees, [10, 50])
        .build()
    )
    
    gbtParamGrid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [2, 3])
        .addGrid(gbt.maxIter, [10])
        .build()
    )
    
    spark_ml_models = [
        {"alias": "LR_model", "model": lr, "param_grid": None},
        {"alias": "RF_model", "model": rf, "param_grid": rfParamGrid},
        {"alias": "GBT_model", "model": gbt, "param_grid": gbtParamGrid},
    ]
    
    # Define Statistical models
    stats_models = [
        {
            "alias": "SeasonalExponentialSmoothing",
            "model": SeasonalExponentialSmoothingOptimized(season_length=12)
        },
        {
            "alias": "CrostonClassic",
            "model": CrostonClassic()
        }
    ]
    
    # Train all models
    with mlflow.start_run(experiment_id=experiment_id, run_name="All_Models_Training") as parent_run:
        mlflow.log_param("train_count", train_count)
        mlflow.log_param("test_count", test_count)
        mlflow.log_param("num_features", len(available_features))
        
        # Train Statistical Models
        print("\n--- Training Statistical Models ---")
        for stats_info in stats_models:
            stats_alias = stats_info["alias"]
            stats_model = stats_info["model"]
            
            with mlflow.start_run(run_name=stats_alias, nested=True):
                print(f"\n[Stats] Training {stats_alias}...")
                try:
                    trained_stats = train_stats_models(
                        models=[stats_model],
                        train_df=train_df,
                        month_end_column=month_end_column,
                        product_id_column=product_id_column,
                        target_column=target_col
                    )
                    
                    evaluate_stats_models(
                        stats_model=trained_stats,
                        test_df=test_df,
                        month_end_column=month_end_column,
                        product_id_column=product_id_column,
                        target_column=target_col,
                        experiment_id=experiment_id,
                        artifact_location=None,
                        model_name=stats_alias
                    )
                    print(f"[Stats] {stats_alias} completed successfully")
                except Exception as e:
                    print(f"[Stats] Error training {stats_alias}: {e}")
        
        # Train Spark ML Models
        print("\n--- Training Spark ML Models ---")
        for ml_dict in spark_ml_models:
            ml_alias = ml_dict["alias"]
            ml_model = ml_dict["model"]
            param_grid = ml_dict["param_grid"]
            
            with mlflow.start_run(run_name=ml_alias, nested=True):
                print(f"\n[SparkML] Training {ml_alias}...")
                try:
                    pipeline_model = train_sparkML_models(
                        model=ml_model,
                        train_df=train_df.select(*available_features, target_col),
                        featuresCols=available_features,
                        labelCol=target_col,
                        paramGrid=param_grid
                    )
                    metrics = evaluate_sparkML_models(
                        model=pipeline_model,
                        test_df=test_df.select(*available_features, target_col),
                        features_cols=available_features,
                        label_col=target_col,
                        requirements_path=None,
                        model_alias=ml_alias
                    )
                    print(f"[SparkML] {ml_alias} completed: {metrics}")
                except Exception as e:
                    print(f"[SparkML] Error training {ml_alias}: {e}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"View results with: mlflow ui")
    print("="*80)
    
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/m5/sales_train_validation.csv",
        help="Path to input CSV data file"
    )
    parser.add_argument(
        "--date-column",
        type=str,
        default="OrderDate",
        help="Name of the date column"
    )
    parser.add_argument(
        "--product-id-column",
        type=str,
        default="item_id",
        help="Name of the product ID column"
    )
    parser.add_argument(
        "--quantity-column",
        type=str,
        default="Quantity",
        help="Name of the quantity column"
    )
    parser.add_argument(
        "--month-end-column",
        type=str,
        default="MonthEndDate",
        help="Name of the month-end column"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Forecasting_Experiment",
        help="Name for MLflow experiment"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=100,
        help="Maximum number of items to train on"
    )
    
    args = parser.parse_args()
    
    main_train(
        data_path=args.data_path,
        date_column=args.date_column,
        product_id_column=args.product_id_column,
        quantity_column=args.quantity_column,
        month_end_column=args.month_end_column,
        experiment_name=args.experiment_name,
        max_items=args.max_items
    )
