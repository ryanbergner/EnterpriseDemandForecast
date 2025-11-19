#!/usr/bin/env python
"""
main_train_databricks.py

Training script for M5 dataset - Azure Databricks with Blob Storage

This script trains both Spark ML and StatsForecast models on the M5 forecasting dataset
loaded from Azure Blob Storage. It includes all ML and statistical models in a single pipeline.

Usage:
    # In Databricks notebook:
    %run /Workspace/path/to/main_train_databricks
    
    # Or as a job with parameters:
    python main_train_databricks.py --max-items 500 --experiment-name M5_Production
"""

import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import mlflow.pyspark.ml

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, expr, explode, row_number,
    monotonically_increasing_id, lit, when, unix_timestamp
)
from pyspark.sql.window import Window
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.tuning import ParamGridBuilder
from statsforecast.models import SeasonalExponentialSmoothingOptimized, CrostonClassic

# Local imports
from src.preprocessing.preprocess import aggregate_sales_data
from src.feature_engineering.feature_engineering import add_features
from src.model_training.ml_models import train_sparkML_models, evaluate_sparkML_models
from src.model_training.stats_models import train_stats_models, evaluate_stats_models

# Azure configuration
try:
    from azure_config import (
        setup_spark_blob_storage,
        get_m5_data_paths,
        validate_azure_config,
        M5_SALES_TRAIN_VALIDATION,
        MLFLOW_TRACKING_URI,
        MLFLOW_EXPERIMENT_PATH
    )
    AZURE_CONFIG_AVAILABLE = True
except ImportError:
    print("[WARNING] azure_config.py not found. Using default paths.")
    AZURE_CONFIG_AVAILABLE = False


def setup_databricks_environment(spark):
    """
    Setup Databricks environment with Azure Blob Storage access.
    
    Args:
        spark: SparkSession object
        
    Returns:
        SparkSession: Configured Spark session
    """
    print("[INFO] Setting up Databricks environment...")
    
    if AZURE_CONFIG_AVAILABLE:
        # Validate Azure configuration
        is_valid, missing = validate_azure_config()
        if not is_valid:
            print(f"[WARNING] Azure configuration incomplete. Missing: {missing}")
            print("[INFO] Attempting to use Databricks secrets...")
            
            try:
                # Try to use Databricks secrets
                from pyspark.dbutils import DBUtils
                dbutils = DBUtils(spark)
                
                # Example: Get from Databricks secrets scope
                account_name = dbutils.secrets.get(scope="azure-storage", key="account-name")
                account_key = dbutils.secrets.get(scope="azure-storage", key="account-key")
                
                spark.conf.set(
                    f"fs.azure.account.key.{account_name}.blob.core.windows.net",
                    account_key
                )
                print("[INFO] Successfully configured using Databricks secrets")
            except Exception as e:
                print(f"[WARNING] Could not configure from Databricks secrets: {e}")
        else:
            # Use azure_config.py
            spark = setup_spark_blob_storage(spark)
            print("[INFO] Successfully configured using azure_config.py")
    else:
        print("[INFO] No Azure configuration found. Assuming Databricks has access configured.")
    
    return spark


def load_m5_from_blob(spark, blob_path=None, limit=None):
    """
    Load and transform M5 dataset from Azure Blob Storage.
    
    Args:
        spark: SparkSession object
        blob_path: Path to M5 CSV in blob storage (wasbs:// format)
        limit: Optional limit on number of rows to process
        
    Returns:
        DataFrame: Unpivoted M5 dataset
    """
    # Determine path
    if blob_path is None:
        if AZURE_CONFIG_AVAILABLE:
            blob_path = M5_SALES_TRAIN_VALIDATION
        else:
            # Default Databricks mount path
            blob_path = "/mnt/m5/sales_train_validation.csv"
    
    print(f"[INFO] Loading M5 dataset from: {blob_path}")
    
    # Read CSV file
    try:
        if limit:
            df = spark.read.csv(blob_path, header=True).limit(limit)
            print(f"[INFO] Limited to {limit} products for testing")
        else:
            df = spark.read.csv(blob_path, header=True)
        
        print(f"[INFO] Loaded {df.count()} products")
    except Exception as e:
        print(f"[ERROR] Failed to load data from {blob_path}")
        print(f"[ERROR] {e}")
        raise
    
    # Define day columns (M5 has 1913 days)
    day_columns = [f"d_{i}" for i in range(1, 1914)]
    
    # Create window for row numbering
    window_spec = Window.partitionBy("item_id").orderBy(monotonically_increasing_id())
    
    # Unpivot the dataset
    print("[INFO] Unpivoting dataset from wide to long format...")
    df_unpivoted = (
        df.select(
            col("item_id"),
            col("store_id"),
            col("cat_id"),
            col("dept_id"),
            col("state_id"),
            explode(expr(f"array({','.join(day_columns)})")).alias("Quantity")
        )
        .withColumn("day_num", row_number().over(window_spec))
        .withColumn("OrderDate", expr("date_add('2011-01-29', day_num - 1)"))  # M5 starts 2011-01-29
        .withColumn("Quantity", col("Quantity").cast("double"))
        .drop("day_num")
    )
    
    print(f"[INFO] Unpivoted dataset: {df_unpivoted.count()} rows")
    return df_unpivoted


def main_train_databricks(
    blob_path=None,
    experiment_name="M5_Forecasting_Databricks",
    limit_products=None,
    max_distinct_items=100,
    user_email=None
):
    """
    Main training function for M5 dataset on Azure Databricks.
    
    Args:
        blob_path: Path to M5 CSV in blob storage
        experiment_name: Name for MLflow experiment
        limit_products: Optional limit on products to load (for testing)
        max_distinct_items: Maximum number of distinct items to train on
        user_email: Databricks user email for experiment path
    """
    print("="*80)
    print("M5 FORECASTING - AZURE DATABRICKS TRAINING")
    print("="*80)
    
    # 1) Get Spark session (already available in Databricks)
    print("\n[1/6] Getting Spark session...")
    try:
        spark = SparkSession.builder.getOrCreate()
    except:
        spark = SparkSession.builder.appName("M5_Forecasting_Databricks").getOrCreate()
    
    # Setup Azure Blob Storage access
    spark = setup_databricks_environment(spark)
    
    # 2) Load M5 dataset from blob storage
    print("\n[2/6] Loading M5 dataset from Azure Blob Storage...")
    df = load_m5_from_blob(spark, blob_path=blob_path, limit=limit_products)
    
    # Column mappings
    date_column = "OrderDate"
    product_id_column = "item_id"
    quantity_column = "Quantity"
    month_end_column = "MonthEndDate"
    target_column = "lead_month_1"
    
    # 3) Aggregate and add features
    print("\n[3/6] Aggregating sales data...")
    df_agg = aggregate_sales_data(
        df,
        date_column,
        product_id_column,
        quantity_column,
        month_end_column
    )
    
    print("[3/6] Adding engineered features...")
    df_feat = add_features(df_agg, month_end_column, product_id_column, quantity_column)
    
    # Filter out products with insufficient data
    df_feat = df_feat.filter(col("total_orders") >= 5)
    print(f"[3/6] Feature-engineered dataset: {df_feat.count()} rows")
    
    # Sample distinct items if needed
    distinct_items = df_feat.select(product_id_column).distinct()
    total_items = distinct_items.count()
    print(f"[INFO] Found {total_items} distinct items")
    
    if total_items > max_distinct_items:
        print(f"[INFO] Sampling {max_distinct_items} items for training")
        fraction = max_distinct_items / float(total_items)
        distinct_items_sampled = distinct_items.sample(withReplacement=False, fraction=fraction, seed=42)
        df_feat = df_feat.join(distinct_items_sampled, on=product_id_column, how="inner")
    
    # 4) Prepare train/test split
    print("\n[4/6] Preparing train/test split...")
    df_feat = df_feat.withColumn("unix_time", unix_timestamp(col(month_end_column)))
    df_feat = df_feat.filter(col("unix_time").isNotNull()).dropna()
    
    # 80/20 split based on date
    approx_list = df_feat.approxQuantile("unix_time", [0.8], 0)
    if not approx_list:
        print("[ERROR] Could not compute quantile for train/test split")
        return
    
    cutoff_ts = approx_list[0]
    cutoff_date = datetime.datetime.fromtimestamp(cutoff_ts)
    print(f"[INFO] Train/Test split cutoff date: {cutoff_date}")
    
    train_df = df_feat.filter(col("unix_time") < cutoff_ts).dropna()
    test_df = df_feat.filter(col("unix_time") >= cutoff_ts).dropna()
    
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"[INFO] Training set: {train_count} rows")
    print(f"[INFO] Test set: {test_count} rows")
    
    if train_count == 0 or test_count == 0:
        print("[ERROR] Empty train or test set. Exiting.")
        return
    
    # Define feature columns
    feature_cols = [
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
    available_features = [f for f in feature_cols if f in df_feat.columns]
    print(f"[INFO] Using {len(available_features)} features: {available_features}")
    
    # 5) Define models
    print("\n[5/6] Defining models...")
    
    # Spark ML models
    lr_model = LinearRegression(labelCol=target_column)
    rf_model = RandomForestRegressor(labelCol=target_column)
    gbt_model = GBTRegressor(labelCol=target_column)
    
    rf_param_grid = (
        ParamGridBuilder()
        .addGrid(rf_model.maxDepth, [3, 5])
        .addGrid(rf_model.numTrees, [10, 50])
        .build()
    )
    
    gbt_param_grid = (
        ParamGridBuilder()
        .addGrid(gbt_model.maxDepth, [3, 5])
        .addGrid(gbt_model.maxIter, [10])
        .build()
    )
    
    spark_ml_models = [
        {"alias": "LinearRegression", "model": lr_model, "param_grid": None},
        {"alias": "RandomForest", "model": rf_model, "param_grid": rf_param_grid},
        {"alias": "GradientBoostedTrees", "model": gbt_model, "param_grid": gbt_param_grid}
    ]
    
    # Statistical models
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
    
    # 6) Train models with MLflow tracking
    print("\n[6/6] Training models...")
    
    # Setup MLflow
    if AZURE_CONFIG_AVAILABLE and MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create MLflow experiment
    if user_email:
        experiment_base = f"/Users/{user_email}/{experiment_name}"
    elif AZURE_CONFIG_AVAILABLE:
        experiment_base = MLFLOW_EXPERIMENT_PATH.replace("{username}", "default")
    else:
        experiment_base = f"/Users/{experiment_name}"
    
    experiment_name_full = f"{experiment_base}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name_full)
    except mlflow.exceptions.MlflowException:
        exp = mlflow.get_experiment_by_name(experiment_name_full)
        experiment_id = exp.experiment_id if exp else None
    
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"[INFO] MLflow experiment: {experiment_name_full}")
    
    # Parent run
    with mlflow.start_run(run_name="M5_Databricks_All_Models") as parent_run:
        mlflow.log_param("dataset", "M5")
        mlflow.log_param("environment", "Azure Databricks")
        mlflow.log_param("train_count", train_count)
        mlflow.log_param("test_count", test_count)
        mlflow.log_param("num_features", len(available_features))
        mlflow.log_param("blob_path", blob_path or "default")
        
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
                        target_column=target_column
                    )
                    
                    evaluate_stats_models(
                        stats_model=trained_stats,
                        test_df=test_df,
                        month_end_column=month_end_column,
                        product_id_column=product_id_column,
                        target_column=target_column,
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
                        train_df=train_df.select(*available_features, target_column),
                        featuresCols=available_features,
                        labelCol=target_column,
                        paramGrid=param_grid
                    )
                    
                    metrics = evaluate_sparkML_models(
                        model=pipeline_model,
                        test_df=test_df.select(*available_features, target_column),
                        features_cols=available_features,
                        label_col=target_column,
                        requirements_path=None,
                        model_alias=ml_alias
                    )
                    print(f"[SparkML] {ml_alias} completed: {metrics}")
                except Exception as e:
                    print(f"[SparkML] Error training {ml_alias}: {e}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"View results in Databricks MLflow UI")
    print(f"Experiment: {experiment_name_full}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models on M5 dataset (Azure Databricks with Blob Storage)"
    )
    parser.add_argument(
        "--blob-path",
        type=str,
        default=None,
        help="Path to M5 sales_train_validation.csv in blob storage (wasbs:// format)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="M5_Forecasting_Databricks",
        help="Name for MLflow experiment"
    )
    parser.add_argument(
        "--limit-products",
        type=int,
        default=None,
        help="Limit number of products to load (for testing)"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=100,
        help="Maximum number of distinct items to train on"
    )
    parser.add_argument(
        "--user-email",
        type=str,
        default=None,
        help="Databricks user email for experiment path"
    )
    
    args = parser.parse_args()
    
    main_train_databricks(
        blob_path=args.blob_path,
        experiment_name=args.experiment_name,
        limit_products=args.limit_products,
        max_distinct_items=args.max_items,
        user_email=args.user_email
    )
