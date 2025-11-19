#!/usr/bin/env python
# main_train.py
# Comprehensive training script that includes all model types:
# - Spark ML models (Linear Regression, Random Forest, GBT)
# - Statistical models (StatsForecast)
# - TabPFN (scikit-learn based)

import mlflow
import mlflow.sklearn
import mlflow.pyspark.ml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp
from statsforecast.models import SeasonalExponentialSmoothingOptimized, CrostonClassic
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.tuning import ParamGridBuilder

# TabPFN (optional - install if needed)
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: TabPFN not available. Install with: pip install tabpfn")

# Local project imports
from src.preprocessing.preprocess import aggregate_sales_data
from src.feature_engineering.feature_engineering import add_features
from src.model_training.ml_models import (
    train_sparkML_models,
    evaluate_sparkML_models
)
from src.model_training.stats_models import (
    train_stats_models,
    evaluate_stats_models
)

def main_train(
    data_path: str = None,
    date_col: str = "OrderDate",
    product_id_col: str = "item_id",
    quantity_col: str = "Quantity",
    month_end_col: str = "MonthEndDate",
    experiment_prefix: str = "/M5_Forecasting"
):
    """
    Main function to train multiple models (Spark ML + StatsForecast + TabPFN) 
    on product categories ("Smooth", "Erratic", "Intermittent", "Lumpy").
    
    Args:
        data_path: Path to input data CSV file. If None, expects data to be loaded separately.
        date_col: Name of date column
        product_id_col: Name of product ID column
        quantity_col: Name of quantity column
        month_end_col: Name of month-end date column
        experiment_prefix: MLflow experiment name prefix
    """
    # -------------------------------------------------------------------------
    # 1) Create Spark session / read data
    # -------------------------------------------------------------------------
    spark = SparkSession.builder.appName("M5ForecastingTrain").getOrCreate()

    # Load data if path provided
    if data_path:
        df = (
            spark.read.format("csv")
            .option("header", True)
            .load(data_path)
        )
        
        # Convert to correct types (adjust based on your data schema)
        df = (
            df.withColumn(date_col, col(date_col).cast("date"))
              .withColumn(quantity_col, col(quantity_col).cast("double"))
        )
    else:
        raise ValueError("data_path must be provided")

    # -------------------------------------------------------------------------
    # 2) Data Aggregation + Feature Engineering
    # -------------------------------------------------------------------------
    df_agg = aggregate_sales_data(
        df,
        date_col,
        product_id_col,
        quantity_col,
        month_end_col
    )
    df_feat = add_features(
        df_agg,
        month_end_col,
        product_id_col,
        quantity_col
    )

    # Optionally filter out products with insufficient data
    df_feat = df_feat.filter(col("total_orders") >= 5)

    # Create train/test split based on date
    df_feat = df_feat.withColumn("unix_time", unix_timestamp(col(month_end_col)))
    df_feat = df_feat.filter(col("unix_time").isNotNull())
    
    cutoff_ts = df_feat.approxQuantile("unix_time", [0.8], 0)[0]
    train_df = df_feat.filter(col("unix_time") < cutoff_ts)
    test_df = df_feat.filter(col("unix_time") >= cutoff_ts)

    # -------------------------------------------------------------------------
    # 3) Define models
    # -------------------------------------------------------------------------
    target_col = "lead_month_1"
    
    # Spark ML models
    lr_model = LinearRegression(labelCol=target_col)
    rf_model = RandomForestRegressor(labelCol=target_col)
    gbt_model = GBTRegressor(labelCol=target_col)

    # Parameter grids
    rfParamGrid = (
        ParamGridBuilder()
        .addGrid(rf_model.maxDepth, [2, 5])
        .addGrid(rf_model.numTrees, [10, 50])
        .build()
    )
    gbtParamGrid = (
        ParamGridBuilder()
        .addGrid(gbt_model.maxDepth, [2, 3])
        .addGrid(gbt_model.maxIter, [10])
        .build()
    )

    spark_models = [
        {"alias": "LR_model", "model": lr_model, "param_grid": None},
        {"alias": "RF_model", "model": rf_model, "param_grid": rfParamGrid},
        {"alias": "GBT_model", "model": gbt_model, "param_grid": gbtParamGrid},
    ]

    # StatsForecast models
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

    # Feature columns for Spark ML and TabPFN
    spark_feature_cols = [
        quantity_col,
        "months_since_last_order",
        "last_order_quantity",
        "month",
        "year",
        "lag_1",
        "lag_2",
        "lag_3",
        "ma_4_month",
        "ma_8_month",
        "cov_quantity"
    ]
    
    tabpfn_feature_cols = spark_feature_cols.copy()

    # -------------------------------------------------------------------------
    # 4) Train models for each product category
    # -------------------------------------------------------------------------
    product_categories = ["Smooth", "Erratic", "Intermittent", "Lumpy"]

    for category in product_categories:
        print(f"\n=== Training on category: {category} ===")

        # Filter datasets by category
        train_cat = train_df.filter(col("product_category") == category).dropna()
        test_cat = test_df.filter(col("product_category") == category).dropna()

        if train_cat.count() < 1:
            print(f"No training data for category '{category}'. Skipping...\n")
            continue

        # Create experiment
        experiment_name = f"{experiment_prefix}_{category}"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            exp = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = exp.experiment_id if exp else None

        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run(run_name=f"AllModels_{category}") as parent_run:
            # (A) Train StatsForecast models
            for stats_info in stats_models:
                stats_alias = stats_info["alias"]
                stats_model = stats_info["model"]

                with mlflow.start_run(run_name=f"Stats_{stats_alias}", nested=True):
                    print(f"Training StatsForecast => {stats_alias}")
                    trained_stats = train_stats_models(
                        models=[stats_model],
                        train_df=train_cat,
                        month_end_column=month_end_col,
                        product_id_column=product_id_col,
                        target_column=target_col
                    )
                    evaluate_stats_models(
                        stats_model=trained_stats,
                        test_df=test_cat if test_cat.count() > 0 else train_cat,
                        month_end_column=month_end_col,
                        product_id_column=product_id_col,
                        target_column=target_col,
                        experiment_id=experiment_id,
                        artifact_location=None,
                        model_name=stats_alias
                    )

            # (B) Train Spark ML models
            for model_info in spark_models:
                ml_alias = model_info["alias"]
                ml_model = model_info["model"]
                param_grid = model_info["param_grid"]

                with mlflow.start_run(run_name=f"Spark_{ml_alias}", nested=True):
                    print(f"Training Spark ML => {ml_alias}")

                    pipeline_model = train_sparkML_models(
                        model=ml_model,
                        train_df=train_cat.select(*spark_feature_cols, target_col),
                        featuresCols=spark_feature_cols,
                        labelCol=target_col,
                        paramGrid=param_grid
                    )

                    metrics = evaluate_sparkML_models(
                        model=pipeline_model,
                        test_df=test_cat.select(*spark_feature_cols, target_col) if test_cat.count() > 0 else train_cat.select(*spark_feature_cols, target_col),
                        features_cols=spark_feature_cols,
                        label_col=target_col,
                        requirements_path=None,
                        model_alias=ml_alias
                    )
                    print(f"Metrics for {ml_alias}: {metrics}")

            # (C) Train TabPFN (if available)
            if TABPFN_AVAILABLE:
                with mlflow.start_run(run_name="TabPFN", nested=True):
                    print("Training TabPFN...")
                    
                    # Convert to Pandas (limit to 10k rows for TabPFN)
                    train_pd = train_cat.select(*tabpfn_feature_cols, target_col).toPandas().dropna()
                    test_pd = test_cat.select(*tabpfn_feature_cols, target_col).toPandas().dropna() if test_cat.count() > 0 else train_pd
                    
                    if len(train_pd) > 10000:
                        train_pd = train_pd.sample(n=10000, random_state=42)
                    
                    if len(train_pd) < 10:
                        print("Insufficient data for TabPFN. Skipping...")
                    else:
                        X_train = train_pd[tabpfn_feature_cols].values
                        y_train = train_pd[target_col].values
                        X_test = test_pd[tabpfn_feature_cols].values if len(test_pd) > 0 else X_train
                        y_test = test_pd[target_col].values if len(test_pd) > 0 else y_train

                        model = TabPFNRegressor(device="auto")
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                        mse = mean_squared_error(y_test, preds)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, preds)
                        r2 = r2_score(y_test, preds)

                        mlflow.log_metric("mse", mse)
                        mlflow.log_metric("rmse", rmse)
                        mlflow.log_metric("mae", mae)
                        mlflow.log_metric("r2", r2)

                        # Log model
                        signature = infer_signature(
                            train_pd[tabpfn_feature_cols].head(5),
                            train_pd[target_col].head(5)
                        )
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="TabPFN_model",
                            signature=signature,
                            input_example=train_pd[tabpfn_feature_cols].head(5)
                        )
                        print(f"TabPFN metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    spark.stop()


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    main_train(data_path=data_path)
