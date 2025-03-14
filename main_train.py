#!/usr/bin/env python
# main_train.py

import mlflow
import mlflow.sklearn
import mlflow.pyspark.ml

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp
from statsforecast.models import SeasonalExponentialSmoothingOptimized, CrostonClassic
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# Local project imports (assuming your repo structure)
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

def main_train():
    """
    Main function to train multiple models (Spark ML + StatsForecast) on four
    product categories ("Smooth", "Erratic", "Intermittent", "Lumpy").
    Each model–category pair is stored in a separate MLflow experiment.
    """
    # -------------------------------------------------------------------------
    # 1) Create Spark session / read data
    # -------------------------------------------------------------------------
    spark = SparkSession.builder.appName("GalePacificTrain").getOrCreate()

    # Example: Load CSV. Adjust path or read from Delta table if needed
    df = (
        spark.read.format("csv")
        .option("header", True)
        .load("/Volumes/main/default/demo_data/gale_pacific_hist_sales (1).csv")
    )

    # Convert to correct types
    df = (
        df.withColumn("DemandDate", col("DemandDate").cast("date"))
          .withColumn("DemandQuantity", col("DemandQuantity").cast("double"))
    )

    date_col = "DemandDate"
    product_id_col = "ItemNumber"
    quantity_col = "DemandQuantity"
    month_end_col = "MonthEndDate"

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

    # -------------------------------------------------------------------------
    # 3) Define models
    # -------------------------------------------------------------------------
    # Spark ML models
    target_col = "lead_month_1"  # Usually set in add_features logic
    lr_model = LinearRegression(labelCol=target_col)
    rf_model = RandomForestRegressor(labelCol=target_col)
    gbt_model = GBTRegressor(labelCol=target_col)

    # Example param grids (optional)
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
        {
            "alias": "LR_model",
            "model": lr_model,
            "param_grid": None
        },
        {
            "alias": "RF_model",
            "model": rf_model,
            "param_grid": rfParamGrid
        },
        {
            "alias": "GBT_model",
            "model": gbt_model,
            "param_grid": gbtParamGrid
        },
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

    # Example feature columns for Spark ML
    spark_feature_cols = [
        "DemandQuantity",
        "months_since_last_order",
        "last_order_quantity",
        "month",
        "year"
        # add more columns if needed
    ]

    # -------------------------------------------------------------------------
    # 4) For each product category => train each model in a separate experiment
    # -------------------------------------------------------------------------
    product_categories = ["Smooth", "Erratic", "Intermittent", "Lumpy"]

    for category in product_categories:
        print(f"\n=== Training on category: {category} ===")

        # Filter the dataset to the specific category
        df_train = df_feat.filter(col("product_category") == category).dropna()

        # If no data, skip
        if df_train.count() < 1:
            print(f"No data available for category '{category}'. Skipping...\n")
            continue

        # Create a unique experiment name for each category
        experiment_name = f"/Users/your.name@company.com/GalePacific_{category}"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # If the experiment already exists
            exp = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = exp.experiment_id if exp else None

        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run(run_name=f"Train_{category}") as run:
            # (A) Train StatsForecast models
            for stats_info in stats_models:
                stats_alias = stats_info["alias"]
                stats_model = stats_info["model"]

                with mlflow.start_run(run_name=stats_alias, nested=True):
                    print(f"Training StatsForecast model => {stats_alias}")
                    trained_stats = train_stats_models(
                        models=[stats_model],
                        train_df=df_train,
                        month_end_column=month_end_col,
                        product_id_column=product_id_col,
                        target_column=target_col
                    )
                    # Evaluate stats
                    evaluate_stats_models(
                        stats_model=trained_stats,
                        test_df=df_train,  # or a separate test set if available
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

                with mlflow.start_run(run_name=ml_alias, nested=True):
                    print(f"Training Spark ML => {ml_alias}")

                    # Train pipeline
                    pipeline_model = train_sparkML_models(
                        model=ml_model,
                        train_df=df_train.select(*spark_feature_cols, target_col),
                        featuresCols=spark_feature_cols,
                        labelCol=target_col,
                        paramGrid=param_grid
                    )

                    # Evaluate pipeline
                    metrics = evaluate_sparkML_models(
                        model=pipeline_model,
                        test_df=df_train.select(*spark_feature_cols, target_col),
                        features_cols=spark_feature_cols,
                        label_col=target_col,
                        requirements_path=None,  # or path to your requirements.txt
                        model_alias=ml_alias
                    )
                    print(f"Metrics for {ml_alias} on category '{category}': {metrics}")

    spark.stop()


if __name__ == "__main__":
    main_train()
