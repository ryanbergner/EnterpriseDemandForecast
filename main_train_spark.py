#!/usr/bin/env python
# main_train_spark.py

import mlflow
import mlflow.sklearn
import mlflow.pyspark.ml

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.tuning import ParamGridBuilder
from statsforecast.models import SeasonalExponentialSmoothingOptimized, CrostonClassic

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


def main_train_spark():
    """
    Main function to train Spark ML and StatsForecast models on each of the four
    product categories: Smooth, Erratic, Intermittent, Lumpy.

    Each model–category pair is recorded in MLflow as a separate experiment.
    No row limit is enforced.
    """
    # 1) Create Spark session
    spark = SparkSession.builder.appName("TrainSparkStatsModels").getOrCreate()

    # 2) Load data (CSV example; adjust if loading from Delta/Parquet)
    df = (
        spark.read
        .format("csv")
        .option("header", True)
        .load("/Volumes/main/default/demo_data/gale_pacific_hist_sales (1).csv")
    )

    # Convert necessary columns to correct types
    df = (
        df.withColumn("DemandDate", col("DemandDate").cast("date"))
          .withColumn("DemandQuantity", col("DemandQuantity").cast("double"))
    )

    # Column references
    date_col = "DemandDate"
    product_id_col = "ItemNumber"
    quantity_col = "DemandQuantity"
    month_end_col = "MonthEndDate"
    target_col = "lead_month_1"  # We assume add_features creates lead_month_1

    # 3) Aggregate + Feature Engineering
    df_agg = aggregate_sales_data(
        df, date_col, product_id_col, quantity_col, month_end_col
    )
    df_feat = add_features(df_agg, month_end_col, product_id_col, quantity_col)

    # Filter out products with insufficient data if desired
    df_feat = df_feat.filter(col("total_orders") >= 5)

    # 4) Define Spark ML models
    lr_model = LinearRegression(labelCol=target_col)
    rf_model = RandomForestRegressor(labelCol=target_col)
    gbt_model = GBTRegressor(labelCol=target_col)

    # Example param grids
    rf_param_grid = (
        ParamGridBuilder()
        .addGrid(rf_model.maxDepth, [2, 5])
        .addGrid(rf_model.numTrees, [10, 50])
        .build()
    )
    gbt_param_grid = (
        ParamGridBuilder()
        .addGrid(gbt_model.maxDepth, [2, 3])
        .addGrid(gbt_model.maxIter, [10])
        .build()
    )

    spark_models = [
        {"alias": "LR_model",  "model": lr_model, "param_grid": None},
        {"alias": "RF_model",  "model": rf_model, "param_grid": rf_param_grid},
        {"alias": "GBT_model", "model": gbt_model, "param_grid": gbt_param_grid},
    ]

    # Define StatsForecast models
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

    # Example Spark feature columns
    spark_feature_cols = [
        "DemandQuantity",
        "months_since_last_order",
        "last_order_quantity",
        "month",
        "year"
    ]

    # 5) Loop over the 4 product categories => new experiment for each
    product_categories = ["Smooth", "Erratic", "Intermittent", "Lumpy"]

    for category in product_categories:
        print(f"\n=== [Spark+Stats] Training on category: {category} ===")

        df_train = df_feat.filter(col("product_category") == category).dropna()

        if df_train.count() == 0:
            print(f"No data for category '{category}'. Skipping...\n")
            continue

        # Create or get MLflow experiment for this category
        experiment_name = f"/Users/you@domain.com/Spark_{category}"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            existing_exp = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = existing_exp.experiment_id if existing_exp else None

        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run(run_name=f"Train_SparkStats_{category}") as parent_run:
            # A) StatsForecast models
            for s_info in stats_models:
                alias_s = s_info["alias"]
                s_model = s_info["model"]

                with mlflow.start_run(run_name=alias_s, nested=True):
                    print(f"[StatsForecast] Training => {alias_s}")
                    trained_stats = train_stats_models(
                        models=[s_model],
                        train_df=df_train,
                        month_end_column=month_end_col,
                        product_id_column=product_id_col,
                        target_column=target_col
                    )
                    evaluate_stats_models(
                        stats_model=trained_stats,
                        test_df=df_train,  # no separate test => same data or custom split
                        month_end_column=month_end_col,
                        product_id_column=product_id_col,
                        target_column=target_col,
                        experiment_id=experiment_id,
                        artifact_location=None,
                        model_name=alias_s
                    )

            # B) Spark ML models
            for model_info in spark_models:
                ml_alias = model_info["alias"]
                ml_model = model_info["model"]
                param_grid = model_info["param_grid"]

                with mlflow.start_run(run_name=ml_alias, nested=True):
                    print(f"[Spark ML] Training => {ml_alias}")

                    pipeline_model = train_sparkML_models(
                        model=ml_model,
                        train_df=df_train.select(*spark_feature_cols, target_col),
                        featuresCols=spark_feature_cols,
                        labelCol=target_col,
                        paramGrid=param_grid
                    )

                    # Evaluate
                    metrics = evaluate_sparkML_models(
                        model=pipeline_model,
                        test_df=df_train.select(*spark_feature_cols, target_col),
                        features_cols=spark_feature_cols,
                        label_col=target_col,
                        requirements_path=None,
                        model_alias=ml_alias
                    )
                    print(f"[{ml_alias}] => {metrics}")

    spark.stop()

if __name__ == "__main__":
    main_train_spark()
