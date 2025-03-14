#!/usr/bin/env python
# main_train_sklearn.py

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TabPFN
from tabpfn import TabPFNRegressor

# Local imports
from src.preprocessing.preprocess import aggregate_sales_data
from src.feature_engineering.feature_engineering import add_features

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def main_train_sklearn():
    """
    Main function to train TabPFN (or other sklearn-based models) on each of the
    four product categories: Smooth, Erratic, Intermittent, Lumpy.

    Enforces a 10k row limit for TabPFN. Each category => separate MLflow experiment.
    """
    # 1) Create Spark session to load/transform data
    spark = SparkSession.builder.appName("TrainTabPFN").getOrCreate()

    # 2) Load data
    df = (
        spark.read
        .format("csv")
        .option("header", True)
        .load("/Volumes/main/default/demo_data/gale_pacific_hist_sales (1).csv")
    )

    df = df.withColumn("DemandDate", col("DemandDate").cast("date")) \
           .withColumn("DemandQuantity", col("DemandQuantity").cast("double"))

    date_col = "DemandDate"
    product_id_col = "ItemNumber"
    quantity_col = "DemandQuantity"
    month_end_col = "MonthEndDate"
    target_col = "lead_month_1"

    # 3) Aggregate + Features
    df_agg = aggregate_sales_data(
        df, date_col, product_id_col, quantity_col, month_end_col
    )
    df_feat = add_features(df_agg, month_end_col, product_id_col, quantity_col)

    # Filter if needed
    df_feat = df_feat.filter(col("total_orders") >= 5)

    # Because TabPFN must run on a Pandas DataFrame
    # => We'll do it category by category
    product_categories = ["Smooth", "Erratic", "Intermittent", "Lumpy"]

    # Example features
    tabpfn_feature_cols = [
        "DemandQuantity",
        "months_since_last_order",
        "last_order_quantity",
        "month",
        "year",
        "lag_1",
        "lag_2",
        "lag_3",
    ]

    for category in product_categories:
        print(f"\n=== [TabPFN] Training on category: {category} ===")

        df_cat = df_feat.filter(col("product_category") == category).dropna()

        row_count = df_cat.count()
        if row_count == 0:
            print(f"No data for category '{category}'. Skipping...\n")
            continue

        # Convert to Pandas
        df_cat_pd = df_cat.select(*tabpfn_feature_cols, target_col).toPandas()
        df_cat_pd.dropna(inplace=True)
        row_count = len(df_cat_pd)
        if row_count == 0:
            print(f"No valid rows after dropna for category '{category}'. Skipping...\n")
            continue

        # Enforce 10k row limit for TabPFN
        if row_count > 10000:
            df_cat_pd = df_cat_pd.sample(n=10000, random_state=42)
            row_count = len(df_cat_pd)
            print(f"Category '{category}' => reduced to {row_count} rows for TabPFN.")

        # 4) Prepare train/test
        X = df_cat_pd[tabpfn_feature_cols].values
        y = df_cat_pd[target_col].values

        if len(X) < 10:
            print(f"Too few rows for category '{category}'. Skipping...\n")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create or get experiment
        experiment_name = f"/Users/you@domain.com/TabPFN_{category}"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            existing_exp = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = existing_exp.experiment_id if existing_exp else None

        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run(run_name=f"TabPFN_{category}"):
            print(f"Training TabPFN on '{category}' with {len(X_train)} train rows / {len(X_test)} test rows.")

            # 5) Train TabPFN
            model = TabPFNRegressor(device="auto")
            model.fit(X_train, y_train)

            # Evaluate
            preds = model.predict(X_test)

            mse_val = mean_squared_error(y_test, preds)
            rmse_val = mse_val**0.5
            mae_val  = mean_absolute_error(y_test, preds)
            r2_val   = r2_score(y_test, preds)

            mlflow.log_metric("mse",  mse_val)
            mlflow.log_metric("rmse", rmse_val)
            mlflow.log_metric("mae",  mae_val)
            mlflow.log_metric("r2",   r2_val)

            print(
                f"[TabPFN_{category}] => mse={mse_val:.3f}, rmse={rmse_val:.3f}, "
                f"mae={mae_val:.3f}, r2={r2_val:.3f}"
            )

            # Log model to MLflow
            # Provide small input_example from X_train
            example_features = df_cat_pd[tabpfn_feature_cols].head(5)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="TabPFN_model",
                input_example=example_features
            )

    spark.stop()


if __name__ == "__main__":
    main_train_sklearn()
