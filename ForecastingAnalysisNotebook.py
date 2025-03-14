# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at:
# MAGIC # https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload, run %autoreload 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## To do Ryan

# COMMAND ----------

# MAGIC %md
# MAGIC **1. Validate working code**
# MAGIC - Run code below and validate that it works on your side -> it should create a modeling experiment with multiple runs!
# MAGIC
# MAGIC **2. EDA and Feature Engineering -> present results in a Notebook on Thursday Jan 16th**
# MAGIC - Perform EDA on Gale Pacific dataset, and recommend additional features to be included
# MAGIC - Write code (in src.feature_engineering.feature_engineering) to add additional features based on step 1
# MAGIC
# MAGIC **3. Compare models**
# MAGIC - Go through some iterations to find best performing Demand Genius candidate models
# MAGIC - Build code to fit TABPFN models (https://github.com/PriorLabs/TabPFN/blob/main/examples/tabpfn_for_regression.py)
# MAGIC   -> ensure we have experiment tracking through mlflow here and go through some iterations to increase performance
# MAGIC - Compare model performance of TABPFN vs Demand Genius models

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC - Check window logic for MA_x_sales and test in notebook
# MAGIC - Create buckets for different products based on their average order quantities -> we will include this as a feature in the model
# MAGIC - Train DG models -> here, you will need to change code a bit to include categorical features
# MAGIC - Train TabPFN model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Existing code

# COMMAND ----------

# MAGIC %md
# MAGIC Below we read the dataframe for our comparison

# COMMAND ----------

from pyspark.sql.functions import col, to_date
from pyspark.sql.types import DoubleType

# Read the CSV file from the specified volume
df = (
    spark.read.format("csv")
    .option("header", True)
    .load("/Volumes/main/default/demo_data/gale_pacific_hist_sales (1).csv")
)

df = df.withColumn(
    "DemandDate",
    to_date(col("DemandDate"), "MM-d-yyyy HH:mm:ss")
).withColumn(
    "DemandQuantity",
    col("DemandQuantity").cast(DoubleType())
)

# Display the DataFrame
display(df)

# COMMAND ----------

# Define columns for data, product_id, quantity, and month_end
date_column = "DemandDate"
product_id_column = "ItemNumber"
quantity_column = "DemandQuantity"
month_end_column = "MonthEndDate"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demand Genius model training

# COMMAND ----------

# MAGIC %md
# MAGIC Below, we redefine the main function with a small tweak (not reading from a source path, but a CSV from Unity Catalog instead) and perform training.

# COMMAND ----------

import datetime
import warnings

warnings.filterwarnings("ignore")

import mlflow
import mlflavors
from mlflow.models.signature import infer_signature

from pyspark.sql.functions import (
    col,
    unix_timestamp
)
from pyspark.sql import DataFrame

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, HoltWinters, ARIMA, CrostonClassic
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler

# Import local project modules
from src.preprocessing.preprocess import aggregate_sales_data
from src.feature_engineering.feature_engineering import add_features
from src.model_training.ml_models import (
    train_sparkML_models as train_sparkML_model,
    evaluate_sparkML_models as evaluate_SparkML_model
)
from src.model_training.stats_models import (
    train_stats_models,
    evaluate_stats_models
)


def main_train(
    df: DataFrame,
    date_column: str,
    product_id_column: str,
    quantity_column: str,
    month_end_column: str,
    list_of_features: list,
    target_column: str,
    list_of_ml_models: list,
    list_of_stats_models: list
) -> None:
    """
    Main function to train machine learning and statistical models for different product categories.

    Args:
        df (DataFrame): Input Spark DataFrame.
        date_column (str): Name of the date column in the dataset.
        product_id_column (str): Name of the product identifier column in the dataset.
        quantity_column (str): Name of the quantity column in the dataset.
        month_end_column (str): Name of the column indicating the end of the month.
        list_of_features (List[str]): List of features to be used in Spark ML training.
        target_column (str): Name of the target column (e.g., lead_month_1).
        list_of_ml_models (List[Dict[str, object]]): List of dictionaries containing
            ML models and their parameter grids.
        list_of_stats_models (List[object]): List of statistical models (StatsForecast) to train.

    Returns:
        None
    """
    # Create mlflow experiment
    experiment_name = f"/Workspace/Users/jeroen.ruissen@mcaconnect.com/Gale_Pacific_full_run_{datetime.datetime.now()}"
    experiment_id = mlflow.create_experiment(experiment_name)

    # Set experiment as active
    exp = mlflow.set_experiment(experiment_id=experiment_id)
    artifact_location = exp.artifact_location

    # 1) Aggregate sales data
    df_agg = aggregate_sales_data(
        df,
        date_column,
        product_id_column,
        quantity_column,
        month_end_column
    )

    # 2) Add features
    df_feat = add_features(
        df_agg,
        month_end_column,
        product_id_column,
        quantity_column
    )

    # Filter out products that have fewer than 5 orders over the entire timeframe
    df_feat = df_feat.filter(col("total_orders") >= 5)

    # Calculate 80th percentile of dates to define cut-off date for train-test split
    df_with_unix = df_feat.withColumn("unix_timestamp", unix_timestamp(col(month_end_column)))
    percentile_timestamp = df_with_unix.approxQuantile("unix_timestamp", [0.8], 0)[0]
    percentile_date = datetime.datetime.fromtimestamp(percentile_timestamp).strftime("%Y-%m-%d")

    # Get distinct product categories
    product_categories = [
        row.product_category
        for row in df_feat.select("product_category").distinct().collect()
    ]

    # For each product category
    for category in product_categories:
        print(f"{datetime.datetime.now()} --------- Training models on {category} category ---------")
        with mlflow.start_run(experiment_id=experiment_id, run_name=category) as category_run:
            # Create training set
            train_df = df_feat.filter(
                (col("product_category") == category)
                & (col(month_end_column) < percentile_date)
            ).dropna()

            # Additional filter: total_orders >= 8
            train_df = train_df.filter(col("total_orders") >= 8)

            # Create test set => only products also found in train
            test_df = df_feat.filter(
                (col("product_category") == category)
                & (col(month_end_column) > percentile_date)
            ).dropna()

            test_df_filtered = test_df.join(
                train_df.select(product_id_column).distinct(),
                product_id_column,
                "inner"
            )

            # ----- Train stats models -----
            print("----- Training StatsForecast models -----")
            for model in list_of_stats_models:
                with mlflow.start_run(experiment_id=experiment_id, run_name=model[0].alias, nested=True):
                    print(f"------ Stats model name: {model} ------")
                    statsModels = train_stats_models(
                        model,
                        train_df,
                        month_end_column,
                        product_id_column,
                        target_column
                    )

                    input_example = (
                        train_df.select(month_end_column, product_id_column, target_column)
                        .limit(5).toPandas()
                    )

                    # Infer the signature
                    signature = infer_signature(
                        input_example.drop(columns=[target_column]),
                        statsModels.predict(h=1)
                    )

                    # Log the model with MLflow
                    mlflavors.statsforecast.log_model(
                        statsforecast_model=statsModels,
                        artifact_path=artifact_location,
                        serialization_format="pickle",
                        input_example=input_example,
                        signature=signature
                    )

                    # Evaluate stats models
                    print(f"{datetime.datetime.now()} Performance of Stats models:")
                    evaluate_stats_models(
                        statsModels,
                        test_df_filtered,
                        month_end_column,
                        product_id_column,
                        target_column,
                        experiment_id,
                        artifact_location,
                        model_name=model[0].alias
                    )

            # ----- Train and evaluate ML models -----
            for ml_model_dict in list_of_ml_models:
                with mlflow.start_run(experiment_id=experiment_id, run_name=ml_model_dict["alias"], nested=True):
                    mlflow.pyspark.ml.autolog()
                    print(f"[{datetime.datetime.now()}] Training {ml_model_dict['alias']} Model")
                    model_to_train = ml_model_dict["model"]
                    model_name = ml_model_dict["alias"]
                    param_grid = ml_model_dict["param_grid"]

                    pipeline_model = train_sparkML_model(
                        model_to_train,
                        train_df,
                        list_of_features,
                        target_column,
                        param_grid
                    )

                    mlflow.set_tag("Model Name", model_name)

                    print(f"[{datetime.datetime.now()}] Performance of {ml_model_dict['alias']} Model:")
                    evaluate_SparkML_model(
                        pipeline_model,
                        test_df_filtered,
                        list_of_features,
                        target_column
                    )

# COMMAND ----------

# Define parameters needed for main_train function

# 2. ML Model training variables
list_of_features = [
    "DemandQuantity",
    "months_since_last_order",
    "last_order_quantity",
    "month",
    "year"
]
target_column = "lead_month_1"

# 2.1 Define ML models to train
gbt = GBTRegressor(labelCol=target_column)
GBTParamGrid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [2, 3, 5])
    .addGrid(gbt.maxIter, [10])
    .build()
)
gbt_model = {
    "alias": "GBT_model",
    "model": gbt,
    "param_grid": GBTParamGrid
}

rf = RandomForestRegressor(labelCol=target_column)
rfParamGrid = (
    ParamGridBuilder()
    .addGrid(rf.maxDepth, [2, 3, 5])
    .addGrid(rf.numTrees, [50])
    .build()
)
rf_model = {
    "alias": "RF_model",
    "model": rf,
    "param_grid": rfParamGrid
}

lr_model = {
    "alias": "LR_model",
    "model": LinearRegression(labelCol=target_column),
    "param_grid": None
}

# 2.2 Wrap all models in a list of dictionaries
list_of_ml_models = [lr_model, rf_model, gbt_model]

# 3. Define Stats models training variables
from statsforecast.models import SeasonalExponentialSmoothingOptimized

list_of_stats_models = [
    [SeasonalExponentialSmoothingOptimized(season_length=12)],
    [CrostonClassic()]
]

# Call main_train
main_train(
    df,
    date_column,
    product_id_column,
    quantity_column,
    month_end_column,
    list_of_features,
    target_column,
    list_of_ml_models,
    list_of_stats_models
)
