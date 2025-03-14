from pyspark.sql import SparkSession
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, HoltWinters, ARIMA, CrostonClassic
import pyspark.sql.functions as F
from pyspark.sql.functions import (
    count,
    lag,
    last_day,
    expr,
    explode,
    sequence,
    date_format,
    col,
    last,
    lead,
    when,
    months_between,
    month,
    year,
    avg,
    to_timestamp,
    to_date,
    mean,
    lit,
    max as spark_max,
    unix_timestamp
)
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DateType
from typing import Optional, List, Any, Dict
from math import sqrt as math_sqrt
from dateutil.relativedelta import relativedelta
from functools import reduce
import pandas as pd
import mlflavors as mlflavors
from datetime import datetime, timedelta
import mlflow

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline, PipelineModel

from src.preprocessing.preprocess import aggregate_sales_data, retrieve_sales_data
from src.feature_engineering.feature_engineering import add_features
from src.model_training.ml_models import train_sparkML_model, evaluate_SparkML_model
from src.model_training.stats_models import train_stats_models, evaluate_stats_models
from src.inference.inference import generate_predictions, get_model_by_tag

mlflow.set_registry_uri("databricks-uc")
client = mlflow.tracking.MlflowClient()
catalog = "main"
schema = "default"
alias = "champion"


def main_inference(
    df: DataFrame,
    date_column: str,
    product_id_column: str,
    quantity_column: str,
    month_end_column: str,
    target_path: Optional[str] = None,
    ind_full_history: Optional[int] = 0
) -> DataFrame:
    """
    Generates predictions for all products with more than 5 months of historical
    sales. Generates these predictions with a model per sales pattern. Can generate
    either predictions for the next month only or for the entire history, and
    optionally writes the predictions to a target path.

    Args:
        df (DataFrame): The Spark DataFrame containing the raw sales data.
        date_column (str): The name of the column containing the order dates.
        product_id_column (str): The name of the column containing the product IDs.
        quantity_column (str): The name of the column containing the quantities sold.
        month_end_column (str): The name of the column containing the month end dates.
        target_path (str, optional): The target path to write the predictions.
            If None, predictions are not written.
        ind_full_history (int, optional): By default, the function only generates
            predictions for the next month (ind_full_history=0). It generates
            predictions for the entire history if set to 1.

    Returns:
        DataFrame: A Spark DataFrame containing champion and challenger predictions
        for each applicable product.
    """
    # 1) Aggregate the data to monthly level and add feature-engineered columns.
    df_agg = aggregate_sales_data(df, date_column, product_id_column, quantity_column, month_end_column)
    df_feat = add_features(df_agg, month_end_column, product_id_column, quantity_column)

    # 2) If ind_full_history != 1, only generate predictions for the final month;
    #    otherwise, generate for the entire history.
    if ind_full_history != 1:
        max_month_end_date = df_feat.agg(expr(f"max({month_end_column})")).collect()[0][0]
        df_inference = df_feat.filter(
            (col(month_end_column) == lit(max_month_end_date)) &
            (col("total_orders") > 5)
        )
    else:
        df_inference = df_feat.filter(col("total_orders") > 5)

    # 3) Generate predictions for each distinct product category.
    dfs_predictions = []
    sales_patterns = [
        row.product_category
        for row in df_inference.select("product_category").distinct().collect()
    ]

    for sales_pattern in sales_patterns:
        print(f"---- Generating predictions for {sales_pattern} category ----")
        df_inference_filtered = df_inference.filter(df_inference["product_category"] == sales_pattern)

        # Champion predictions
        model_name = f"{catalog}.{schema}.{sales_pattern}"
        champ_version_info = client.get_model_version_by_alias(name=model_name, alias="champion")
        champ_model_uri = f"models:/{model_name}/{champ_version_info.version}"

        champ_run_id = champ_version_info.run_id
        champ_run_details = client.get_run(champ_run_id)
        champ_model_alias = champ_run_details.data.tags["mlflow.runName"]

        champ_df = generate_predictions(
            champ_model_uri,
            champ_model_alias,
            sales_pattern,
            df_inference_filtered,
            month_end_column,
            product_id_column
        ).withColumn("is_champion", lit(1))
        dfs_predictions.append(champ_df)

        # Challenger predictions
        chall_version_info = client.get_model_version_by_alias(name=model_name, alias="challenger")
        chall_model_uri = f"models:/{model_name}/{chall_version_info.version}"

        chall_run_id = chall_version_info.run_id
        chall_run_details = client.get_run(chall_run_id)
        chall_model_alias = chall_run_details.data.tags["mlflow.runName"]

        chall_df = generate_predictions(
            chall_model_uri,
            chall_model_alias,
            sales_pattern,
            df_inference_filtered,
            month_end_column,
            product_id_column
        ).withColumn("is_champion", lit(0))
        dfs_predictions.append(chall_df)

    # 4) Union all champion and challenger predictions into a single DataFrame.
    df_predictions = reduce(DataFrame.unionAll, dfs_predictions)

    # 5) Optionally write predictions to a target path.
    if target_path is not None:
        df_predictions.write.mode("overwrite").parquet(target_path)

    return df_predictions
