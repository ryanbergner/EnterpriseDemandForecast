from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, expr, lit
from typing import Optional, List
from functools import reduce
import mlflow

from src.preprocessing.preprocess import aggregate_sales_data
from src.feature_engineering.feature_engineering import add_features
from src.inference.inference import generate_predictions
from src.inference.monthly_predictions import MonthlyPredictionGenerator

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
    ind_full_history: Optional[int] = 0,
    include_intervals: bool = False,
    confidence_level: float = 0.95,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "lead_month_1"
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

    # Default feature columns for interval estimation
    if feature_cols is None:
        feature_cols = [
            quantity_column,
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

        if include_intervals and ind_full_history != 1:
            champ_generator = MonthlyPredictionGenerator(
                model_uri=champ_model_uri,
                model_name=champ_model_alias,
                spark=df.sparkSession,
                confidence_level=confidence_level,
                date_col=month_end_column,
                product_id_col=product_id_column
            )
            champ_df = champ_generator.generate(
                df_inference=df_inference_filtered,
                sales_pattern=sales_pattern,
                include_intervals=True,
                feature_cols=feature_cols,
                target_col=target_col
            ).withColumn("is_champion", lit(1))
        else:
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

        if include_intervals and ind_full_history != 1:
            chall_generator = MonthlyPredictionGenerator(
                model_uri=chall_model_uri,
                model_name=chall_model_alias,
                spark=df.sparkSession,
                confidence_level=confidence_level,
                date_col=month_end_column,
                product_id_col=product_id_column
            )
            chall_df = chall_generator.generate(
                df_inference=df_inference_filtered,
                sales_pattern=sales_pattern,
                include_intervals=True,
                feature_cols=feature_cols,
                target_col=target_col
            ).withColumn("is_champion", lit(0))
        else:
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
    df_predictions = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs_predictions)

    # 5) Optionally write predictions to a target path.
    if target_path is not None:
        df_predictions.write.mode("overwrite").parquet(target_path)

    return df_predictions
