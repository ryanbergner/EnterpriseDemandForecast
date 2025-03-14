from pyspark.sql import SparkSession
from statsforecast import StatsForecast
from pyspark.sql.functions import (
    col,
    when,
    lag,
    lead,
    unix_timestamp,
    date_format,
    expr,
    to_date,
    abs as spark_abs,
    mean as spark_mean,
    lit
)
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType,
    DecimalType,
    IntegerType,
    DoubleType
)
from typing import Optional, List, Any
import numpy as np

from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflavors.statsforecast


def train_stats_models(
    models: List[Any],
    train_df: DataFrame,
    month_end_column: str,
    product_id_column: str,
    target_column: str
) -> StatsForecast:
    """
    Trains statistical forecasting models using the StatsForecast library.
    Expects a single StatsForecast instance with multiple sub-models,
    or a single stats model in `models` list.

    Steps:
      1) Convert Spark -> Pandas
      2) Initialize StatsForecast(models=..., freq='M')
      3) Fit the StatsForecast instance
    """
    print("=== train_stats_models: Converting train_df to Pandas for StatsForecast ===")
    df = (
        train_df
        .withColumn("ds", to_date(col(month_end_column)))
        .withColumn("y", col(target_column).cast("float"))
        .withColumn("unique_id", col(product_id_column))
        .select(["unique_id", "ds", "y"])
        .dropna()
        .toPandas()
    )
    print(f"StatsForecast training set => {len(df)} rows in Pandas")

    # Initialize StatsForecast
    sf = StatsForecast(models=models, freq='M')

    # Fit
    print(f"=== train_stats_models: Fitting StatsForecast with {len(models)} model(s) ===")
    sf.fit(df)
    print("StatsForecast fit complete.\n")

    return sf


def return_sf_model(
    model_name: str,
    statsforecast_instance: StatsForecast
) -> Optional[Any]:
    """
    Given a StatsForecast instance that may contain multiple sub-models,
    return the single sub-model whose 'alias' matches 'model_name'.
    """
    for sub_model in statsforecast_instance.models:
        if hasattr(sub_model, 'alias') and sub_model.alias == model_name:
            return sub_model
    return None


def evaluate_stats_models(
    stats_model,         # Single StatsForecast instance
    test_df: DataFrame,  # Spark DataFrame with [month_end_column, product_id_column, target_column]
    month_end_column: str,
    product_id_column: str,
    target_column: str,
    experiment_id: str,
    artifact_location: str,
    model_name: str
) -> None:
    """
    Evaluate a single StatsForecast model. Produces columns for forecasted values,
    merges with test data, and computes RMSE, R², MAE, MAPE in one pass.

    Args:
        stats_model: A trained StatsForecast instance.
        test_df (DataFrame): Test data with date + product_id + target columns.
        month_end_column (str): The date column name in test_df.
        product_id_column (str): The product identifier column name.
        target_column (str): The numeric target column name.
        experiment_id (str): MLflow experiment ID (not used in this snippet).
        artifact_location (str): Path or location for artifacts (not used in this snippet).
        model_name (str): Name or alias of the model, for logging/printing.

    Returns:
        None. Logs metrics to MLflow and prints status.
    """
    print(f"=== evaluate_stats_models: Starting evaluation for model '{model_name}' ===")

    # 1) Prepare test DataFrame => only date + product_id + target
    print("Step 1) Preparing test DataFrame ...")
    test_df = test_df.select(month_end_column, product_id_column, target_column).dropna()
    test_count = test_df.count()
    print(f"Test DF after dropna => {test_count} rows")

    # 2) Distinct date values => h
    print("Step 2) Determining forecast horizon from distinct test months ...")
    test_horizon = test_df.select(col(month_end_column)).distinct().count()
    print(f"Test horizon = {test_horizon} distinct time periods in test data")

    # 3) Use StatsForecast to predict => returns a Pandas DF with columns [ds, unique_id, forecast_cols...]
    print("Step 3) Generating StatsForecast predictions ...")
    predicted_quantities_pdf = stats_model.predict(h=test_horizon).reset_index(drop=True)
    print(f"Got predicted Pandas DF => shape={predicted_quantities_pdf.shape}")

    # Convert to Spark => rename columns to match test_df
    if "unique_id" not in predicted_quantities_pdf.columns:
        # Single-ID scenario => add dummy product_id in both predictions & test
        dummy_id_value = "SingleID"
        spark_pred = (
            spark.createDataFrame(predicted_quantities_pdf)
            .withColumnRenamed("ds", month_end_column)
        )
        test_df = test_df.withColumn(product_id_column, lit(dummy_id_value))
    else:
        # Normal multi-ID scenario
        spark_pred = (
            spark.createDataFrame(predicted_quantities_pdf)
            .withColumnRenamed("ds", month_end_column)
            .withColumnRenamed("unique_id", product_id_column)
        )

    print("Converted predictions to Spark DataFrame => columns:", spark_pred.columns)

    # The forecast columns => everything except ds/month_end, unique_id/product_id
    exclude_cols = {month_end_column, product_id_column}
    forecast_columns = [c for c in spark_pred.columns if c not in exclude_cols]

    # 4) Join predictions with test
    print(f"Step 4) Joining predictions with test set on {product_id_column}, {month_end_column} ...")
    joined_df = (
        test_df.join(spark_pred, on=[product_id_column, month_end_column], how="left")
                .dropna(subset=forecast_columns)
    )
    joined_count = joined_df.count()
    print(f"Predictions + test => {joined_count} rows after dropping null in forecast columns\n")

    # Evaluate each forecast column => single pass for RMSE, R², MAE, MAPE
    for fc_col in forecast_columns:
        print(f"Evaluating performance of {model_name} => forecast col: '{fc_col}'")

        # Single pass => compute sums for RMSE, R², MAE, MAPE
        from pyspark.sql.functions import pow as spark_pow, abs as spark_abs

        mean_y = joined_df.agg(spark_mean(col(target_column)).alias("mean_y")).collect()[0]["mean_y"]

        aggregated = (
            joined_df.select(
                (col(fc_col) - col(target_column)).alias("err"),
                spark_pow(col(fc_col) - col(target_column), lit(2.0)).alias("sq_err"),
                (col(target_column) - lit(mean_y)).alias("y_minus_mean"),
                spark_abs(col(fc_col) - col(target_column)).alias("abs_err"),
                when(
                    col(target_column) != 0,
                    spark_abs(col(fc_col) - col(target_column)) / spark_abs(col(target_column))
                ).alias("pct_err")
            )
            .agg(
                spark_mean(col("sq_err")).alias("mse"),
                spark_mean(spark_pow(col("y_minus_mean"), lit(2.0))).alias("var_y"),
                spark_mean(col("abs_err")).alias("mae"),
                spark_mean(col("pct_err")).alias("mape_fraction")
            )
            .collect()[0]
        )

        mse_val = aggregated["mse"]
        var_y   = aggregated["var_y"]
        mae_val = aggregated["mae"]
        mape_fr = aggregated["mape_fraction"]
        if mape_fr is not None:
            mape_val = mape_fr * 100.0
        else:
            mape_val = None

        rmse_val = np.sqrt(mse_val) if mse_val >= 0 else None
        # r2 => 1 - MSE / var_y
        r2_val = 1.0 - (mse_val / var_y) if var_y != 0 else None

        # Log metrics in MLflow
        if rmse_val is not None:
            mlflow.log_metric("rmse", rmse_val)
        if r2_val is not None:
            mlflow.log_metric("r2", r2_val)
        if mae_val is not None:
            mlflow.log_metric("mae", mae_val)
        if mape_val is not None:
            mlflow.log_metric("mape", mape_val)

        print(
            f"  => rmse={rmse_val if rmse_val else None}, "
            f"r2={r2_val if r2_val else None}, mae={mae_val}, "
            f"mape={mape_val if mape_val else 0}%"
        )

    print(f"=== Finished evaluate_stats_models for '{model_name}' ===\n")
