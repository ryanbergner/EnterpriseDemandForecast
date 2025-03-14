from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import (
    RandomForestRegressor,
    GBTRegressor,
    LinearRegression
)
from pyspark.sql.functions import (
    col,
    abs as spark_abs,
    when,
    mean as spark_mean,
    pow as spark_pow,
    sqrt,
    collect_list,
    lit,
    last,
    lag
)
import mlflow
from typing import List, Optional
import numpy as np


def train_sparkML_models(
    model,
    train_df: DataFrame,
    featuresCols: list,
    labelCol: str,
    paramGrid=None
) -> PipelineModel:
    """
    A minimal Spark ML training function:
      - No one-hot encoding. We assume add_features() created numeric columns (month, bucket, etc).
      - VectorAssembler => 'features'
      - Possibly wrap in CrossValidator if paramGrid is provided
      - Fit pipeline => PipelineModel
    """
    print("=== Starting train_sparkML_model (minimal transformations) ===")
    if paramGrid is None:
        paramGrid = []

    # VectorAssembler => 'features'
    assembler = VectorAssembler(
        inputCols=featuresCols,
        outputCol="features",
        handleInvalid="skip"
    )

    # If the model supports label/features, set them
    if hasattr(model, "setLabelCol"):
        model.setLabelCol(labelCol)
    if hasattr(model, "setFeaturesCol"):
        model.setFeaturesCol("features")

    pipeline_stages = [assembler]

    # Possibly wrap in CrossValidator
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol=labelCol,
        predictionCol="prediction"
    )

    if paramGrid:
        print("Using CrossValidator with param grid.")
        from pyspark.ml.tuning import CrossValidator
        cv = CrossValidator(
            estimator=model,
            evaluator=evaluator,
            estimatorParamMaps=paramGrid
        )
        pipeline_stages.append(cv)
    else:
        print("No param grid provided => using model directly.")
        pipeline_stages.append(model)

    pipeline = Pipeline(stages=pipeline_stages)

    # Fit pipeline => PipelineModel
    pipelineModel = pipeline.fit(train_df)
    print("=== Finished train_sparkML_model ===")
    return pipelineModel


def evaluate_sparkML_models(
    model: PipelineModel,
    test_df: DataFrame,
    features_cols: List[str],
    label_col: str,
    requirements_path: Optional[str] = None,
    model_alias: Optional[str] = None
) -> dict:
    """
    Evaluate a Spark ML pipeline model on test_df, computing & logging:
      - RMSE, R², MAE, MAPE in one pass.

    Returns a dict of metrics in lower case.
    """
    if model_alias is None:
        model_alias = "MyModel"

    print(f"===== EVALUATION for {model_alias} =====")

    # 1) Select columns needed for transform
    needed_cols = set(features_cols + [label_col])
    test_df_selected = test_df.select(
        [c for c in needed_cols if c in test_df.columns]
    )

    print(f"Selected columns for prediction: {list(needed_cols)}")

    # 2) Predict
    preds = model.transform(test_df_selected).cache()

    # 3) Single pass for all metrics
    from pyspark.sql.functions import (
        mean as spark_mean,
        lit,
        when,
        abs as spark_abs,
        pow as spark_pow
    )

    # Compute mean_y
    mean_y = preds.agg(spark_mean(col(label_col)).alias("mean_y")).collect()[0]["mean_y"]
    if mean_y is None:
        print(f"WARNING: no data in test set for {model_alias}. Returning empty metrics.")
        return {}

    # aggregator
    row_agg = (
        preds.select(
            (col("prediction") - col(label_col)).alias("err"),
            spark_pow(col("prediction") - col(label_col), lit(2.0)).alias("sq_err"),
            spark_abs(col("prediction") - col(label_col)).alias("abs_err"),
            when(
                col(label_col) != 0,
                spark_abs(col("prediction") - col(label_col)) / spark_abs(col(label_col))
            ).alias("pct_err"),
            (col(label_col) - lit(mean_y)).alias("y_minus_mean")
        )
        .agg(
            spark_mean(col("sq_err")).alias("mse"),
            spark_mean(spark_pow(col("y_minus_mean"), lit(2.0))).alias("var_y"),
            spark_mean(col("abs_err")).alias("mae"),
            spark_mean(col("pct_err")).alias("mape_fraction")
        )
        .collect()[0]
    )

    mse_val = float(row_agg["mse"])
    var_y = float(row_agg["var_y"])
    mae_val = float(row_agg["mae"])
    mape_frac = row_agg["mape_fraction"]
    if mape_frac is not None:
        mape_val = float(mape_frac * 100.0)
    else:
        mape_val = None

    rmse_val = float(np.sqrt(mse_val)) if mse_val >= 0 else None
    r2_val = 1.0 - (mse_val / var_y) if var_y != 0 else None

    # 4) Log metrics in lower case
    import mlflow
    if rmse_val is not None:
        mlflow.log_metric("rmse", rmse_val)
    if r2_val is not None:
        mlflow.log_metric("r2", r2_val)
    mlflow.log_metric("mae", mae_val)
    if mape_val is not None:
        mlflow.log_metric("mape", mape_val)

    print(
        f"Evaluation => rmse={rmse_val if rmse_val else 0}, "
        f"r²={r2_val if r2_val else 0}, mae={mae_val}, "
        f"mape={mape_val if mape_val else 0}%\n"
    )

    # 5) Log model to MLflow if desired
    if requirements_path:
        artifact_path = f"{model_alias}_SparkModel"
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path=artifact_path,
            pip_requirements=requirements_path
        )

    return {
        "rmse": rmse_val,
        "r2":   r2_val,
        "mae":  mae_val,
        "mape": mape_val
    }
