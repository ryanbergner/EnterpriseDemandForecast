#!/usr/bin/env python
# main_train_m5.py

import mlflow
import mlflow.sklearn
import mlflow.pyspark.ml

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    expr,
    explode,
    row_number,
    monotonically_increasing_id,
    lit,
    when
)
from pyspark.sql.window import Window
from pyspark.ml.regression import (
    GBTRegressor,
    RandomForestRegressor,
    LinearRegression
)
from pyspark.ml.tuning import ParamGridBuilder
from statsforecast.models import SeasonalExponentialSmoothingOptimized, CrostonClassic

# Local imports: aggregator & feature functions
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

def main_train_m5():
    """
    A 'dummy' training script that reads the M5 dataset, performs minimal
    transformations, and runs Spark ML & StatsForecast models. It artificially
    creates a 'product_category' column and loops through the usual categories
    ('Smooth', 'Erratic', 'Intermittent', 'Lumpy') to demonstrate the pipeline.
    """

    # 1) Create Spark session
    spark = SparkSession.builder.appName("M5_TrainingDummy").getOrCreate()

    # 2) Read M5 data (CSV). Adjust path as needed.
    #    Suppose the file has columns: 'item_id', 'd_1', 'd_2', ... 'd_1913', etc.
    m5_sdf = (
        spark.read
        .format("csv")
        .option("header", True)
        .load("/mnt/m5/sales_train_validation.csv")
    )

    # For a "dummy" transformation, let's unpivot day columns into (item_id, day, Quantity)
    # Adjust day_columns list based on actual M5 shape.
    day_columns = [f"d_{i}" for i in range(1, 1914)]

    # Unpivot
    window_spec = Window.partitionBy("item_id").orderBy(monotonically_increasing_id())
    m5_sdf = (
        m5_sdf
        .select(
            col("item_id"),
            explode(expr(f"array({','.join(day_columns)})")).alias("Quantity")
        )
        .withColumn("day", row_number().over(window_spec))
    )

    # 3) Create a date column => e.g. "OrderDate" starting from 2015-01-01 plus 'day' offset
    #    Adjust base date as appropriate.
    m5_sdf = m5_sdf.withColumn(
        "OrderDate",
        expr("date_add('2015-01-01', day - 1)")
    )

    # 4) For aggregator, rename columns to match the expected arguments
    #    i.e. date_column='OrderDate', product_id_column='item_id', quantity_column='Quantity'
    #    We'll define `month_end_col` = "MonthEndDate"
    date_col = "OrderDate"
    product_id_col = "item_id"
    quantity_col = "Quantity"
    month_end_col = "MonthEndDate"

    # 5) "Aggregate + add_features" from our existing pipeline
    df_agg = aggregate_sales_data(
        m5_sdf,
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

    # 6) For demonstration, we artificially create a 'product_category' column
    #    that randomly assigns one of the four categories. In real usage, you'd
    #    have your own category logic or skip categorization altogether.
    categories = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    df_feat = df_feat.withColumn(
        "product_category",
        when(col(product_id_col).substr(-1, 1).isin(["0", "1"]), lit("Smooth"))
        .when(col(product_id_col).substr(-1, 1).isin(["2", "3"]), lit("Erratic"))
        .when(col(product_id_col).substr(-1, 1).isin(["4", "5"]), lit("Intermittent"))
        .otherwise(lit("Lumpy"))
    )

    # 7) Define Spark/Stats models as usual
    target_col = "lead_month_1"
    lr_model = LinearRegression(labelCol=target_col)
    rf_model = RandomForestRegressor(labelCol=target_col)
    gbt_model = GBTRegressor(labelCol=target_col)

    # Example param grids
    rf_param_grid = (
        ParamGridBuilder()
        .addGrid(rf_model.maxDepth, [2, 5])
        .addGrid(rf_model.numTrees, [10])
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
        "Quantity",
        "months_since_last_order",
        "last_order_quantity",
        "month",
        "year"
    ]

    # 8) Loop over categories => train
    for cat in categories:
        print(f"\n=== [M5 Dummy] Training on category: {cat} ===")
        df_cat = df_feat.filter(col("product_category") == cat).dropna()

        if df_cat.count() == 0:
            print(f"No data for category '{cat}'. Skipping...")
            continue

        # Create or get an MLflow experiment
        experiment_name = f"/Users/you@domain.com/M5Dummy_{cat}"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            existing_exp = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = existing_exp.experiment_id if existing_exp else None

        mlflow.set_experiment(experiment_id=experiment_id)

        # Parent run for this category
        with mlflow.start_run(run_name=f"M5_{cat}_training") as parent_run:
            # A) StatsForecast
            for s_info in stats_models:
                s_alias = s_info["alias"]
                s_model = s_info["model"]

                with mlflow.start_run(run_name=s_alias, nested=True):
                    print(f"[StatsForecast] => {s_alias} on '{cat}'")
                    trained_stats = train_stats_models(
                        models=[s_model],
                        train_df=df_cat,
                        month_end_column=month_end_col,
                        product_id_column=product_id_col,
                        target_column=target_col
                    )
                    evaluate_stats_models(
                        stats_model=trained_stats,
                        test_df=df_cat,  # dummy: same data for "test"
                        month_end_column=month_end_col,
                        product_id_column=product_id_col,
                        target_column=target_col,
                        experiment_id=experiment_id,
                        artifact_location=None,
                        model_name=s_alias
                    )

            # B) Spark ML
            for ml_info in spark_models:
                ml_alias = ml_info["alias"]
                ml_model = ml_info["model"]
                param_grid = ml_info["param_grid"]

                with mlflow.start_run(run_name=ml_alias, nested=True):
                    print(f"[Spark ML] => {ml_alias} on '{cat}'")

                    pipeline_model = train_sparkML_models(
                        model=ml_model,
                        train_df=df_cat.select(*spark_feature_cols, target_col),
                        featuresCols=spark_feature_cols,
                        labelCol=target_col,
                        paramGrid=param_grid
                    )
                    metrics = evaluate_sparkML_models(
                        model=pipeline_model,
                        test_df=df_cat.select(*spark_feature_cols, target_col),
                        features_cols=spark_feature_cols,
                        label_col=target_col,
                        requirements_path=None,
                        model_alias=ml_alias
                    )
                    print(f"[{ml_alias}] => {metrics}")

    spark.stop()


if __name__ == "__main__":
    main_train_m5()
