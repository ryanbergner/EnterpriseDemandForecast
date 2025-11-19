"""
Reusable training workflow for Spark ML and StatsForecast models on the M5 dataset.

Both the local and Databricks entrypoints import these helpers to avoid duplicated
logic and to ensure a unified set of defaults across environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mlflow
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.feature_engineering.feature_engineering import add_features
from src.model_training.ml_models import evaluate_sparkML_models, train_sparkML_models
from src.model_training.stats_models import evaluate_stats_models, train_stats_models
from src.preprocessing.preprocess import aggregate_sales_data


DEFAULT_FEATURE_CANDIDATES = [
    "Quantity",
    "months_since_last_order",
    "last_order_quantity",
    "cov_quantity",
    "avg_demand_interval",
    "ma_4_month",
    "ma_8_month",
    "ma_12_month",
    "ma_3_sales",
    "ma_5_sales",
    "ma_7_sales",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_5",
    "month",
    "year",
]


@dataclass
class TrainingConfig:
    experiment_name: str
    date_column: str
    product_id_column: str
    quantity_column: str
    month_end_column: str = "MonthEndDate"
    ml_target_column: str = "lead_month_1"
    stats_target_column: Optional[str] = None
    min_total_orders: int = 5
    sample_products: Optional[int] = None
    feature_columns: Optional[Sequence[str]] = None
    category_column: Optional[str] = None
    train_quantile: float = 0.8
    requirements_path: Optional[str] = "requirements.txt"
    enable_stats: bool = True
    enable_ml: bool = True

    def resolve_feature_columns(self, df: DataFrame) -> List[str]:
        if self.feature_columns:
            available = [c for c in self.feature_columns if c in df.columns]
        else:
            available = [c for c in DEFAULT_FEATURE_CANDIDATES if c in df.columns]
        if not available:
            raise ValueError(
                "No usable feature columns found. "
                "Check that feature engineering created numeric predictors."
            )
        return available

    @property
    def effective_stats_target(self) -> str:
        return self.stats_target_column or self.quantity_column


def default_spark_models(target_col: str) -> List[Dict]:
    lr_model = LinearRegression(labelCol=target_col)
    rf_model = RandomForestRegressor(labelCol=target_col)
    gbt_model = GBTRegressor(labelCol=target_col)

    rf_grid = (
        ParamGridBuilder()
        .addGrid(rf_model.maxDepth, [4, 8])
        .addGrid(rf_model.numTrees, [100])
        .build()
    )
    gbt_grid = (
        ParamGridBuilder()
        .addGrid(gbt_model.maxDepth, [3, 5])
        .addGrid(gbt_model.maxIter, [50])
        .build()
    )
    return [
        {"alias": "LinearRegression", "model": lr_model, "param_grid": None},
        {"alias": "RandomForest", "model": rf_model, "param_grid": rf_grid},
        {"alias": "GradientBoostedTrees", "model": gbt_model, "param_grid": gbt_grid},
    ]


def default_stats_models() -> List[Dict]:
    from statsforecast.models import CrostonClassic, SeasonalExponentialSmoothingOptimized

    return [
        {
            "alias": "SeasonalExponentialSmoothingOptimized",
            "models": [SeasonalExponentialSmoothingOptimized(season_length=12)],
        },
        {"alias": "CrostonClassic", "models": [CrostonClassic()]},
    ]


def prepare_feature_frame(df: DataFrame, cfg: TrainingConfig) -> DataFrame:
    df_agg = aggregate_sales_data(
        df,
        date_column=cfg.date_column,
        product_id_column=cfg.product_id_column,
        quantity_column=cfg.quantity_column,
        month_end_column=cfg.month_end_column,
    )
    df_feat = add_features(
        df_agg,
        month_end_column=cfg.month_end_column,
        product_id_column=cfg.product_id_column,
        quantity_column=cfg.quantity_column,
    )
    if cfg.min_total_orders:
        df_feat = df_feat.filter(F.col("total_orders") >= cfg.min_total_orders)

    if cfg.sample_products:
        # Sample a subset of products deterministically via window row_number
        product_ids = (
            df_feat.select(cfg.product_id_column)
            .distinct()
            .orderBy(cfg.product_id_column)
            .limit(cfg.sample_products)
        )
        df_feat = df_feat.join(product_ids, cfg.product_id_column, "inner")

    return df_feat


def split_train_test(df: DataFrame, cfg: TrainingConfig) -> Tuple[DataFrame, DataFrame, Optional[str]]:
    df_ts = df.withColumn("unix_time", F.unix_timestamp(F.col(cfg.month_end_column)))
    quantiles = df_ts.approxQuantile("unix_time", [cfg.train_quantile], 0.0)
    cutoff = quantiles[0] if quantiles else None
    if cutoff is None:
        df_clean = df_ts.drop("unix_time")
        return df_clean, df_clean.limit(0), None

    train_df = df_ts.filter(F.col("unix_time") <= cutoff).drop("unix_time")
    test_df = df_ts.filter(F.col("unix_time") > cutoff).drop("unix_time")

    cutoff_iso = (
        train_df.agg(F.max(F.col(cfg.month_end_column)).alias("cutoff")).collect()[0]["cutoff"]
        if cutoff
        else None
    )
    return train_df, test_df, cutoff_iso


def _clean_for_ml(df: DataFrame, features: Sequence[str], label: str) -> DataFrame:
    cols = list({*features, label})
    return df.select(*cols).dropna(subset=cols)


def run_training_workflow(
    df_feat: DataFrame,
    cfg: TrainingConfig,
    spark_models: Optional[List[Dict]] = None,
    stats_models: Optional[List[Dict]] = None,
    segment_label: str = "global",
) -> None:
    feature_cols = cfg.resolve_feature_columns(df_feat)
    train_df, test_df, cutoff = split_train_test(df_feat, cfg)

    if train_df.count() == 0 or test_df.count() == 0:
        raise RuntimeError(
            "Train/test split produced empty partitions. "
            "Ensure the dataset spans enough time or adjust the train_quantile."
        )

    spark_models = spark_models if spark_models is not None else default_spark_models(cfg.ml_target_column)
    stats_models = stats_models if stats_models is not None else default_stats_models()

    ml_ready_train = _clean_for_ml(train_df, feature_cols, cfg.ml_target_column)
    ml_ready_test = _clean_for_ml(test_df, feature_cols, cfg.ml_target_column)

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=f"training::{segment_label}") as parent_run:
        if cutoff:
            mlflow.log_param("train_cutoff", cutoff.isoformat())
        mlflow.log_param("segment_label", segment_label)
        mlflow.log_param("feature_count", len(feature_cols))

        if cfg.enable_stats:
            stats_target = cfg.effective_stats_target
            stats_train = train_df.select(cfg.product_id_column, cfg.month_end_column, stats_target).dropna()
            stats_test = test_df.select(cfg.product_id_column, cfg.month_end_column, stats_target).dropna()
            for stats_info in stats_models:
                alias = stats_info["alias"]
                models = stats_info["models"]
                if stats_train.count() == 0 or stats_test.count() == 0:
                    print(f"[Stats::{alias}] Skipping – insufficient non-null rows.")
                    continue
                with mlflow.start_run(run_name=f"stats::{alias}", nested=True):
                    sf_model = train_stats_models(
                        models=models,
                        train_df=stats_train,
                        month_end_column=cfg.month_end_column,
                        product_id_column=cfg.product_id_column,
                        target_column=stats_target,
                    )
                    evaluate_stats_models(
                        stats_model=sf_model,
                        test_df=stats_test,
                        month_end_column=cfg.month_end_column,
                        product_id_column=cfg.product_id_column,
                        target_column=stats_target,
                        experiment_id=mlflow.active_run().info.experiment_id,
                        artifact_location=mlflow.get_artifact_uri(),
                        model_name=alias,
                    )

        if cfg.enable_ml:
            for ml_info in spark_models:
                alias = ml_info["alias"]
                model = ml_info["model"]
                param_grid = ml_info.get("param_grid")
                with mlflow.start_run(run_name=f"spark::{alias}", nested=True):
                    pipeline_model = train_sparkML_models(
                        model=model,
                        train_df=ml_ready_train,
                        featuresCols=feature_cols,
                        labelCol=cfg.ml_target_column,
                        paramGrid=param_grid,
                    )
                    evaluate_sparkML_models(
                        model=pipeline_model,
                        test_df=ml_ready_test,
                        features_cols=feature_cols,
                        label_col=cfg.ml_target_column,
                        requirements_path=cfg.requirements_path,
                        model_alias=alias,
                    )
