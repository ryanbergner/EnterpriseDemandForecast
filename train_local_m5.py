#!/usr/bin/env python3
"""
Train forecasting models on the M5 dataset (or compatible schema) from a local environment.

The script unifies Spark ML and StatsForecast training into a single entry point and
replaces the previous collection of specialised training scripts.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import mlflow
from pyspark.sql import SparkSession, functions as F

from src.model_training.training_workflow import (
    TrainingConfig,
    prepare_feature_frame,
    run_training_workflow,
)
from src.preprocessing.datasets import (
    M5_DEFAULT_START_DATE,
    load_custom_daily_dataset,
    load_m5_wide_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified local trainer for the M5 dataset or M5-compatible schemas.",
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/m5/sales_train_validation.csv",
        help="Path to the dataset. Defaults to the Kaggle M5 wide CSV.",
    )
    parser.add_argument(
        "--format",
        choices=["m5_wide", "long"],
        default="m5_wide",
        help="Input format: 'm5_wide' for the Kaggle layout or 'long' for pre-unpivoted daily data.",
    )
    parser.add_argument(
        "--calendar-start",
        default=M5_DEFAULT_START_DATE,
        help="Calendar start date corresponding to d_1 when using --format=m5_wide.",
    )
    parser.add_argument(
        "--date-column",
        default="date",
        help="Date column name when --format=long.",
    )
    parser.add_argument(
        "--product-column",
        default="item_id",
        help="Product identifier column when --format=long.",
    )
    parser.add_argument(
        "--quantity-column",
        default="demand",
        help="Quantity column when --format=long.",
    )
    parser.add_argument(
        "--experiment-name",
        default="M5_Local_Training",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-uri",
        help="Optional MLflow tracking URI. Defaults to the local file store.",
    )
    parser.add_argument(
        "--sample-items",
        type=int,
        default=0,
        help="Optional cap on number of items to train on (0 = no cap).",
    )
    parser.add_argument(
        "--min-orders",
        type=int,
        default=5,
        help="Minimum number of non-zero order months per item.",
    )
    parser.add_argument(
        "--train-quantile",
        type=float,
        default=0.8,
        help="Quantile used for the chronological train/test split (range: 0-1).",
    )
    parser.add_argument(
        "--feature-columns",
        help="Comma-separated list of feature columns to use. Defaults to sensible heuristics.",
    )
    parser.add_argument(
        "--requirements-path",
        default="requirements.txt",
        help="Path to a pip requirements file for logging Spark models to MLflow.",
    )
    parser.add_argument(
        "--disable-stats",
        action="store_true",
        help="Disable StatsForecast models.",
    )
    parser.add_argument(
        "--disable-ml",
        action="store_true",
        help="Disable Spark ML models.",
    )
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="Train separate experiments per intermittent demand category.",
    )
    return parser.parse_args()


def create_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("M5LocalTrainer")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def parse_feature_columns(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [col.strip() for col in raw.split(",") if col.strip()]


def load_dataset(args: argparse.Namespace, spark: SparkSession):
    if args.format == "m5_wide":
        sdf = load_m5_wide_dataset(
            spark,
            path=args.data_path,
            sample_items=args.sample_items if args.sample_items > 0 else None,
            calendar_start=args.calendar_start,
        )
        date_column = "OrderDate"
        product_column = "item_id"
        quantity_column = "Quantity"
    else:
        sdf = load_custom_daily_dataset(
            spark,
            path=args.data_path,
            date_column=args.date_column,
            product_column=args.product_column,
            quantity_column=args.quantity_column,
        )
        date_column = "OrderDate"
        product_column = "item_id"
        quantity_column = "Quantity"

    return sdf, date_column, product_column, quantity_column


def main() -> None:
    args = parse_args()
    spark = create_spark()

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)

    sdf, date_column, product_column, quantity_column = load_dataset(args, spark)

    cfg = TrainingConfig(
        experiment_name=args.experiment_name,
        date_column=date_column,
        product_id_column=product_column,
        quantity_column=quantity_column,
        min_total_orders=args.min_orders,
        sample_products=args.sample_items if args.sample_items > 0 else None,
        feature_columns=parse_feature_columns(args.feature_columns),
        train_quantile=args.train_quantile,
        requirements_path=args.requirements_path,
        enable_stats=not args.disable_stats,
        enable_ml=not args.disable_ml,
    )

    df_feat = prepare_feature_frame(sdf, cfg)

    if args.per_category and "product_category" in df_feat.columns:
        categories = [row[0] for row in df_feat.select("product_category").distinct().collect()]
        for category in categories:
            segment_df = df_feat.filter(F.col("product_category") == category)
            if segment_df.count() == 0:
                continue
            run_training_workflow(segment_df, cfg, segment_label=category)
    else:
        run_training_workflow(df_feat, cfg)

    spark.stop()


if __name__ == "__main__":
    main()
