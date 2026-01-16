#!/usr/bin/env python
"""
Local training entrypoint with format detection.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from pyspark.sql import SparkSession, DataFrame

from src.data_ingestion import LocalDataLoader
from main_train import main_train


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "LocalForecasting") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "4")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def train_local(
    data_path: str,
    date_col: str = "OrderDate",
    product_id_col: str = "item_id",
    quantity_col: str = "Quantity",
    month_end_col: str = "MonthEndDate",
    experiment_prefix: str = "/Local_Forecasting",
    m5_start_date: str = "2011-01-29",
    spark: Optional[SparkSession] = None,
    auto_transform_m5: bool = True,
    use_advanced_features: bool = False
) -> Dict[str, Any]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    created_spark = False
    if spark is None:
        spark = create_spark_session()
        created_spark = True

    loader = LocalDataLoader(
        spark=spark,
        date_col=date_col,
        product_id_col=product_id_col,
        quantity_col=quantity_col,
        auto_transform_m5=auto_transform_m5,
        m5_start_date=m5_start_date
    )

    df = loader.load(data_path)
    logger.info("Loaded data with %s rows", df.count())

    result = main_train(
        data_df=df,
        date_col=date_col,
        product_id_col=product_id_col,
        quantity_col=quantity_col,
        month_end_col=month_end_col,
        experiment_prefix=experiment_prefix,
        spark=spark,
        use_advanced_features=use_advanced_features
    )
    result["data_path"] = str(data_path)

    if created_spark:
        spark.stop()

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train forecasting models on local data")
    parser.add_argument("data_path", help="Path to data file (CSV/Parquet/M5)")
    parser.add_argument("--date-col", default="OrderDate")
    parser.add_argument("--product-col", default="item_id")
    parser.add_argument("--quantity-col", default="Quantity")
    parser.add_argument("--month-end-col", default="MonthEndDate")
    parser.add_argument("--experiment", default="/Local_Forecasting")
    parser.add_argument("--m5-start-date", default="2011-01-29")
    parser.add_argument("--no-transform-m5", action="store_true")
    parser.add_argument("--advanced-features", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        result = train_local(
            data_path=args.data_path,
            date_col=args.date_col,
            product_id_col=args.product_col,
            quantity_col=args.quantity_col,
            month_end_col=args.month_end_col,
            experiment_prefix=args.experiment,
            m5_start_date=args.m5_start_date,
            auto_transform_m5=not args.no_transform_m5,
            use_advanced_features=args.advanced_features
        )
        print(f"Training result: {result}")
        return 0
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
