#!/usr/bin/env python
"""
Unified training entrypoint for local and programmatic usage.
"""

from typing import Optional, Dict, Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

from src.model_training.training_workflow import (
    TrainingConfig,
    prepare_feature_frame,
    run_training_workflow,
)
from src.model_training.batch_optimizer import BatchTrainingOptimizer


def main_train(
    data_path: str = None,
    data_df: Optional["DataFrame"] = None,
    date_col: str = "OrderDate",
    product_id_col: str = "item_id",
    quantity_col: str = "Quantity",
    month_end_col: str = "MonthEndDate",
    experiment_prefix: str = "/M5_Forecasting",
    spark: Optional["SparkSession"] = None,
    use_advanced_features: bool = False,
    per_category: bool = True
) -> Dict[str, Any]:
    created_spark = False
    if spark is None:
        spark = SparkSession.builder.appName("DemandForecastTraining").getOrCreate()
        created_spark = True

    if data_df is not None:
        df = data_df
    elif data_path:
        df = spark.read.format("csv").option("header", True).load(data_path)
    else:
        raise ValueError("Either data_path or data_df must be provided")

    df = (
        df.withColumn(date_col, col(date_col).cast("date"))
          .withColumn(quantity_col, col(quantity_col).cast("double"))
    )

    cfg = TrainingConfig(
        experiment_name=experiment_prefix,
        date_column=date_col,
        product_id_column=product_id_col,
        quantity_column=quantity_col,
        month_end_column=month_end_col,
        use_advanced_features=use_advanced_features,
    )

    df_feat = prepare_feature_frame(df, cfg)
    optimizer = BatchTrainingOptimizer(spark)

    if per_category and "product_category" in df_feat.columns:
        categories = [row[0] for row in df_feat.select("product_category").distinct().collect()]
        for category in categories:
            segment = df_feat.filter(col("product_category") == category)
            if segment.count() == 0:
                continue
            with optimizer.optimized_context(segment, partition_cols=[product_id_col]) as opt_df:
                run_training_workflow(opt_df, cfg, segment_label=category)
    else:
        with optimizer.optimized_context(df_feat, partition_cols=[product_id_col]) as opt_df:
            run_training_workflow(opt_df, cfg, segment_label="global")

    if created_spark:
        spark.stop()

    return {
        "status": "success",
        "experiment_prefix": experiment_prefix,
    }


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    result = main_train(data_path=data_path)
    print(f"Training result: {result}")
