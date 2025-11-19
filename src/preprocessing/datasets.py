"""Utility loaders for the M5 dataset and compatible schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

M5_DEFAULT_START_DATE = "2011-01-29"
M5_IDENTIFIER_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


@dataclass
class DailyDataset:
    """Simple container describing the important column names for downstream steps."""

    date_column: str = "OrderDate"
    product_column: str = "item_id"
    quantity_column: str = "Quantity"


def load_m5_wide_dataset(
    spark: SparkSession,
    path: str,
    sample_items: Optional[int] = None,
    calendar_start: str = M5_DEFAULT_START_DATE,
) -> DataFrame:
    """
    Read the canonical M5 competition CSV (wide layout) and return a tidy Spark DataFrame
    with columns [item_id, OrderDate, Quantity] plus the standard identifier metadata.
    """
    sdf = (
        spark.read
        .format("csv")
        .option("header", True)
        .load(path)
    )

    if sample_items and sample_items > 0:
        sdf = sdf.limit(sample_items)

    day_columns = sorted(
        [c for c in sdf.columns if c.startswith("d_")],
        key=lambda c: int(c.split("_")[1])
    )
    if not day_columns:
        raise ValueError(
            "No day columns found. Ensure the dataset uses the standard d_1 ... d_N layout."
        )

    missing_ids = [col for col in M5_IDENTIFIER_COLUMNS if col not in sdf.columns]
    if missing_ids:
        raise ValueError(
            f"The dataset is missing required identifier columns: {missing_ids}"
        )

    stack_expr = ", ".join([f"'{col}', `{col}`" for col in day_columns])
    long_df = sdf.selectExpr(
        *M5_IDENTIFIER_COLUMNS,
        f"stack({len(day_columns)}, {stack_expr}) as (day_id, Quantity)"
    )

    long_df = (
        long_df.withColumn(
            "day_number",
            F.regexp_extract("day_id", "d_(\\d+)", 1).cast("int")
        )
        .withColumn(
            "OrderDate",
            F.expr(f"date_add('{calendar_start}', day_number - 1)")
        )
        .drop("day_id", "day_number")
        .withColumn("Quantity", F.col("Quantity").cast(DoubleType()))
    )

    return long_df


def load_custom_daily_dataset(
    spark: SparkSession,
    path: str,
    date_column: str,
    product_column: str,
    quantity_column: str,
    select_columns: Optional[Sequence[str]] = None,
) -> DataFrame:
    """
    Load a user-provided dataset that already contains daily records and rename columns
    to the canonical [OrderDate, item_id, Quantity] layout expected by the pipeline.
    """
    reader = spark.read.format("csv").option("header", True)
    if path.lower().endswith((".parquet", ".pq")):
        sdf = spark.read.parquet(path)
    else:
        sdf = reader.load(path)

    required_cols = {date_column, product_column, quantity_column}
    missing = required_cols - set(sdf.columns)
    if missing:
        raise ValueError(f"Custom dataset missing required columns: {sorted(missing)}")

    sdf = (
        sdf.withColumn("OrderDate", F.to_date(F.col(date_column)))
        .withColumn("Quantity", F.col(quantity_column).cast(DoubleType()))
        .withColumn("item_id", F.col(product_column))
    )

    keep_columns = {"OrderDate", "item_id", "Quantity"}
    if select_columns:
        keep_columns.update(set(select_columns) & set(sdf.columns))

    return sdf.select(*keep_columns)
