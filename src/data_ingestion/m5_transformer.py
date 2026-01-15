"""
M5 dataset transformer utilities.
"""

from functools import reduce
from typing import List, Optional, Iterable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    date_add,
    regexp_extract
)


def _get_spark_session(spark: Optional[SparkSession]) -> SparkSession:
    if spark is not None:
        return spark
    return SparkSession.builder.getOrCreate()


def _chunk_list(items: List[str], chunk_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _build_stack_expr(day_columns: List[str]) -> str:
    stack_parts = ", ".join([f"'{c}', `{c}`" for c in day_columns])
    return f"stack({len(day_columns)}, {stack_parts}) as (day_col, Quantity)"


def _unpivot_with_stack(
    df_wide: DataFrame,
    item_col: str,
    day_columns: List[str],
    date_col: str,
    product_id_col: str,
    quantity_col: str,
    start_date: str,
    chunk_size: int
) -> DataFrame:
    parts: List[DataFrame] = []

    for chunk in _chunk_list(day_columns, chunk_size):
        stack_expr = _build_stack_expr(chunk)
        part_df = (
            df_wide
            .selectExpr(f"`{item_col}` as `{product_id_col}`", stack_expr)
            .withColumn("day_num", regexp_extract(col("day_col"), r"^d_(\\d+)$", 1).cast("int"))
            .withColumn(date_col, date_add(lit(start_date), col("day_num") - lit(1)))
            .drop("day_col", "day_num")
            .withColumn(quantity_col, col(quantity_col).cast("double"))
            .fillna(0.0, subset=[quantity_col])
        )
        parts.append(part_df)

    if not parts:
        raise ValueError("No day columns found to unpivot.")

    return reduce(lambda a, b: a.unionByName(b), parts)


def transform_m5_wide_to_long(
    path: str,
    spark: Optional[SparkSession] = None,
    date_col: str = "OrderDate",
    product_id_col: str = "item_id",
    quantity_col: str = "Quantity",
    start_date: str = "2011-01-29",
    chunk_size: int = 100
) -> DataFrame:
    """
    Transform M5 wide-format CSV (d_1..d_N) into long-format DataFrame.
    """
    spark = _get_spark_session(spark)
    df_wide = spark.read.format("csv").option("header", True).load(path)

    if "item_id" in df_wide.columns:
        item_col = "item_id"
    else:
        raise ValueError("Expected 'item_id' column in M5 dataset.")

    day_columns = [c for c in df_wide.columns if c.startswith("d_")]
    if not day_columns:
        raise ValueError("No day columns (d_*) found in M5 dataset.")

    return _unpivot_with_stack(
        df_wide=df_wide,
        item_col=item_col,
        day_columns=day_columns,
        date_col=date_col,
        product_id_col=product_id_col,
        quantity_col=quantity_col,
        start_date=start_date,
        chunk_size=chunk_size
    )


class M5Transformer:
    """
    Wrapper class for M5 dataset transformation.
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        start_date: str = "2011-01-29"
    ):
        self.spark = _get_spark_session(spark)
        self.start_date = start_date

    def transform(
        self,
        path: str,
        date_col: str = "OrderDate",
        product_id_col: str = "item_id",
        quantity_col: str = "Quantity",
        chunk_size: int = 100
    ) -> DataFrame:
        return transform_m5_wide_to_long(
            path=path,
            spark=self.spark,
            date_col=date_col,
            product_id_col=product_id_col,
            quantity_col=quantity_col,
            start_date=self.start_date,
            chunk_size=chunk_size
        )

    def transform_to_file(
        self,
        path: str,
        output_path: str,
        format: str = "parquet",
        mode: str = "overwrite",
        date_col: str = "OrderDate",
        product_id_col: str = "item_id",
        quantity_col: str = "Quantity",
        chunk_size: int = 100
    ) -> None:
        df = self.transform(
            path=path,
            date_col=date_col,
            product_id_col=product_id_col,
            quantity_col=quantity_col,
            chunk_size=chunk_size
        )
        df.write.mode(mode).format(format).save(output_path)
