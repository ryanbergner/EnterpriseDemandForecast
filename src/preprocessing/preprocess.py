from pyspark.sql.functions import (
    count,
    lag,
    last_day,
    expr,
    explode,
    sequence,
    date_format,
    col,
    last,
    lead,
    when,
    months_between,
    month,
    year,
    avg,
    to_timestamp,
    to_date,
    mean,
    lit,
    sum as spark_sum,
    avg as spark_avg,
    min as spark_min,
    max as spark_max
)
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType,
    DecimalType,
    DoubleType
)
from typing import Optional, List, Any, Dict
from datetime import datetime, timedelta
from math import sqrt as math_sqrt


def retrieve_sales_data(table_path: str) -> DataFrame:
    """
    Retrieves sales data from the combined_gold database for a specified category.

    Args:
        table_path (str): Path to the Delta table.

    Returns:
        DataFrame: Sales data DataFrame filtered based on the specified category.
    """
    # Load data from Delta table
    df = spark.read.format("delta").load(table_path)

    # Apply necessary filters and transformations
    df = (
        df.filter("SalesInvoiceProductKey NOT LIKE '%-9999%'")
          .filter("CurrencyCode = 'USD'")
          .filter("OrderType <> 'ReturnItem'")
          .filter("Quantity > 0")
          .filter("UnitPrice IS NOT NULL")
          .filter("UnitCost IS NOT NULL")
          .orderBy("InvoiceDate")
    )

    # Convert DecimalType to DoubleType
    for field in df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            df = df.withColumn(field.name, col(field.name).cast(DoubleType()))

    return df

def aggregate_sales_data(
    df: DataFrame,
    date_column: str,
    product_id_column: str,
    quantity_column: str,
    month_end_column: str = "MonthEndDate"
) -> DataFrame:
    """
    1) Convert raw date_column => month_end_column (last_day).
    2) Sum quantity_column by [product_id_column, month_end_column].
    3) For each product, find min & max date => store them as min_date, max_date.

    Returns df_agg with columns:
      [product_id_column, month_end_column, quantity_column, min_date, max_date].
    """

    # 1) Convert daily date => last_day-of-month
    df_month = df.withColumn(
        month_end_column,
        last_day(col(date_column))
    )

    # 2) Sum quantity by [product_id, MonthEndDate]
    df_grouped = (
        df_month
        .groupBy(product_id_column, month_end_column)
        .agg(spark_sum(col(quantity_column)).alias(quantity_column))
    )

    # 3) For each product => earliest & latest MonthEndDate => rename as min_date, max_date
    df_minmax = (
        df_grouped
        .groupBy(product_id_column)
        .agg(
            spark_min(col(month_end_column)).alias("min_date"),
            spark_max(col(month_end_column)).alias("max_date")
        )
    )

    # 4) Join => final aggregated dataset
    df_agg = df_grouped.join(df_minmax, on=product_id_column, how="left")

    return df_agg
