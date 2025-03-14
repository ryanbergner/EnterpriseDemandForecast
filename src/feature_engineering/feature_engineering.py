from pyspark.sql.functions import (
    col,
    month,
    year,
    coalesce,
    lit,
    explode,
    sequence,
    expr,
    when,
    last,
    lag,
    lead,
    stddev_samp,
    months_between,
    sum as spark_sum,
    avg as spark_avg,
)
from pyspark.sql.window import Window
from pyspark.sql import DataFrame

def add_features(
    df_agg: DataFrame,
    month_end_column: str = "MonthEndDate",
    product_id_column: str = "ItemNumber",
    quantity_column: str = "DemandQuantity",
) -> DataFrame:
    """
    Aggregates + expands months, plus minimal transformations for numeric columns:
      - month => cast to float
      - bucket => cast to float if you want a numeric approach
    Then returns a final df with numeric columns ready for Spark ML.
    """

    # -------------------------------------------------------------------------
    # A) Expand => fill missing => quantity=0
    # -------------------------------------------------------------------------
    df_minmax = (
        df_agg
        .select(
            product_id_column,
            col("min_date").alias("global_min_date"),
            col("max_date").alias("global_max_date"),
        )
        .distinct()
    )

    df_expanded = (
        df_minmax
        .withColumn(
            "all_months",
            sequence(
                col("global_min_date"),
                col("global_max_date"),
                expr("INTERVAL 1 MONTH"),
            ),
        )
        .select(product_id_column, explode(col("all_months")).alias(month_end_column))
    )
    df_filled = (
        df_expanded
        .join(
            df_agg.select(product_id_column, month_end_column, quantity_column),
            on=[product_id_column, month_end_column],
            how="left",
        )
        .withColumn(quantity_column, coalesce(col(quantity_column), lit(0.0)))
    )
    # Add month & year => cast to float (or string)
    df_filled = df_filled.withColumn(
        "month",
        month(col(month_end_column)).cast("float"),
    )
    df_filled = df_filled.withColumn(
        "year",
        year(col(month_end_column)).cast("float"),
    )
    # -------------------------------------------------------------------------
    # C) months_since_last_order, last_order_quantity
    # -------------------------------------------------------------------------
    w_init = (
        Window.partitionBy(product_id_column)
        .orderBy(month_end_column)
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    df_filled = df_filled.withColumn(
        "last_order_date",
        last(
            when(col(quantity_column) > 0, col(month_end_column)),
            ignorenulls=True,
        ).over(w_init),
    )
    df_filled = df_filled.withColumn(
        "months_since_last_order",
        months_between(col(month_end_column), col("last_order_date")),
    ).drop("last_order_date")

    df_filled = df_filled.withColumn(
        "last_order_quantity",
        last(
            when(col(quantity_column) > 0, col(quantity_column)),
            ignorenulls=True,
        ).over(w_init),
    )
    # -------------------------------------------------------------------------
    # D) lag_1..lag_5, lead_month_1
    # -------------------------------------------------------------------------
    w_lag = Window.partitionBy(product_id_column).orderBy(month_end_column)
    for i in range(1, 6):
        df_filled = df_filled.withColumn(
            f"lag_{i}", lag(col(quantity_column), i).over(w_lag)
        )
    df_filled = df_filled.withColumn(
        "lead_month_1",
        lead(col(quantity_column)).over(w_lag),
    )
    # -------------------------------------------------------------------------
    # E) total_orders => # months with quantity>0
    # -------------------------------------------------------------------------
    total_df = (
        df_filled
        .groupBy(product_id_column)
        .agg(
            spark_sum(
                when(col(quantity_column) > 0, lit(1)).otherwise(lit(0))
            ).alias("total_orders")
        )
    )
    df_filled = df_filled.join(total_df, on=product_id_column, how="left")
    # -------------------------------------------------------------------------
    # F) CoV + avg_demand_interval => product_category
    # -------------------------------------------------------------------------
    df_pos = df_filled.filter(col(quantity_column) > 0)

    # (a) CoV
    mean_q = (
        df_pos.groupBy(product_id_column)
        .agg(spark_avg(col(quantity_column)).alias("mean_quantity"))
    )
    std_q = (
        df_pos.groupBy(product_id_column)
        .agg(stddev_samp(col(quantity_column)).alias("stddev_quantity"))
    )
    df_cov = (
        mean_q
        .join(std_q, on=product_id_column, how="inner")
        .withColumn("cov_quantity", col("stddev_quantity") / col("mean_quantity"))
        .select(product_id_column, "cov_quantity")
    )
    df_filled = df_filled.join(df_cov, on=product_id_column, how="left")

    # (b) avg_demand_interval
    w_interval = Window.partitionBy(product_id_column).orderBy(month_end_column)
    df_pos = df_pos.withColumn(
        "demand_interval",
        months_between(
            col(month_end_column),
            lag(col(month_end_column), 1).over(w_interval),
        ),
    )
    df_adi = (
        df_pos.groupBy(product_id_column)
        .agg(spark_avg(col("demand_interval")).alias("avg_demand_interval"))
    )
    df_filled = df_filled.join(df_adi, on=product_id_column, how="left")

    # (c) product_category
    df_filled = df_filled.withColumn(
        "product_category",
        when(
            (col("cov_quantity") <= 0.8) & (col("avg_demand_interval") <= 1.5),
            "Smooth",
        )
        .when(
            (col("cov_quantity") > 0.8) & (col("avg_demand_interval") <= 1.5),
            "Erratic",
        )
        .when(
            (col("cov_quantity") <= 0.8) & (col("avg_demand_interval") > 1.5),
            "Intermittent",
        )
        .otherwise("Lumpy"),
    )

    # -------------------------------------------------------------------------
    # G) Moving averages for last 4/8/12 months => rowBetween(-3,0), etc.
    # -------------------------------------------------------------------------
    w_order = Window.partitionBy(product_id_column).orderBy(month_end_column)

    df_moves = (
        df_filled
        .withColumn(
            "ma_4_month",
            spark_avg(col(quantity_column)).over(w_order.rowsBetween(-3, 0)),
        )
        .withColumn(
            "ma_8_month",
            spark_avg(col(quantity_column)).over(w_order.rowsBetween(-7, 0)),
        )
        .withColumn(
            "ma_12_month",
            spark_avg(col(quantity_column)).over(w_order.rowsBetween(-11, 0)),
        )
    )
    # -------------------------------------------------------------------------
    # H) For non-zero sales => last 3/5/7 => forward fill
    # -------------------------------------------------------------------------
    df_nonzero = df_moves.filter(col(quantity_column) > 0)
    w_nz = Window.partitionBy(product_id_column).orderBy(month_end_column)

    df_nz_temp = (
        df_nonzero
        .withColumn(
            "ma_3_temp",
            spark_avg(col(quantity_column)).over(w_nz.rowsBetween(-2, 0)),
        )
        .withColumn(
            "ma_5_temp",
            spark_avg(col(quantity_column)).over(w_nz.rowsBetween(-4, 0)),
        )
        .withColumn(
            "ma_7_temp",
            spark_avg(col(quantity_column)).over(w_nz.rowsBetween(-6, 0)),
        )
        .select(
            product_id_column,
            month_end_column,
            "ma_3_temp",
            "ma_5_temp",
            "ma_7_temp",
        )
    )
    df_join_nz = df_moves.join(
        df_nz_temp, on=[product_id_column, month_end_column], how="left"
    )
    w_ff = (
        Window.partitionBy(product_id_column)
        .orderBy(month_end_column)
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    df_sales = (
        df_join_nz
        .withColumn("ma_3_sales", last(col("ma_3_temp"), ignorenulls=True).over(w_ff))
        .withColumn("ma_5_sales", last(col("ma_5_temp"), ignorenulls=True).over(w_ff))
        .withColumn("ma_7_sales", last(col("ma_7_temp"), ignorenulls=True).over(w_ff))
        .drop("ma_3_temp", "ma_5_temp", "ma_7_temp")
    )
    # -------------------------------------------------------------------------
    # I) avg_demand_quantity => 10 buckets => B1..B10
    # -------------------------------------------------------------------------
    w_prod = Window.partitionBy(product_id_column)
    df_buck = df_sales.withColumn(
        "avg_demand_quantity",
        spark_avg(col(quantity_column)).over(w_prod),
    )
    deciles = [float(i) / 10.0 for i in range(1, 10)]
    qtiles = df_buck.approxQuantile("avg_demand_quantity", deciles, 0.0)
    while len(qtiles) < 9:
        qtiles.append(qtiles[-1] if qtiles else 0.0)
    df_buck = df_buck.withColumn(
        "bucket",
        when(col("avg_demand_quantity") <= qtiles[0], "B1")
        .when(col("avg_demand_quantity") <= qtiles[1], "B2")
        .when(col("avg_demand_quantity") <= qtiles[2], "B3")
        .when(col("avg_demand_quantity") <= qtiles[3], "B4")
        .when(col("avg_demand_quantity") <= qtiles[4], "B5")
        .when(col("avg_demand_quantity") <= qtiles[5], "B6")
        .when(col("avg_demand_quantity") <= qtiles[6], "B7")
        .when(col("avg_demand_quantity") <= qtiles[7], "B8")
        .when(col("avg_demand_quantity") <= qtiles[8], "B9")
        .otherwise("B10"),
    )
    return df_buck
