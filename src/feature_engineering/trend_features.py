"""
Trend Feature Engineering Module

Adds advanced trend-based features for time series forecasting:
- Time indices and monotonic counters
- Growth rates and percentage changes
- Momentum and acceleration indicators
- Trend strength metrics
- Detrending features
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, lag, lead, when, coalesce, lit,
    row_number, sum as spark_sum, avg as spark_avg,
    pow as spark_pow, sqrt, abs as spark_abs,
    months_between, datediff
)
from pyspark.sql.window import Window
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_trend_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    windows: list = [3, 6, 12]
) -> DataFrame:
    """
    Add comprehensive trend-based features to time series data.
    
    Features added:
    - time_index: Monotonic counter from first observation
    - growth_rate_Xm: (current - X months ago) / X months ago
    - momentum_Xm: Second derivative of demand
    - trend_strength_Xm: Linear trend slope over window
    - acceleration: Change in growth rate
    - cumulative_sum: Running total of demand
    - relative_position: Position in product lifecycle (0-1)
    
    Args:
        df: Input DataFrame
        value_col: Column name for the value to compute trends on
        date_col: Date column name
        product_col: Product identifier column
        windows: List of window sizes for rolling trend calculations
    
    Returns:
        DataFrame with trend features added
    
    Example:
        df_with_trends = add_trend_features(
            df, 
            value_col='DemandQuantity',
            date_col='MonthEndDate',
            product_col='ItemNumber',
            windows=[3, 6, 12]
        )
    """
    logger.info(f"🔄 Adding trend features with windows: {windows}")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    # 1. Time index - monotonic counter from first observation
    df = df.withColumn(
        "time_index",
        row_number().over(w)
    )
    
    logger.info("   ✓ Added time_index")
    
    # 2. Growth rates for different periods
    for period in windows:
        lag_col = f"_lag_{period}"
        growth_col = f"growth_rate_{period}m"
        
        df = df.withColumn(
            lag_col,
            lag(col(value_col), period).over(w)
        )
        
        # Calculate percentage change: (current - past) / past
        df = df.withColumn(
            growth_col,
            when(
                (col(lag_col).isNotNull()) & (col(lag_col) != 0),
                (col(value_col) - col(lag_col)) / col(lag_col)
            ).otherwise(None)
        )
        
        df = df.drop(lag_col)
    
    logger.info(f"   ✓ Added growth_rate features for {windows}")
    
    # 3. Momentum - rate of change of growth (second derivative)
    for period in windows:
        growth_col = f"growth_rate_{period}m"
        momentum_col = f"momentum_{period}m"
        
        if growth_col in df.columns:
            df = df.withColumn(
                momentum_col,
                col(growth_col) - lag(col(growth_col), 1).over(w)
            )
    
    logger.info(f"   ✓ Added momentum features for {windows}")
    
    # 4. Trend strength - slope of linear regression over window
    for period in windows:
        trend_col = f"trend_strength_{period}m"
        
        # Simple trend approximation: (current - period_ago) / period
        df = df.withColumn(
            f"_lag_{period}",
            lag(col(value_col), period).over(w)
        )
        
        df = df.withColumn(
            trend_col,
            when(
                col(f"_lag_{period}").isNotNull(),
                (col(value_col) - col(f"_lag_{period}")) / period
            ).otherwise(None)
        )
        
        df = df.drop(f"_lag_{period}")
    
    logger.info(f"   ✓ Added trend_strength features for {windows}")
    
    # 5. Acceleration - change in growth rate (third derivative indicator)
    if "growth_rate_3m" in df.columns:
        df = df.withColumn(
            "acceleration",
            col("growth_rate_3m") - lag(col("growth_rate_3m"), 1).over(w)
        )
        logger.info("   ✓ Added acceleration")
    
    # 6. Cumulative sum - running total
    df = df.withColumn(
        "cumulative_demand",
        spark_sum(col(value_col)).over(
            w.rowsBetween(Window.unboundedPreceding, 0)
        )
    )
    logger.info("   ✓ Added cumulative_demand")
    
    # 7. Relative position in lifecycle (0 = start, 1 = end)
    w_full = Window.partitionBy(product_col)
    df = df.withColumn(
        "_max_time_index",
        spark_avg(col("time_index")).over(w_full)  # Using max approximation
    )
    
    df = df.withColumn(
        "relative_position",
        col("time_index") / col("_max_time_index")
    ).drop("_max_time_index")
    
    logger.info("   ✓ Added relative_position")
    
    # 8. Velocity - rate of change over recent periods
    df = df.withColumn(
        "velocity_1m",
        col(value_col) - lag(col(value_col), 1).over(w)
    )
    logger.info("   ✓ Added velocity_1m")
    
    # 9. Direction indicator - is trend increasing or decreasing
    if "trend_strength_3m" in df.columns:
        df = df.withColumn(
            "trend_direction",
            when(col("trend_strength_3m") > 0, lit(1))
            .when(col("trend_strength_3m") < 0, lit(-1))
            .otherwise(lit(0))
        )
        logger.info("   ✓ Added trend_direction")
    
    logger.info(f"✅ Trend features complete! Added {len(windows) * 3 + 6} new features")
    
    return df


def add_detrending_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    window: int = 12
) -> DataFrame:
    """
    Add detrended features by removing trend component.
    Useful for isolating seasonal and irregular components.
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier
        window: Window size for trend calculation
    
    Returns:
        DataFrame with detrended features
    """
    logger.info(f"📉 Adding detrending features (window={window})")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    w_centered = w.rowsBetween(-window // 2, window // 2)
    
    # Calculate centered moving average as trend
    df = df.withColumn(
        f"trend_{window}m",
        spark_avg(col(value_col)).over(w_centered)
    )
    
    # Detrended value = actual - trend
    df = df.withColumn(
        f"detrended_{window}m",
        col(value_col) - col(f"trend_{window}m")
    )
    
    # Detrended ratio = actual / trend
    df = df.withColumn(
        f"detrended_ratio_{window}m",
        when(
            col(f"trend_{window}m") != 0,
            col(value_col) / col(f"trend_{window}m")
        ).otherwise(lit(1.0))
    )
    
    logger.info(f"   ✓ Added trend_{window}m, detrended_{window}m, detrended_ratio_{window}m")
    
    return df


def add_lifecycle_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str
) -> DataFrame:
    """
    Add product lifecycle stage features.
    Classifies products into Introduction, Growth, Maturity, Decline stages.
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier
    
    Returns:
        DataFrame with lifecycle features
    """
    logger.info("🔄 Adding product lifecycle features")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    # Calculate early-stage and late-stage averages
    df = df.withColumn("_row_num", row_number().over(w))
    
    w_product = Window.partitionBy(product_col)
    df = df.withColumn("_total_periods", spark_sum(lit(1)).over(w_product))
    
    # Early stage: first 25% of periods
    df = df.withColumn(
        "_early_stage_avg",
        spark_avg(
            when(col("_row_num") <= col("_total_periods") * 0.25, col(value_col))
        ).over(w_product)
    )
    
    # Late stage: last 25% of periods
    df = df.withColumn(
        "_late_stage_avg",
        spark_avg(
            when(col("_row_num") > col("_total_periods") * 0.75, col(value_col))
        ).over(w_product)
    )
    
    # Mid stage: middle 50%
    df = df.withColumn(
        "_mid_stage_avg",
        spark_avg(
            when(
                (col("_row_num") > col("_total_periods") * 0.25) &
                (col("_row_num") <= col("_total_periods") * 0.75),
                col(value_col)
            )
        ).over(w_product)
    )
    
    # Lifecycle stage classification based on relative performance
    df = df.withColumn(
        "lifecycle_stage",
        when(
            col("_row_num") <= col("_total_periods") * 0.25,
            lit("Introduction")
        ).when(
            (col("_mid_stage_avg") > col("_early_stage_avg") * 1.2) &
            (col("_row_num") <= col("_total_periods") * 0.5),
            lit("Growth")
        ).when(
            col("_row_num") > col("_total_periods") * 0.75,
            when(
                col("_late_stage_avg") < col("_mid_stage_avg") * 0.8,
                lit("Decline")
            ).otherwise(lit("Maturity"))
        ).otherwise(lit("Maturity"))
    )
    
    # Lifecycle growth indicator
    df = df.withColumn(
        "lifecycle_growth",
        (col("_late_stage_avg") - col("_early_stage_avg")) / 
        coalesce(col("_early_stage_avg"), lit(1.0))
    )
    
    # Clean up temporary columns
    df = df.drop(
        "_row_num", "_total_periods", "_early_stage_avg",
        "_late_stage_avg", "_mid_stage_avg"
    )
    
    logger.info("   ✓ Added lifecycle_stage, lifecycle_growth")
    
    return df


def add_change_point_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    threshold: float = 0.5
) -> DataFrame:
    """
    Detect and flag significant change points in the time series.
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier
        threshold: Minimum relative change to flag as change point
    
    Returns:
        DataFrame with change point features
    """
    logger.info(f"🎯 Adding change point detection (threshold={threshold})")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    # Calculate period-over-period change
    df = df.withColumn(
        "_prev_value",
        lag(col(value_col), 1).over(w)
    )
    
    df = df.withColumn(
        "change_magnitude",
        when(
            col("_prev_value") != 0,
            spark_abs(col(value_col) - col("_prev_value")) / col("_prev_value")
        ).otherwise(None)
    )
    
    # Flag significant changes
    df = df.withColumn(
        "is_change_point",
        when(
            col("change_magnitude") > threshold,
            lit(1)
        ).otherwise(lit(0))
    )
    
    # Periods since last change point
    df = df.withColumn(
        "periods_since_change",
        spark_sum(lit(1)).over(w) - 
        spark_sum(col("is_change_point")).over(
            w.rowsBetween(Window.unboundedPreceding, 0)
        )
    )
    
    df = df.drop("_prev_value")
    
    logger.info("   ✓ Added change_magnitude, is_change_point, periods_since_change")
    
    return df
