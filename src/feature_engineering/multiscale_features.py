"""
Multiscale Feature Engineering Module (SCINet-Inspired)

Implements multi-resolution temporal features for demand forecasting:
- Multiple aggregation scales (weekly, monthly, quarterly)
- Downsampling and upsampling features
- Cross-scale interactions
- Temporal hierarchy features

Based on research from SCINet (Sample Convolution and Interaction Network)
which shows that multiscale decomposition improves time series forecasting.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, lit, when, coalesce,
    avg as spark_avg, sum as spark_sum, count as spark_count,
    min as spark_min, max as spark_max,
    stddev_samp, variance,
    lag, lead, first, last,
    weekofyear, month, year, quarter,
    date_trunc, datediff, date_add,
    row_number, dense_rank
)
from pyspark.sql.window import Window
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_multiscale_aggregation_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    scales: List[str] = ['week', 'month', 'quarter']
) -> DataFrame:
    """
    Add features aggregated at multiple temporal scales.
    
    This captures patterns at different frequencies:
    - Weekly: Short-term fluctuations
    - Monthly: Medium-term trends
    - Quarterly: Seasonal patterns
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier column
        scales: List of time scales ('week', 'month', 'quarter', 'year')
    
    Returns:
        DataFrame with multiscale features
    
    Example:
        df = add_multiscale_aggregation_features(
            df,
            value_col='DemandQuantity',
            date_col='MonthEndDate',
            product_col='ItemNumber',
            scales=['week', 'month', 'quarter']
        )
    """
    logger.info(f"Adding multiscale aggregation features at scales: {scales}")
    
    # Define window partitions
    w_product = Window.partitionBy(product_col)
    
    for scale in scales:
        logger.info(f"  Processing scale: {scale}")
        
        # Add scale identifier column
        if scale == 'week':
            df = df.withColumn(
                f"_{scale}_key",
                weekofyear(col(date_col))
            )
        elif scale == 'month':
            df = df.withColumn(
                f"_{scale}_key",
                month(col(date_col))
            )
        elif scale == 'quarter':
            df = df.withColumn(
                f"_{scale}_key",
                quarter(col(date_col))
            )
        elif scale == 'year':
            df = df.withColumn(
                f"_{scale}_key",
                year(col(date_col))
            )
        
        # Create window for aggregation within scale
        w_scale = Window.partitionBy(product_col, f"_{scale}_key")
        
        # Aggregate statistics at this scale
        df = df.withColumn(
            f"scale_{scale}_mean",
            spark_avg(col(value_col)).over(w_scale)
        )
        df = df.withColumn(
            f"scale_{scale}_sum",
            spark_sum(col(value_col)).over(w_scale)
        )
        df = df.withColumn(
            f"scale_{scale}_std",
            stddev_samp(col(value_col)).over(w_scale)
        )
        df = df.withColumn(
            f"scale_{scale}_min",
            spark_min(col(value_col)).over(w_scale)
        )
        df = df.withColumn(
            f"scale_{scale}_max",
            spark_max(col(value_col)).over(w_scale)
        )
        
        # Calculate deviation from scale mean
        df = df.withColumn(
            f"deviation_from_{scale}_mean",
            col(value_col) - col(f"scale_{scale}_mean")
        )
        
        # Relative position within scale
        df = df.withColumn(
            f"relative_to_{scale}",
            when(
                col(f"scale_{scale}_mean") != 0,
                col(value_col) / col(f"scale_{scale}_mean")
            ).otherwise(lit(1.0))
        )
        
        # Drop temporary key column
        df = df.drop(f"_{scale}_key")
        
        logger.info(f"    Added: scale_{scale}_mean, scale_{scale}_std, deviation_from_{scale}_mean")
    
    logger.info(f"Multiscale aggregation features complete!")
    return df


def add_cross_scale_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str
) -> DataFrame:
    """
    Add features that capture relationships between different time scales.
    
    These features help identify:
    - Whether recent demand differs from long-term average
    - Trend acceleration across scales
    - Scale-to-scale momentum
    
    Args:
        df: Input DataFrame (should have multiscale features already)
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier column
    
    Returns:
        DataFrame with cross-scale features
    """
    logger.info("Adding cross-scale interaction features")
    
    # Month vs Quarter ratio (short-term vs medium-term)
    if 'scale_month_mean' in df.columns and 'scale_quarter_mean' in df.columns:
        df = df.withColumn(
            "month_to_quarter_ratio",
            when(
                col("scale_quarter_mean") != 0,
                col("scale_month_mean") / col("scale_quarter_mean")
            ).otherwise(lit(1.0))
        )
        logger.info("  Added: month_to_quarter_ratio")
    
    # Week vs Month ratio (very short-term vs short-term)
    if 'scale_week_mean' in df.columns and 'scale_month_mean' in df.columns:
        df = df.withColumn(
            "week_to_month_ratio",
            when(
                col("scale_month_mean") != 0,
                col("scale_week_mean") / col("scale_month_mean")
            ).otherwise(lit(1.0))
        )
        logger.info("  Added: week_to_month_ratio")
    
    # Current value vs all scales (deviation from each scale)
    for scale in ['week', 'month', 'quarter']:
        mean_col = f"scale_{scale}_mean"
        if mean_col in df.columns:
            df = df.withColumn(
                f"value_vs_{scale}_zscore",
                when(
                    col(f"scale_{scale}_std") != 0,
                    (col(value_col) - col(mean_col)) / col(f"scale_{scale}_std")
                ).otherwise(lit(0.0))
            )
            logger.info(f"  Added: value_vs_{scale}_zscore")
    
    logger.info("Cross-scale features complete!")
    return df


def add_temporal_hierarchy_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str
) -> DataFrame:
    """
    Add hierarchical temporal features for multi-level forecasting.
    
    Features:
    - Year-over-year comparisons
    - Quarter-over-quarter changes
    - Month-over-month changes
    - Same-period-last-year values
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier column
    
    Returns:
        DataFrame with temporal hierarchy features
    """
    logger.info("Adding temporal hierarchy features")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    # Add temporal identifiers
    df = df.withColumn("_month", month(col(date_col)))
    df = df.withColumn("_year", year(col(date_col)))
    df = df.withColumn("_quarter", quarter(col(date_col)))
    
    # Same month last year (lag 12 months)
    df = df.withColumn(
        "same_month_last_year",
        lag(col(value_col), 12).over(w)
    )
    
    # Year-over-year growth rate
    df = df.withColumn(
        "yoy_growth_rate",
        when(
            (col("same_month_last_year").isNotNull()) & (col("same_month_last_year") != 0),
            (col(value_col) - col("same_month_last_year")) / col("same_month_last_year")
        ).otherwise(None)
    )
    
    # Same quarter last year (lag 4 quarters = 12 months for monthly data)
    df = df.withColumn(
        "same_quarter_last_year",
        lag(col(value_col), 12).over(w)
    )
    
    # Quarter-over-quarter (lag 3 months)
    df = df.withColumn(
        "prev_quarter_value",
        lag(col(value_col), 3).over(w)
    )
    
    df = df.withColumn(
        "qoq_growth_rate",
        when(
            (col("prev_quarter_value").isNotNull()) & (col("prev_quarter_value") != 0),
            (col(value_col) - col("prev_quarter_value")) / col("prev_quarter_value")
        ).otherwise(None)
    )
    
    # Month-over-month (already captured in lag_1 typically)
    df = df.withColumn(
        "prev_month_value",
        lag(col(value_col), 1).over(w)
    )
    
    df = df.withColumn(
        "mom_growth_rate",
        when(
            (col("prev_month_value").isNotNull()) & (col("prev_month_value") != 0),
            (col(value_col) - col("prev_month_value")) / col("prev_month_value")
        ).otherwise(None)
    )
    
    # Clean up temporary columns
    df = df.drop("_month", "_year", "_quarter")
    
    logger.info("  Added: same_month_last_year, yoy_growth_rate, qoq_growth_rate, mom_growth_rate")
    logger.info("Temporal hierarchy features complete!")
    
    return df


def add_rolling_multiscale_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    windows: List[int] = [3, 6, 12, 24]
) -> DataFrame:
    """
    Add rolling statistics at multiple window sizes.
    
    This provides a smooth transition between scales and captures
    patterns at various frequencies.
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier column
        windows: List of rolling window sizes (in periods)
    
    Returns:
        DataFrame with rolling multiscale features
    """
    logger.info(f"Adding rolling multiscale features with windows: {windows}")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    for window_size in windows:
        # Define rolling window
        w_rolling = w.rowsBetween(-(window_size - 1), 0)
        
        # Rolling mean
        df = df.withColumn(
            f"rolling_mean_{window_size}",
            spark_avg(col(value_col)).over(w_rolling)
        )
        
        # Rolling standard deviation
        df = df.withColumn(
            f"rolling_std_{window_size}",
            stddev_samp(col(value_col)).over(w_rolling)
        )
        
        # Rolling min
        df = df.withColumn(
            f"rolling_min_{window_size}",
            spark_min(col(value_col)).over(w_rolling)
        )
        
        # Rolling max
        df = df.withColumn(
            f"rolling_max_{window_size}",
            spark_max(col(value_col)).over(w_rolling)
        )
        
        # Rolling range (max - min)
        df = df.withColumn(
            f"rolling_range_{window_size}",
            col(f"rolling_max_{window_size}") - col(f"rolling_min_{window_size}")
        )
        
        # Rolling coefficient of variation
        df = df.withColumn(
            f"rolling_cv_{window_size}",
            when(
                col(f"rolling_mean_{window_size}") != 0,
                col(f"rolling_std_{window_size}") / col(f"rolling_mean_{window_size}")
            ).otherwise(lit(0.0))
        )
        
        logger.info(f"  Added rolling features for window={window_size}")
    
    logger.info("Rolling multiscale features complete!")
    return df


def add_scale_decomposition_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str
) -> DataFrame:
    """
    Add features that decompose the signal into multiple scale components.
    
    Uses difference between rolling means of different windows to extract
    components at different frequencies (similar to wavelet decomposition).
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier column
    
    Returns:
        DataFrame with scale decomposition features
    """
    logger.info("Adding scale decomposition features")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    # Ensure we have rolling means at different scales
    windows = [3, 6, 12]
    for window_size in windows:
        w_rolling = w.rowsBetween(-(window_size - 1), 0)
        col_name = f"_rolling_mean_{window_size}"
        if col_name not in df.columns:
            df = df.withColumn(
                col_name,
                spark_avg(col(value_col)).over(w_rolling)
            )
    
    # High-frequency component: value - short-term mean
    df = df.withColumn(
        "high_freq_component",
        col(value_col) - col("_rolling_mean_3")
    )
    
    # Medium-frequency component: short-term mean - medium-term mean
    df = df.withColumn(
        "mid_freq_component",
        col("_rolling_mean_3") - col("_rolling_mean_6")
    )
    
    # Low-frequency component: medium-term mean - long-term mean
    df = df.withColumn(
        "low_freq_component",
        col("_rolling_mean_6") - col("_rolling_mean_12")
    )
    
    # Trend component: long-term mean
    df = df.withColumn(
        "trend_component",
        col("_rolling_mean_12")
    )
    
    # Signal-to-noise ratio (trend / high-freq component)
    df = df.withColumn(
        "signal_to_noise",
        when(
            col("high_freq_component") != 0,
            col("trend_component") / col("high_freq_component")
        ).otherwise(lit(0.0))
    )
    
    # Drop temporary columns
    df = df.drop("_rolling_mean_3", "_rolling_mean_6", "_rolling_mean_12")
    
    logger.info("  Added: high_freq_component, mid_freq_component, low_freq_component, trend_component")
    logger.info("Scale decomposition features complete!")
    
    return df


def add_all_multiscale_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str
) -> DataFrame:
    """
    Convenience function to add all multiscale features.
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier column
    
    Returns:
        DataFrame with all multiscale features
    """
    logger.info("Adding ALL multiscale features...")
    
    # Add multiscale aggregations
    df = add_multiscale_aggregation_features(
        df, value_col, date_col, product_col,
        scales=['month', 'quarter']
    )
    
    # Add cross-scale features
    df = add_cross_scale_features(df, value_col, date_col, product_col)
    
    # Add temporal hierarchy
    df = add_temporal_hierarchy_features(df, value_col, date_col, product_col)
    
    # Add rolling multiscale
    df = add_rolling_multiscale_features(
        df, value_col, date_col, product_col,
        windows=[3, 6, 12]
    )
    
    # Add scale decomposition
    df = add_scale_decomposition_features(df, value_col, date_col, product_col)
    
    logger.info("ALL multiscale features complete!")
    return df
