"""
Seasonal Decomposition and Fourier Features Module

Implements time series decomposition and cyclical encoding:
- STL-inspired decomposition (trend, seasonal, residual)
- Fourier features for cyclical patterns
- Extended lag features with autocorrelation analysis
- Seasonal strength metrics

These features help models capture:
- Multiple seasonality patterns (weekly, monthly, yearly)
- Trend changes and level shifts
- Irregular components and outliers
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, lit, when, coalesce,
    avg as spark_avg, sum as spark_sum,
    stddev_samp, variance,
    lag, lead, first, last,
    sin, cos, month, year, dayofyear, weekofyear,
    row_number, dense_rank,
    pow as spark_pow, sqrt, abs as spark_abs
)
from pyspark.sql.window import Window
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mathematical constants for Fourier transforms
_TWO_PI = 6.283185307179586  # 2 * pi


def add_fourier_features(
    df: DataFrame,
    date_col: str,
    periods: List[int] = [12, 6, 4, 3],
    n_harmonics: int = 2
) -> DataFrame:
    """
    Add Fourier features for cyclical time encoding.
    
    Fourier features encode cyclical patterns using sin/cos transformations:
    - Period 12: Annual seasonality (for monthly data)
    - Period 6: Semi-annual patterns
    - Period 4: Quarterly patterns
    - Period 3: Trimester patterns
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        periods: List of periods to encode (in number of time steps)
        n_harmonics: Number of Fourier harmonics per period
    
    Returns:
        DataFrame with Fourier features
    
    Example:
        df = add_fourier_features(
            df,
            date_col='MonthEndDate',
            periods=[12, 6, 4],
            n_harmonics=2
        )
    """
    logger.info(f"Adding Fourier features for periods: {periods}, harmonics: {n_harmonics}")
    
    # Get month number for Fourier encoding
    df = df.withColumn("_month_num", month(col(date_col)))
    
    for period in periods:
        for k in range(1, n_harmonics + 1):
            # Angular frequency: 2*pi*k / period
            freq = lit(_TWO_PI * k / period)
            
            # Sin and cos features for this harmonic
            df = df.withColumn(
                f"fourier_sin_p{period}_h{k}",
                sin(col("_month_num") * freq)
            )
            df = df.withColumn(
                f"fourier_cos_p{period}_h{k}",
                cos(col("_month_num") * freq)
            )
            
            logger.info(f"  Added: fourier_sin_p{period}_h{k}, fourier_cos_p{period}_h{k}")
    
    # Drop temporary column
    df = df.drop("_month_num")
    
    # Add day-of-year based Fourier features for finer granularity if date column exists
    df = df.withColumn("_day_of_year", dayofyear(col(date_col)))
    
    # Annual cycle with daily granularity (365-day period)
    df = df.withColumn(
        "fourier_sin_annual",
        sin(col("_day_of_year") * lit(_TWO_PI / 365.25))
    )
    df = df.withColumn(
        "fourier_cos_annual",
        cos(col("_day_of_year") * lit(_TWO_PI / 365.25))
    )
    
    # Semi-annual cycle
    df = df.withColumn(
        "fourier_sin_semiannual",
        sin(col("_day_of_year") * lit(2 * _TWO_PI / 365.25))
    )
    df = df.withColumn(
        "fourier_cos_semiannual",
        cos(col("_day_of_year") * lit(2 * _TWO_PI / 365.25))
    )
    
    df = df.drop("_day_of_year")
    
    logger.info("Fourier features complete!")
    return df


def add_extended_lag_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    lag_range: Tuple[int, int] = (6, 24)
) -> DataFrame:
    """
    Add extended lag features beyond the basic lag_1 to lag_5.
    
    Extended lags capture:
    - Same period patterns (lag_6, lag_12, lag_24 for semi-annual/annual)
    - Intermediate patterns (lag_8, lag_10, lag_18)
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier column
        lag_range: Tuple of (start_lag, end_lag) to generate
    
    Returns:
        DataFrame with extended lag features
    """
    start_lag, end_lag = lag_range
    logger.info(f"Adding extended lag features from lag_{start_lag} to lag_{end_lag}")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    # Key lags for monthly data
    important_lags = [6, 9, 12, 18, 24]
    
    for i in range(start_lag, end_lag + 1):
        # Only add important lags + every 3rd lag
        if i in important_lags or i % 3 == 0:
            col_name = f"lag_{i}"
            if col_name not in df.columns:
                df = df.withColumn(
                    col_name,
                    lag(col(value_col), i).over(w)
                )
                logger.info(f"  Added: {col_name}")
    
    logger.info("Extended lag features complete!")
    return df


def add_seasonal_lag_differences(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str
) -> DataFrame:
    """
    Add seasonal differences (year-over-year, quarter-over-quarter changes).
    
    These capture:
    - Year-over-year changes (current - 12 months ago)
    - Quarter-over-quarter changes (current - 3 months ago)
    - Semi-annual changes (current - 6 months ago)
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier column
    
    Returns:
        DataFrame with seasonal difference features
    """
    logger.info("Adding seasonal lag difference features")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    # Ensure lags exist
    for lag_period in [3, 6, 12]:
        lag_col = f"lag_{lag_period}"
        if lag_col not in df.columns:
            df = df.withColumn(
                lag_col,
                lag(col(value_col), lag_period).over(w)
            )
    
    # Quarterly difference
    df = df.withColumn(
        "seasonal_diff_3m",
        col(value_col) - col("lag_3")
    )
    
    # Semi-annual difference
    df = df.withColumn(
        "seasonal_diff_6m",
        col(value_col) - col("lag_6")
    )
    
    # Annual difference
    df = df.withColumn(
        "seasonal_diff_12m",
        col(value_col) - col("lag_12")
    )
    
    # Seasonal ratio features
    df = df.withColumn(
        "seasonal_ratio_3m",
        when(
            (col("lag_3").isNotNull()) & (col("lag_3") != 0),
            col(value_col) / col("lag_3")
        ).otherwise(lit(1.0))
    )
    
    df = df.withColumn(
        "seasonal_ratio_12m",
        when(
            (col("lag_12").isNotNull()) & (col("lag_12") != 0),
            col(value_col) / col("lag_12")
        ).otherwise(lit(1.0))
    )
    
    logger.info("  Added: seasonal_diff_3m, seasonal_diff_6m, seasonal_diff_12m, seasonal_ratio_3m, seasonal_ratio_12m")
    logger.info("Seasonal lag difference features complete!")
    
    return df


def add_stl_decomposition_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    seasonal_period: int = 12,
    trend_window: int = 7
) -> DataFrame:
    """
    Add STL-inspired decomposition features using PySpark.
    
    Decomposes time series into:
    - Trend component (long-term direction)
    - Seasonal component (periodic patterns)
    - Residual component (irregular/noise)
    
    Note: This is an approximation of STL using rolling windows,
    as PySpark doesn't have native STL support.
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier column
        seasonal_period: Period of seasonality (12 for monthly with yearly pattern)
        trend_window: Window size for trend smoothing (should be odd)
    
    Returns:
        DataFrame with STL-inspired decomposition features
    """
    logger.info(f"Adding STL-inspired decomposition (period={seasonal_period}, trend_window={trend_window})")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    
    # 1. TREND COMPONENT: Centered moving average
    half_window = trend_window // 2
    w_trend = w.rowsBetween(-half_window, half_window)
    
    df = df.withColumn(
        "stl_trend",
        spark_avg(col(value_col)).over(w_trend)
    )
    
    # 2. DETRENDED SERIES
    df = df.withColumn(
        "_detrended",
        col(value_col) - col("stl_trend")
    )
    
    # 3. SEASONAL COMPONENT: Average detrended value for each seasonal position
    # For monthly data, this is the average for each month across years
    df = df.withColumn("_seasonal_position", month(col(date_col)))
    
    w_seasonal = Window.partitionBy(product_col, "_seasonal_position")
    df = df.withColumn(
        "stl_seasonal",
        spark_avg(col("_detrended")).over(w_seasonal)
    )
    
    # 4. RESIDUAL COMPONENT: What's left after removing trend and seasonal
    df = df.withColumn(
        "stl_residual",
        col(value_col) - col("stl_trend") - col("stl_seasonal")
    )
    
    # 5. Additional derived features
    # Trend strength: 1 - Var(residual) / Var(trend + residual)
    w_product = Window.partitionBy(product_col)
    
    df = df.withColumn(
        "_var_residual",
        variance(col("stl_residual")).over(w_product)
    )
    df = df.withColumn(
        "_var_trend_residual",
        variance(col("stl_trend") + col("stl_residual")).over(w_product)
    )
    df = df.withColumn(
        "trend_strength",
        when(
            col("_var_trend_residual") != 0,
            lit(1.0) - col("_var_residual") / col("_var_trend_residual")
        ).otherwise(lit(0.0))
    )
    
    # Seasonal strength: 1 - Var(residual) / Var(seasonal + residual)
    df = df.withColumn(
        "_var_seasonal_residual",
        variance(col("stl_seasonal") + col("stl_residual")).over(w_product)
    )
    df = df.withColumn(
        "seasonal_strength",
        when(
            col("_var_seasonal_residual") != 0,
            lit(1.0) - col("_var_residual") / col("_var_seasonal_residual")
        ).otherwise(lit(0.0))
    )
    
    # Signal-to-noise ratio
    df = df.withColumn(
        "stl_snr",
        when(
            col("_var_residual") != 0,
            (variance(col("stl_trend")).over(w_product) + 
             variance(col("stl_seasonal")).over(w_product)) / col("_var_residual")
        ).otherwise(lit(0.0))
    )
    
    # Clean up temporary columns
    df = df.drop(
        "_detrended", "_seasonal_position",
        "_var_residual", "_var_trend_residual", "_var_seasonal_residual"
    )
    
    logger.info("  Added: stl_trend, stl_seasonal, stl_residual, trend_strength, seasonal_strength, stl_snr")
    logger.info("STL decomposition features complete!")
    
    return df


def add_autocorrelation_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str,
    lags: List[int] = [1, 2, 3, 6, 12]
) -> DataFrame:
    """
    Add autocorrelation-based features.
    
    Autocorrelation measures how correlated the time series is with
    lagged versions of itself. This helps identify:
    - Persistence in demand patterns
    - Seasonal autocorrelation (high ACF at lag 12 for monthly data)
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier column
        lags: List of lags to compute autocorrelation for
    
    Returns:
        DataFrame with autocorrelation features
    """
    logger.info(f"Adding autocorrelation features for lags: {lags}")
    
    w = Window.partitionBy(product_col).orderBy(date_col)
    w_product = Window.partitionBy(product_col)
    
    # Mean and std of the series
    df = df.withColumn("_series_mean", spark_avg(col(value_col)).over(w_product))
    df = df.withColumn("_series_std", stddev_samp(col(value_col)).over(w_product))
    
    for lag_val in lags:
        lag_col = f"lag_{lag_val}"
        
        # Ensure lag column exists
        if lag_col not in df.columns:
            df = df.withColumn(lag_col, lag(col(value_col), lag_val).over(w))
        
        # Compute product of centered values
        df = df.withColumn(
            "_centered_product",
            (col(value_col) - col("_series_mean")) * 
            (col(lag_col) - col("_series_mean"))
        )
        
        # Autocorrelation approximation (normalized covariance)
        df = df.withColumn(
            f"acf_lag_{lag_val}",
            when(
                col("_series_std") != 0,
                spark_avg(col("_centered_product")).over(w_product) / 
                spark_pow(col("_series_std"), lit(2.0))
            ).otherwise(lit(0.0))
        )
        
        df = df.drop("_centered_product")
        logger.info(f"  Added: acf_lag_{lag_val}")
    
    # Partial autocorrelation-like features (simplified)
    # High ACF at lag 12 but low at lag 6 suggests yearly but not semi-annual seasonality
    if 'acf_lag_12' in df.columns and 'acf_lag_6' in df.columns:
        df = df.withColumn(
            "acf_yearly_vs_semiannual",
            col("acf_lag_12") - col("acf_lag_6")
        )
        logger.info("  Added: acf_yearly_vs_semiannual")
    
    # Clean up
    df = df.drop("_series_mean", "_series_std")
    
    logger.info("Autocorrelation features complete!")
    return df


def add_all_seasonal_features(
    df: DataFrame,
    value_col: str,
    date_col: str,
    product_col: str
) -> DataFrame:
    """
    Convenience function to add all seasonal decomposition features.
    
    Args:
        df: Input DataFrame
        value_col: Value column name
        date_col: Date column name
        product_col: Product identifier column
    
    Returns:
        DataFrame with all seasonal features
    """
    logger.info("Adding ALL seasonal decomposition features...")
    
    # Fourier features
    df = add_fourier_features(df, date_col, periods=[12, 6, 4], n_harmonics=2)
    
    # Extended lags
    df = add_extended_lag_features(df, value_col, date_col, product_col, lag_range=(6, 24))
    
    # Seasonal differences
    df = add_seasonal_lag_differences(df, value_col, date_col, product_col)
    
    # STL decomposition
    df = add_stl_decomposition_features(df, value_col, date_col, product_col)
    
    # Autocorrelation
    df = add_autocorrelation_features(df, value_col, date_col, product_col)
    
    logger.info("ALL seasonal decomposition features complete!")
    return df
